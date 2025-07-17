import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# Toggle for sampling (set to True to only output first 100 rows)
SAMPLE = False  # Set to False for full output

# Get the project root (the directory containing this script's parent directories)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Path to the organizations.csv file
DATA_DIR = PROJECT_ROOT / "data/250612_cb_data"
ORG_FILE = DATA_DIR / "organizations.csv"

# Output path
OUTPUT_DIR = PROJECT_ROOT / "data/processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# Update output filename to reflect new period and sample
output_name = "orgs_2012_2018_survived_sample.csv" if SAMPLE else "orgs_2012_2018_survived.csv"
OUTPUT_FILE = OUTPUT_DIR / output_name

def survived_at_least_12_months(row):
    """Check if company survived at least 12 months."""
    if pd.isna(row['founded_on']):
        return False
    
    # If company is still active (no closed_on date), consider it survived
    if pd.isna(row['closed_on']):
        return True
    
    # Calculate survival time using year difference to avoid overflow
    founded_year = row['founded_on'].year
    closed_year = row['closed_on'].year
    
    # If more than 1 year difference, definitely survived
    if closed_year - founded_year > 1:
        return True
    
    # If same year, check if at least 12 months passed
    if closed_year == founded_year:
        return False  # Same year means less than 12 months
    
    # If 1 year difference, check the months
    if closed_year - founded_year == 1:
        founded_month = row['founded_on'].month
        closed_month = row['closed_on'].month
        
        # If closed month is after founded month, survived at least 12 months
        if closed_month > founded_month:
            return True
        elif closed_month == founded_month:
            # Check days
            return row['closed_on'].day >= row['founded_on'].day
        
    return False

# Read in chunks to handle large file
chunks = []
for chunk in pd.read_csv(ORG_FILE, chunksize=100_000, low_memory=False):
    # Convert date columns to datetime
    chunk["founded_on"] = pd.to_datetime(chunk["founded_on"], errors="coerce")
    chunk["closed_on"] = pd.to_datetime(chunk["closed_on"], errors="coerce")
    
    # Apply filters:
    # 1. Founded between 2012-2018
    founded_mask = chunk["founded_on"].dt.year.between(2012, 2018, inclusive="both")
    
    # 2. Has homepage URL (not null and not empty)
    homepage_mask = chunk["homepage_url"].notna() & (chunk["homepage_url"] != "")
    
    # 3. Survived at least 12 months
    survival_mask = chunk.apply(survived_at_least_12_months, axis=1)

    # 4. US-based startups
    us_mask = chunk["country_code"] == "USA"

    # 6. Primary role is 'company'
    role_mask = chunk["primary_role"] == "company"
    
    # 5. Has postal code (not null and not empty)
    postal_mask = chunk["postal_code"].notna() & (chunk["postal_code"] != "")

    # Combine all filters (removed funding_mask)
    combined_mask = founded_mask & homepage_mask & survival_mask & us_mask & role_mask & postal_mask
    
    filtered = chunk.loc[combined_mask]
    chunks.append(filtered)

# Concatenate all filtered chunks
result = pd.concat(chunks, ignore_index=True)

# If sampling, only keep first 100 rows
if SAMPLE:
    result = result.head(100)

# After filtering and before saving, merge in the description field from organization_descriptions.csv using uuid
# Path to the descriptions file
DESC_FILE = DATA_DIR / "organization_descriptions.csv"
if DESC_FILE.exists():
    desc_df = pd.read_csv(DESC_FILE, usecols=["uuid", "description"])
    if "uuid" in result.columns:
        result = result.merge(desc_df, on="uuid", how="left")
    else:
        print("Warning: 'uuid' column not found in filtered results, cannot merge descriptions.")
else:
    print(f"Warning: Descriptions file not found: {DESC_FILE}")

# Exclude companies with a blank or missing description
if "description" in result.columns:
    result = result[result["description"].notna() & (result["description"].str.strip() != "")]

# Remove companies with less than 150 characters in the description
if "description" in result.columns:
    result = result[result["description"].str.len() >= 50]

# After filtering and before saving, and after excluding blank descriptions
if "description" in result.columns:
    pass # Removed histogram plotting

# Remove companies with 'Consulting' in the category_list (case-insensitive)
if "category_list" in result.columns:
    result = result[~result["category_list"].str.contains("Consulting", case=False, na=False)]
    # Remove companies with 'Rental Company' in the category_list (case-insensitive)
    result = result[~result["category_list"].str.contains("Rental Property", case=False, na=False)]
    # Remove companies with 'Property Management' in the category_list (case-insensitive)
    result = result[~result["category_list"].str.contains("Property Management", case=False, na=False)]

# Update 'closed_update': 1 if 'updated_at' is before 2021-01-01 OR status is 'closed'
if "updated_at" in result.columns:
    result["updated_at"] = pd.to_datetime(result["updated_at"], errors="coerce")
    closed_update_mask = (result["updated_at"] < pd.Timestamp("2021-01-01"))
    if "status" in result.columns:
        closed_update_mask = closed_update_mask | (result["status"].str.lower() == "closed")
    result["closed_update"] = closed_update_mask.astype(int)

# Save to CSV
result.to_csv(OUTPUT_FILE, index=False)

print(f"Filtered organizations saved to {OUTPUT_FILE}")
print(f"Total organizations found: {len(result)}")

# Print summary statistics
if len(result) > 0:
    print(f"\nSummary:")
    print(f"Founded years range: {result['founded_on'].dt.year.min()} - {result['founded_on'].dt.year.max()}")
    print(f"Average founding year: {result['founded_on'].dt.year.mean():.1f}")
    print(f"Companies with funding data: {result['total_funding_usd'].notna().sum()}")
    print(f"Companies still active: {(result['closed_on'].isna()).sum()}") 
    
    # Additional summary statistics
    total = len(result)
    closed = result['closed_on'].notna().sum()
    fraction_closed = closed / total if total > 0 else float('nan')
    
    with_funding = result['total_funding_usd'].notna() & (result['total_funding_usd'] > 0)
    closed_with_funding = result.loc[with_funding, 'closed_on'].notna().sum()
    total_with_funding = with_funding.sum()
    fraction_closed_with_funding = closed_with_funding / total_with_funding if total_with_funding > 0 else float('nan')
    
    print(f"Fraction of total that is closed: {fraction_closed:.2%} ({closed}/{total})")
    print(f"Fraction of those that raised money that closed: {fraction_closed_with_funding:.2%} ({closed_with_funding}/{total_with_funding})")
    
    if 'status' in result.columns:
        print("\nPercent share of companies in each 'status' category:")
        status_counts = result['status'].value_counts(normalize=True) * 100
        for status, pct in status_counts.items():
            print(f"  {status}: {pct:.2f}%")
    else:
        print("No 'status' column found in the data.") 

# Print fraction of companies with closed_update == 1
if "closed_update" in result.columns:
    closed_update_count = result["closed_update"].sum()
    closed_update_fraction = closed_update_count / len(result) if len(result) > 0 else float('nan')
    print(f"Fraction of companies with closed_update=1: {closed_update_fraction:.2%} ({closed_update_count}/{len(result)})") 