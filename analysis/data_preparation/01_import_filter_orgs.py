import pandas as pd
from pathlib import Path

# Toggle for sampling (set to True to only output first 100 rows)
SAMPLE = True  # Set to False for full output

# Get the project root (the directory containing this script's parent directories)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Path to the organizations.csv file
DATA_DIR = PROJECT_ROOT / "data/250612_cb_data"
ORG_FILE = DATA_DIR / "organizations.csv"

# Output path
OUTPUT_DIR = PROJECT_ROOT / "data/processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# Update output filename to reflect new period and sample
output_name = "orgs_2012_2025_sample.csv" if SAMPLE else "orgs_2012_2025.csv"
OUTPUT_FILE = OUTPUT_DIR / output_name

# Columns to keep
COLUMNS = [
    "uuid",
    "name",
    "founded_on",
    "closed_on",
    "total_funding_usd",
    "employee_count",
    "homepage_url",
    "status",
    "short_description",
    "category_list"
]

# Read in chunks to handle large file
chunks = []
for chunk in pd.read_csv(ORG_FILE, usecols=lambda c: c in COLUMNS, chunksize=100_000, low_memory=False):
    # Convert founded_on to datetime
    chunk["founded_on"] = pd.to_datetime(chunk["founded_on"], errors="coerce")
    # Filter by founded_on year (2012-2025)
    mask = chunk["founded_on"].dt.year.between(2012, 2025, inclusive="both")
    filtered = chunk.loc[mask, COLUMNS]
    chunks.append(filtered)

# Concatenate all filtered chunks
result = pd.concat(chunks, ignore_index=True)

# If sampling, only keep first 100 rows
if SAMPLE:
    result = result.head(100)

# Save to CSV
result.to_csv(OUTPUT_FILE, index=False)

print(f"Filtered organizations saved to {OUTPUT_FILE}") 