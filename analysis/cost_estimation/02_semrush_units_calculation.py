import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Get the project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Path to the filtered organizations file
DATA_DIR = PROJECT_ROOT / "data/processed"
ORG_FILE = DATA_DIR / "orgs_2012_2019_survived.csv"

def calculate_semrush_units():
    """Calculate SEMrush API units needed based on their pricing model."""
    
    # Read the filtered data
    print("Reading filtered organizations data...")
    df = pd.read_csv(ORG_FILE)
    
    # Convert date columns
    df['founded_on'] = pd.to_datetime(df['founded_on'], errors='coerce')
    df['closed_on'] = pd.to_datetime(df['closed_on'], errors='coerce')
    
    # Calculate years for each company
    current_year = datetime.now().year
    
    def calculate_years_active(row):
        """Calculate how many years a company was active."""
        if pd.isna(row['founded_on']):
            return 0
        
        # If still active (no closed_on date), count up to current year
        if pd.isna(row['closed_on']):
            end_year = current_year
        else:
            end_year = row['closed_on'].year
        
        # Count years from founding to end (inclusive)
        years_active = end_year - row['founded_on'].year + 1
        
        # Ensure at least 1 year (since we filtered for 12+ months survival)
        return max(1, years_active)
    
    # Calculate years active for each company
    df['years_active'] = df.apply(calculate_years_active, axis=1)
    
    # Calculate total company-years
    total_company_years = df['years_active'].sum()
    
    print(f"\n=== SEMRUSH UNITS CALCULATION ===")
    print(f"Total companies: {len(df):,}")
    print(f"Total company-years: {total_company_years:,}")
    
    # SEMrush pricing scenarios based on their example
    # From their example: 1,000 keywords * 50 units for historical data * 100 domains = 5,000,000 units
    
    scenarios = {
        'Conservative - 10 keywords per company': {
            'keywords_per_company': 10,
            'units_per_keyword_historical': 50,
            'description': 'Assuming 10 keywords per company (very conservative)'
        },
        'Moderate - 50 keywords per company': {
            'keywords_per_company': 50,
            'units_per_keyword_historical': 50,
            'description': 'Assuming 50 keywords per company (moderate estimate)'
        },
        'High - 100 keywords per company': {
            'keywords_per_company': 100,
            'units_per_keyword_historical': 50,
            'description': 'Assuming 100 keywords per company (high estimate)'
        },
        'Premium - 500 keywords per company': {
            'keywords_per_company': 500,
            'units_per_keyword_historical': 50,
            'description': 'Assuming 500 keywords per company (premium estimate)'
        }
    }
    
    print(f"\n=== UNITS CALCULATION BY SCENARIO ===")
    
    for scenario_name, params in scenarios.items():
        keywords_per_company = params['keywords_per_company']
        units_per_keyword = params['units_per_keyword_historical']
        
        # Calculate total units needed
        # Formula: companies * keywords_per_company * units_per_keyword * years_active
        total_units = len(df) * keywords_per_company * units_per_keyword * df['years_active'].mean()
        
        print(f"\n{scenario_name}:")
        print(f"  {params['description']}")
        print(f"  Total units needed: {total_units:,.0f}")
        
        # Convert to cost (assuming $1 per 10,000 units as a baseline)
        # Note: This is a rough estimate - check actual SEMrush pricing
        cost_per_10k_units = 1.00  # Adjust this based on actual pricing
        estimated_cost = (total_units / 10000) * cost_per_10k_units
        
        print(f"  Estimated cost (at $1 per 10k units): ${estimated_cost:,.2f}")
    
    # Alternative: Monthly data (12x more data points)
    print(f"\n=== MONTHLY DATA SCENARIOS ===")
    
    for scenario_name, params in scenarios.items():
        keywords_per_company = params['keywords_per_company']
        units_per_keyword = params['units_per_keyword_historical']
        
        # Monthly data: multiply by 12 for monthly granularity
        total_units_monthly = len(df) * keywords_per_company * units_per_keyword * df['years_active'].mean() * 12
        
        cost_per_10k_units = 1.00
        estimated_cost_monthly = (total_units_monthly / 10000) * cost_per_10k_units
        
        print(f"\n{scenario_name} (Monthly):")
        print(f"  Total units needed: {total_units_monthly:,.0f}")
        print(f"  Estimated cost: ${estimated_cost_monthly:,.2f}")
    
    # Alternative: Quarterly data (4x more data points)
    print(f"\n=== QUARTERLY DATA SCENARIOS ===")
    
    for scenario_name, params in scenarios.items():
        keywords_per_company = params['keywords_per_company']
        units_per_keyword = params['units_per_keyword_historical']
        
        # Quarterly data: multiply by 4 for quarterly granularity
        total_units_quarterly = len(df) * keywords_per_company * units_per_keyword * df['years_active'].mean() * 4
        
        cost_per_10k_units = 1.00
        estimated_cost_quarterly = (total_units_quarterly / 10000) * cost_per_10k_units
        
        print(f"\n{scenario_name} (Quarterly):")
        print(f"  Total units needed: {total_units_quarterly:,.0f}")
        print(f"  Estimated cost: ${estimated_cost_quarterly:,.2f}")
    
    return df, scenarios

def calculate_by_founding_year():
    """Calculate units needed by founding year to see distribution."""
    
    df = pd.read_csv(ORG_FILE)
    df['founded_on'] = pd.to_datetime(df['founded_on'], errors='coerce')
    df['closed_on'] = pd.to_datetime(df['closed_on'], errors='coerce')
    
    current_year = datetime.now().year
    
    def calculate_years_active(row):
        if pd.isna(row['founded_on']):
            return 0
        if pd.isna(row['closed_on']):
            end_year = current_year
        else:
            end_year = row['closed_on'].year
        years_active = end_year - row['founded_on'].year + 1
        return max(1, years_active)
    
    df['years_active'] = df.apply(calculate_years_active, axis=1)
    
    # Group by founding year
    yearly_stats = df.groupby(df['founded_on'].dt.year).agg({
        'uuid': 'count',
        'years_active': ['sum', 'mean']
    }).round(2)
    
    yearly_stats.columns = ['Companies', 'Total_Years', 'Avg_Years']
    
    # Calculate units for moderate scenario (50 keywords per company)
    keywords_per_company = 50
    units_per_keyword = 50
    
    yearly_stats['Units_Needed'] = (
        yearly_stats['Companies'] * 
        keywords_per_company * 
        units_per_keyword * 
        yearly_stats['Avg_Years']
    )
    
    print(f"\n=== UNITS BY FOUNDING YEAR (50 keywords per company) ===")
    print(yearly_stats)
    
    return yearly_stats

def main():
    """Main function to run the units calculation."""
    
    # Calculate units for different scenarios
    df, scenarios = calculate_semrush_units()
    
    # Calculate by founding year
    yearly_stats = calculate_by_founding_year()
    
    # Save detailed breakdown
    output_file = PROJECT_ROOT / "analysis/cost_estimation/semrush_units_breakdown.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    yearly_stats.to_csv(output_file)
    
    print(f"\nDetailed breakdown saved to: {output_file}")

if __name__ == "__main__":
    main() 