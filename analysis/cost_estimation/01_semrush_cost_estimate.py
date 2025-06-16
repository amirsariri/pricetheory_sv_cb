import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Get the project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Path to the filtered organizations file
DATA_DIR = PROJECT_ROOT / "data/processed"
ORG_FILE = DATA_DIR / "orgs_2012_2019_survived.csv"

def calculate_company_years():
    """Calculate total company-years for cost estimation."""
    
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
    
    # Summary statistics
    print(f"\n=== COMPANY-YEARS CALCULATION ===")
    print(f"Total companies: {len(df):,}")
    print(f"Total company-years: {total_company_years:,}")
    print(f"Average years per company: {df['years_active'].mean():.1f}")
    print(f"Median years per company: {df['years_active'].median():.1f}")
    print(f"Min years per company: {df['years_active'].min()}")
    print(f"Max years per company: {df['years_active'].max()}")
    
    # Distribution by founding year
    print(f"\n=== DISTRIBUTION BY FOUNDING YEAR ===")
    year_dist = df.groupby(df['founded_on'].dt.year)['years_active'].agg(['count', 'sum', 'mean'])
    year_dist.columns = ['Companies', 'Company-Years', 'Avg Years per Company']
    print(year_dist)
    
    return df, total_company_years

def estimate_semrush_costs(total_company_years):
    """Estimate SEMrush API costs based on different scenarios."""
    
    print(f"\n=== SEMRUSH COST ESTIMATION ===")
    print(f"Total company-years to query: {total_company_years:,}")
    
    # SEMrush pricing scenarios (these are estimates - check current pricing)
    pricing_scenarios = {
        'Conservative': 0.10,  # $0.10 per API call
        'Moderate': 0.25,      # $0.25 per API call  
        'High': 0.50,          # $0.50 per API call
        'Premium': 1.00        # $1.00 per API call
    }
    
    print(f"\nCost estimates (assuming 1 API call per company-year):")
    for scenario, price_per_call in pricing_scenarios.items():
        total_cost = total_company_years * price_per_call
        print(f"{scenario}: ${total_cost:,.2f}")
    
    # Alternative scenarios with different data granularity
    print(f"\n=== ALTERNATIVE SCENARIOS ===")
    
    # Monthly data (12x more calls)
    monthly_calls = total_company_years * 12
    print(f"Monthly data calls: {monthly_calls:,}")
    for scenario, price_per_call in pricing_scenarios.items():
        total_cost = monthly_calls * price_per_call
        print(f"{scenario} (monthly): ${total_cost:,.2f}")
    
    # Quarterly data (4x more calls)
    quarterly_calls = total_company_years * 4
    print(f"\nQuarterly data calls: {quarterly_calls:,}")
    for scenario, price_per_call in pricing_scenarios.items():
        total_cost = quarterly_calls * price_per_call
        print(f"{scenario} (quarterly): ${total_cost:,.2f}")

def main():
    """Main function to run the cost estimation."""
    
    # Calculate company-years
    df, total_company_years = calculate_company_years()
    
    # Estimate costs
    estimate_semrush_costs(total_company_years)
    
    # Save detailed breakdown
    output_file = PROJECT_ROOT / "analysis/cost_estimation/company_years_breakdown.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Create summary by founding year
    summary = df.groupby(df['founded_on'].dt.year).agg({
        'uuid': 'count',
        'years_active': ['sum', 'mean', 'median']
    }).round(2)
    
    summary.columns = ['Companies', 'Company_Years', 'Avg_Years', 'Median_Years']
    summary.to_csv(output_file)
    
    print(f"\nDetailed breakdown saved to: {output_file}")

if __name__ == "__main__":
    main() 