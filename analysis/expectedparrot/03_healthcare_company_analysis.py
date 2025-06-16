"""
Healthcare Company Analysis with ExpectedParrot
Analyzing Crunchbase healthcare companies using EDSL
"""

import edsl
from edsl import Question, Survey, Agent, Scenario, Model
import pandas as pd
import numpy as np
from pathlib import Path
import random

def filter_healthcare_companies():
    """Filter healthcare companies based on criteria."""
    
    print("=== Step 1: Filtering Healthcare Companies ===\n")
    
    # Load organizations data
    org_file = Path("data/250612_cb_data/organizations.csv")
    if not org_file.exists():
        print(f"❌ Organizations file not found: {org_file}")
        return None
    
    print("Loading organizations data...")
    df_orgs = pd.read_csv(org_file)
    print(f"Loaded {len(df_orgs):,} organizations\n")
    
    # Apply filters
    print("Applying filters...")
    
    # 1. Health Care in category_groups_list
    health_mask = df_orgs['category_groups_list'].str.contains('Health Care', na=False)
    print(f"Health Care companies: {health_mask.sum():,}")
    
    # 2. Founded between 2012-2018
    df_orgs['founded_on'] = pd.to_datetime(df_orgs['founded_on'], errors='coerce')
    founded_mask = (df_orgs['founded_on'].dt.year >= 2012) & (df_orgs['founded_on'].dt.year <= 2018)
    print(f"Founded 2012-2018: {founded_mask.sum():,}")
    
    # 3. Survived at least one year
    df_orgs['closed_on'] = pd.to_datetime(df_orgs['closed_on'], errors='coerce')
    
    def survived_at_least_1_year(row):
        """Check if company survived at least 1 year."""
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
        
        # If 1 year difference, check if at least 12 months passed
        if closed_year - founded_year == 1:
            founded_month = row['founded_on'].month
            closed_month = row['closed_on'].month
            return closed_month >= founded_month
        
        return False
    
    # Apply the survival function
    survived_mask = df_orgs.apply(survived_at_least_1_year, axis=1)
    print(f"Survived at least 1 year: {survived_mask.sum():,}")
    
    # 4. Raised at least USD100,000
    funding_mask = df_orgs['total_funding_usd'] >= 100000
    print(f"Raised at least $100K: {funding_mask.sum():,}")
    
    # 5. Primary role is 'company'
    role_mask = df_orgs['primary_role'] == 'company'
    print(f"Primary role is company: {role_mask.sum():,}")
    
    # Combine all filters
    final_mask = health_mask & founded_mask & survived_mask & funding_mask & role_mask
    filtered_df = df_orgs[final_mask].copy()
    
    print(f"\n✅ Total companies meeting all criteria: {len(filtered_df):,}")
    
    return filtered_df

def get_company_descriptions(filtered_companies):
    """Get descriptions for the filtered companies."""
    
    print("\n=== Step 2: Getting Company Descriptions ===\n")
    
    # Load descriptions data
    desc_file = Path("data/250612_cb_data/organization_descriptions.csv")
    if not desc_file.exists():
        print(f"❌ Descriptions file not found: {desc_file}")
        return None
    
    print("Loading descriptions data...")
    df_desc = pd.read_csv(desc_file)
    print(f"Loaded {len(df_desc):,} descriptions\n")
    
    # Merge with filtered companies
    merged_df = filtered_companies.merge(
        df_desc[['uuid', 'description']], 
        on='uuid', 
        how='left'
    )
    
    # Remove companies without descriptions
    merged_df = merged_df.dropna(subset=['description'])
    print(f"Companies with descriptions: {len(merged_df):,}")
    
    return merged_df

def create_edsl_survey():
    """Create the EDSL survey for company analysis."""
    
    print("\n=== Step 3: Creating EDSL Survey ===\n")
    
    # Create agent
    print("Creating market analyst agent...")
    agent = Agent(traits = {
        "persona": "You are an experienced market analyst skilled in analyzing competitive landscape"
    })
    print(f"Agent created: {agent.traits['persona']}\n")
    
    # Create questions
    print("Creating survey questions...")
    
    question1 = Question(
        question_name = "main_customers",
        question_text = "Based on this company description: {{ scenario.company_description }}\n\nDescribe in 5 words or less the main customers this company aims to serve:",
        question_type = "free_text"
    )
    
    question2 = Question(
        question_name = "main_product",
        question_text = "Based on this company description: {{ scenario.company_description }}\n\nDescribe in 5 words or less the main product or service this company provides:",
        question_type = "free_text"
    )
    
    print("Questions created:")
    print(f"1. {question1.question_text}")
    print(f"2. {question2.question_text}\n")
    
    # Create survey
    survey = Survey(questions = [question1, question2])
    print(f"Survey created with {len(survey.questions)} questions\n")
    
    return survey, agent

def run_edsl_analysis(companies_df, survey, agent, sample_size=100):
    """Run EDSL analysis on the companies."""
    
    print(f"\n=== Step 4: Running EDSL Analysis ===\n")
    
    # Randomly select companies
    if len(companies_df) > sample_size:
        selected_companies = companies_df.sample(n=sample_size, random_state=42)
        print(f"Randomly selected {sample_size} companies from {len(companies_df):,} total")
    else:
        selected_companies = companies_df
        print(f"Using all {len(selected_companies)} companies (less than {sample_size})")
    
    print(f"Selected companies: {len(selected_companies)}\n")
    
    # Create scenarios
    print("Creating scenarios for each company...")
    scenarios = []
    for idx, row in selected_companies.iterrows():
        scenario = Scenario({
            "company_name": row['name'],
            "company_description": row['description'],
            "company_uuid": row['uuid']
        })
        scenarios.append(scenario)
    
    print(f"Created {len(scenarios)} scenarios\n")
    
    # Run survey
    print("Running EDSL survey...")
    try:
        model = Model("gpt-4o-mini")
        results = survey.by(agent).by(scenarios).by(model).run()
        
        print("✅ Survey completed successfully!")
        print(f"Results: {len(results)} responses\n")
        
        return results, selected_companies
        
    except Exception as e:
        print(f"❌ Error running survey: {e}")
        return None, None

def save_results(results, companies_df, output_file):
    """Save results to CSV."""
    
    print(f"\n=== Step 5: Saving Results ===\n")
    
    # Create output directory
    output_dir = Path("data/temp_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract results
    data_rows = []
    for i, result in enumerate(results):
        company_row = companies_df.iloc[i]
        
        data_row = {
            'uuid': company_row['uuid'],
            'company_name': company_row['name'],
            'founded_on': company_row['founded_on'],
            'total_funding_usd': company_row['total_funding_usd'],
            'category_list': company_row['category_list'],
            'main_customers': result.answer['main_customers'],
            'main_product': result.answer['main_product']
        }
        data_rows.append(data_row)
    
    # Create DataFrame and save
    results_df = pd.DataFrame(data_rows)
    results_df.to_csv(output_file, index=False)
    
    print(f"✅ Results saved to: {output_file}")
    print(f"📊 Total companies analyzed: {len(results_df)}")
    
    return results_df

def save_sample_companies(companies_df, output_file, sample_size=100):
    """Save sample companies to CSV for review."""
    
    print(f"\n=== Step 2.5: Creating Sample of {sample_size} Companies ===\n")
    
    # Create output directory
    output_dir = Path("data/temp_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Randomly select companies
    if len(companies_df) > sample_size:
        selected_companies = companies_df.sample(n=sample_size, random_state=42)
        print(f"Randomly selected {sample_size} companies from {len(companies_df):,} total")
    else:
        selected_companies = companies_df
        print(f"Using all {len(selected_companies)} companies (less than {sample_size})")
    
    # Save to CSV
    selected_companies.to_csv(output_file, index=False)
    
    print(f"✅ Sample companies saved to: {output_file}")
    print(f"📊 Sample size: {len(selected_companies)}")
    
    # Display sample info
    print(f"\n=== Sample Company Info ===\n")
    print(f"Founded years: {selected_companies['founded_on'].dt.year.min()} - {selected_companies['founded_on'].dt.year.max()}")
    print(f"Average funding: ${selected_companies['total_funding_usd'].mean():,.0f}")
    print(f"Companies still active: {(selected_companies['closed_on'].isna()).sum()}")
    
    # Show first few companies
    print(f"\nFirst 10 companies:")
    for i, (_, row) in enumerate(selected_companies.head(10).iterrows()):
        print(f"{i+1}. {row['name']} (Founded: {row['founded_on'].year}, Funding: ${row['total_funding_usd']:,.0f})")
    
    return selected_companies

def load_or_create_sample(companies_with_desc, sample_file, sample_size=100):
    """Load existing sample or create new one if it doesn't exist."""
    
    if sample_file.exists():
        print(f"📁 Loading existing sample from: {sample_file}")
        selected_companies = pd.read_csv(sample_file)
        print(f"✅ Loaded {len(selected_companies)} companies from existing sample")
        return selected_companies
    else:
        print(f"📁 Creating new sample of {sample_size} companies...")
        return save_sample_companies(companies_with_desc, sample_file, sample_size)

def run_multiple_models_analysis(companies_df, survey, agent, models_list):
    """Run EDSL analysis with multiple models for comparison."""
    
    print(f"\n=== Step 4: Running Multi-Model Analysis ===\n")
    
    # Use the first 3 companies
    edsl_companies = companies_df.head(3)
    print(f"Using first 3 companies for model comparison...")
    
    # Create scenarios
    print("Creating scenarios for each company...")
    scenarios = []
    for idx, row in edsl_companies.iterrows():
        scenario = Scenario({
            "company_name": row['name'],
            "company_description": row['description'],
            "company_uuid": row['uuid'],
            "company_index": idx
        })
        scenarios.append(scenario)
    
    print(f"Created {len(scenarios)} scenarios\n")
    
    # Run survey with each model
    all_results = {}
    
    for model_name in models_list:
        print(f"Running analysis with {model_name}...")
        try:
            model = Model(model_name)
            results = survey.by(agent).by(scenarios).by(model).run()
            
            print(f"✅ {model_name} completed successfully!")
            print(f"Results: {len(results)} responses\n")
            
            all_results[model_name] = results
            
        except Exception as e:
            print(f"❌ Error with {model_name}: {e}")
            all_results[model_name] = None
    
    return all_results, edsl_companies

def save_multi_model_results(all_results, companies_df, output_file):
    """Save results from multiple models to CSV."""
    
    print(f"\n=== Step 5: Saving Multi-Model Results ===\n")
    
    # Create output directory
    output_dir = Path("data/temp_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract results for each model
    data_rows = []
    
    for model_name, results in all_results.items():
        if results is None:
            continue
            
        for i, result in enumerate(results):
            company_row = companies_df.iloc[i]
            
            data_row = {
                'uuid': company_row['uuid'],
                'company_name': company_row['name'],
                'founded_on': company_row['founded_on'],
                'total_funding_usd': company_row['total_funding_usd'],
                'category_list': company_row['category_list'],
                'model': model_name,
                'main_customers': result.answer['main_customers'],
                'main_product': result.answer['main_product']
            }
            data_rows.append(data_row)
    
    # Create DataFrame and save
    results_df = pd.DataFrame(data_rows)
    results_df.to_csv(output_file, index=False)
    
    print(f"✅ Multi-model results saved to: {output_file}")
    print(f"📊 Total responses: {len(results_df)}")
    
    return results_df

def save_multi_model_results_batched(results, companies_df, output_file):
    """Save results from batched ModelList to CSV."""
    
    print(f"\n=== Step 5: Saving Batched Multi-Model Results ===\n")
    
    # Create output directory
    output_dir = Path("data/temp_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract results from batched response
    data_rows = []
    
    for result in results:
        # Get company index from result
        company_idx = result.scenario.get('company_index', 0)
        if company_idx < len(companies_df):
            company_row = companies_df.iloc[company_idx]
            
            data_row = {
                'uuid': company_row['uuid'],
                'company_name': company_row['name'],
                'founded_on': company_row['founded_on'],
                'total_funding_usd': company_row['total_funding_usd'],
                'category_list': company_row['category_list'],
                'model': str(result.model),
                'main_customers': result.answer['main_customers'],
                'main_product': result.answer['main_product']
            }
            data_rows.append(data_row)
    
    # Create DataFrame and save
    results_df = pd.DataFrame(data_rows)
    results_df.to_csv(output_file, index=False)
    
    print(f"✅ Batched multi-model results saved to: {output_file}")
    print(f"📊 Total responses: {len(results_df)}")
    
    return results_df

def main():
    """Main execution function."""
    
    print("🏥 Healthcare Company Analysis with ExpectedParrot (Multi-Model)\n")
    
    # Check if sample already exists
    sample_file = Path("data/temp_output/healthcare_companies_sample.csv")
    
    if sample_file.exists():
        print("📁 Loading existing sample...")
        selected_companies = pd.read_csv(sample_file)
        print(f"✅ Loaded {len(selected_companies)} companies from existing sample")
    else:
        print("📁 Sample not found. Creating new sample...")
        # Step 1: Filter companies
        filtered_companies = filter_healthcare_companies()
        if filtered_companies is None:
            return
        
        # Step 2: Get descriptions
        companies_with_desc = get_company_descriptions(filtered_companies)
        if companies_with_desc is None:
            return
        
        # Step 2.5: Create sample
        selected_companies = save_sample_companies(companies_with_desc, sample_file)
    
    # Step 3: Create EDSL survey
    survey, agent = create_edsl_survey()
    
    # Step 4: Run analysis with multiple models (batched)
    from edsl import ModelList
    
    models = ModelList([
        Model("gpt-4o-mini"),
        Model("gemini-2.0-flash-exp")
    ])
    
    # Use the first 30 companies
    edsl_companies = selected_companies.head(30)
    print(f"\nUsing first 30 companies for model comparison...")
    
    # Create scenarios
    print("Creating scenarios for each company...")
    scenarios = []
    for idx, row in edsl_companies.iterrows():
        scenario = Scenario({
            "company_name": row['name'],
            "company_description": row['description'],
            "company_uuid": row['uuid'],
            "company_index": idx
        })
        scenarios.append(scenario)
    
    print(f"Created {len(scenarios)} scenarios")
    print(f"Running batched analysis with {len(models)} models...")
    print(f"Expected total responses: {len(scenarios)} × {len(models)} = {len(scenarios) * len(models)}")
    
    # Run all models in one batched query
    try:
        results = survey.by(agent).by(scenarios).by(models).run()
        print(f"✅ Batched analysis completed successfully!")
        print(f"Results: {len(results)} total responses\n")
        
        # Save results
        output_file = Path("data/temp_output/healthcare_company_analysis_30_companies.csv")
        results_df = save_multi_model_results_batched(results, edsl_companies, output_file)
        
        # Display comparison results
        print(f"\n=== Model Comparison Results (First 10 companies) ===\n")
        comparison_df = results_df.pivot(index='company_name', columns='model', values=['main_customers', 'main_product'])
        print(comparison_df.head(10))
        
        print(f"\n🎉 Multi-model analysis complete! Check {output_file} for full results.")
        
    except Exception as e:
        print(f"❌ Error in batched analysis: {e}")
        return

if __name__ == "__main__":
    main() 