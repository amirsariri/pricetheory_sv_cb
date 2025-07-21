"""
Full Dataset Analysis with ExpectedParrot
Analyzing all 2012-2018 organizations using EDSL with batch processing
"""

import edsl
from edsl import Question, Survey, Agent, Scenario, Model
import pandas as pd
import numpy as np
from pathlib import Path
import random
import time
import os
from datetime import datetime

# Parameters
# Use absolute paths that work from any directory
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

INPUT_FILE = os.path.join(PROJECT_ROOT, 'data/processed/orgs_2012_2018_survived.csv')
OUTPUT_FILE = os.path.join(PROJECT_ROOT, 'data/processed/edsl_survey.csv')
CHUNK_SIZE = 50  # Process companies in batches
MAX_WORKERS = 1  # EDSL doesn't support threading well, so use 1
PROGRESS_INTERVAL = 10
MODEL_NAME = "gpt-4o-mini"

# Test mode parameters
TEST_MODE = False
TEST_N = 10

def load_organizations():
    """Load the full dataset of 2012-2018 organizations."""
    
    print("=== Step 1: Loading Organizations Data ===\n")
    
    if not Path(INPUT_FILE).exists():
        print(f"❌ Input file not found: {INPUT_FILE}")
        return None
    
    print(f"Loading organizations from: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df):,} organizations\n")
    
    # Add description length for reference (no filtering)
    df['desc_length'] = df['description'].str.len() if 'description' in df.columns else 0
    print(f"Companies with descriptions: {df['description'].notna().sum():,}")
    print(f"Companies without descriptions: {df['description'].isna().sum():,}")
    
    return df

def create_edsl_survey():
    """Create the EDSL survey for company analysis."""
    
    print("\n=== Step 2: Creating EDSL Survey ===\n")
    
    # Create agent
    print("Creating market analyst agent...")
    agent = Agent(traits = {
        "persona": "You are an experienced market analyst skilled in analyzing competitive landscape and business models"
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

def process_chunk(companies_chunk, survey, agent, model, chunk_id):
    """Process a chunk of companies using EDSL."""
    
    print(f"\n--- Processing Chunk {chunk_id} ({len(companies_chunk)} companies) ---")
    
    # Create scenarios
    scenarios = []
    for idx, row in companies_chunk.iterrows():
        scenario = Scenario({
            "company_name": row['name'],
            "company_description": row['description'],
            "company_uuid": row['uuid'],
            "company_index": idx
        })
        scenarios.append(scenario)
    
    print(f"Created {len(scenarios)} scenarios")
    
    # Run survey
    try:
        print(f"Running EDSL survey with {MODEL_NAME}...")
        
        # Create job with description
        job = survey.by(agent).by(scenarios).by(model)
        job_description = f"Product-Market Labelling - Chunk {chunk_id}"
        
        # Set job description and run
        results = job.run(
            remote_inference_description=job_description
        )
        
        print(f"✅ Chunk {chunk_id} completed successfully!")
        print(f"Results: {len(results)} responses\n")
        
        return results
        
    except Exception as e:
        print(f"❌ Error in chunk {chunk_id}: {e}")
        return None

def save_chunk_results(results, companies_chunk, output_file, is_first_chunk=False):
    """Save chunk results to CSV."""
    
    # Create output directory
    output_dir = Path(output_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract results
    data_rows = []
    for i, result in enumerate(results):
        company_row = companies_chunk.iloc[i]
        

        
        data_row = {
            'uuid': company_row['uuid'],
            'company_name': company_row['name'],
            'category_list': company_row['category_list'],
            'description_length': company_row['desc_length'],
            'main_customers': result.answer['main_customers'],
            'main_product': result.answer['main_product'],
            'model_used': MODEL_NAME,
            'processed_at': datetime.now().isoformat()
        }
        data_rows.append(data_row)
    
    # Create DataFrame and save
    results_df = pd.DataFrame(data_rows)
    
    # Write to CSV (append mode)
    mode = 'w' if is_first_chunk else 'a'
    header = is_first_chunk
    results_df.to_csv(output_file, mode=mode, header=header, index=False)
    
    print(f"✅ Chunk results saved to: {output_file}")
    print(f"📊 Companies processed in this chunk: {len(results_df)}")
    
    return results_df

def resume_processing(df, output_file):
    """Check if we can resume from existing output."""
    
    if not Path(output_file).exists():
        print("No existing output found. Starting fresh.")
        return df, 0
    
    print("Checking existing output for resume...")
    existing_df = pd.read_csv(output_file)
    processed_uuids = set(existing_df['uuid'])
    
    print(f"Already processed: {len(processed_uuids):,} companies")
    
    # Filter out already processed companies
    remaining_df = df[~df['uuid'].isin(processed_uuids)]
    print(f"Remaining to process: {len(remaining_df):,} companies")
    
    return remaining_df, len(processed_uuids)

def main():
    """Main execution function."""
    
    # Set output file (modify for test mode)
    output_file = OUTPUT_FILE
    if TEST_MODE:
        output_file = OUTPUT_FILE.replace('.csv', '_test.csv')
    
    print("🏢 Full Dataset Analysis with ExpectedParrot\n")
    print(f"Model: {MODEL_NAME}")
    print(f"Chunk size: {CHUNK_SIZE}")
    print(f"Output file: {output_file}\n")
    
    # Step 1: Load organizations
    df = load_organizations()
    if df is None:
        return
    
    # Step 2: Check for resume
    df_to_process, already_processed = resume_processing(df, output_file)
    
    if len(df_to_process) == 0:
        print("✅ All companies already processed!")
        return
    
    # Test mode: limit to first N companies
    if TEST_MODE:
        df_to_process = df_to_process.head(TEST_N)
        print(f"🧪 TEST MODE: Only processing {len(df_to_process)} companies (limited by TEST_N={TEST_N})")
        print(f"Test results will be saved to: {output_file}")
    
    # Step 3: Create EDSL survey
    survey, agent = create_edsl_survey()
    
    # Step 4: Create model
    model = Model(MODEL_NAME)
    
    # Step 5: Process in chunks
    print(f"\n=== Step 3: Processing {len(df_to_process):,} Companies in Chunks ===\n")
    
    # Split into chunks
    chunks = [df_to_process[i:i+CHUNK_SIZE] for i in range(0, len(df_to_process), CHUNK_SIZE)]
    print(f"Total chunks to process: {len(chunks)}")
    
    start_time = time.time()
    total_processed = already_processed
    
    for chunk_id, chunk in enumerate(chunks, 1):
        print(f"\n{'='*60}")
        print(f"Processing chunk {chunk_id}/{len(chunks)}")
        print(f"{'='*60}")
        
        # Process chunk
        results = process_chunk(chunk, survey, agent, model, chunk_id)
        
        if results is not None:
            # Save results
            is_first_chunk = (chunk_id == 1 and already_processed == 0)
            save_chunk_results(results, chunk, output_file, is_first_chunk)
            
            total_processed += len(chunk)
            
            # Progress update
            elapsed = time.time() - start_time
            avg_time_per_chunk = elapsed / chunk_id
            remaining_chunks = len(chunks) - chunk_id
            estimated_remaining = remaining_chunks * avg_time_per_chunk
            
            print(f"\n📊 Progress Update:")
            print(f"   Chunks completed: {chunk_id}/{len(chunks)}")
            print(f"   Total companies processed: {total_processed:,}")
            print(f"   Average time per chunk: {avg_time_per_chunk:.1f}s")
            print(f"   Estimated time remaining: {estimated_remaining/60:.1f} minutes")
        
        # Polite delay between chunks
        if chunk_id < len(chunks):
            delay = random.uniform(2, 5)
            print(f"Waiting {delay:.1f}s before next chunk...")
            time.sleep(delay)
    
    # Final summary
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"🎉 ANALYSIS COMPLETE!")
    print(f"{'='*60}")
    print(f"Total companies processed: {total_processed:,}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Average time per company: {total_time/total_processed:.1f}s")
    print(f"Results saved to: {output_file}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 