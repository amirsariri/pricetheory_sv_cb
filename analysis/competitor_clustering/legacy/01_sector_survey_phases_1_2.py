#!/usr/bin/env python
"""Step 1 – Sector classification survey via ExpectedParrot EDSL.

This script reads the master Crunchbase-style company file, randomly samples a
subset (default = 100 rows) and uses ExpectedParrot’s EDSL to ask a multiple-
choice question that classifies each company into a single sector.

Usage (from project root):
    poetry run python analysis/competitor_clustering/01_sector_survey.py \
        --input data/processed/cb_data_main.csv \
        --output data/temp_output/sector_assignment_sample.csv \
        --sample-size 100 \
        --seed 42

The output is a CSV that contains the original fields (uuid, name,
category_list, main_product, main_customers) plus the model-selected `sector`
column returned by EDSL.
"""

from __future__ import annotations

import argparse
import pathlib
from typing import List

import pandas as pd
from edsl import Question, Survey, Agent, Scenario, Model

# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

SECTOR_OPTIONS: List[str] = [
    "Software / SaaS",
    "E-commerce",
    "Healthcare / MedTech",
    "Financial Services / FinTech",
    "Education / EdTech",
    "Manufacturing & Industrial",
    "Media & Entertainment",
    "Transportation & Logistics",
    "Real Estate / PropTech",
    "Energy & Utilities",
    "Telecommunications",
    "Retail & Consumer Goods",
    "Food & Beverage",
    "Fashion / Apparel",
    "Automotive & Mobility",
    "Travel & Tourism",
    "Legal & Compliance",
    "Consulting & Professional Svcs",
    "Advertising / Marketing / AdTech",
    "Security & Cybersecurity",
    "Biotechnology",
    "CleanTech / Green Energy",
    "Hardware / Devices / IoT",
    "Gaming & Esports",
    "Social Media / Community",
    "Other",
]

SELECT_COLUMNS = [
    "uuid",
    "name",
    "category_list",
    "main_product",
    "main_customers",
]


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def build_scenarios(df: pd.DataFrame):
    """Build a list of `Scenario` objects for each company row.

    This follows the same pattern used in the previously working
    `03_healthcare_company_analysis.py` script, avoiding the unsupported
    `ScenarioList.from_source` call.
    """
    scenarios = []
    for _, row in df.iterrows():
        scenarios.append(
            Scenario(
                {
                    "uuid": row["uuid"],
                    "name": row["name"],
                    "category_list": row["category_list"],
                    "main_product": row["main_product"],
                    "main_customers": row["main_customers"],
                }
            )
        )
    return scenarios


def run_sector_question(scenarios):
    """Run the sector classification question and return answers as DataFrame."""

    # Create an agent with specific market analyst traits
    agent = Agent(
        name="Henry Finder",
        traits={
            "persona": "Senior market analyst who understands industry classifications",
            "expertise": "Industry classification, competitor mapping, market research",
            "approach": "Analytical, weighing product & customer overlap"
        }
    )

    # Define the main multiple-choice question
    question1 = Question(
        question_name="sector",
        question_text=(
            "Given the company's main product: {{ scenario.main_product }} and "
            "its primary customers: {{ scenario.main_customers }}, select the "
            "sector that best describes this company."
        ),
        question_type="multiple_choice",
        question_options=SECTOR_OPTIONS,
    )

    # Define the conditional question for "Other" sectors
    question2 = Question(
        question_name="sector_other",
        question_text=(
            "You chose 'Other' for the sector classification. "
            "Please provide a more specific sector name and your confidence level. "
            "Return a JSON with a sector name (4 or fewer words) and your confidence between 0-1. "
            "If unsure, set sector='Unsure'. "
            "Format: {\"sector\": \"sector_name\", \"confidence\": 0.8}"
        ),
        question_type="free_text"
    )

    # Create survey with both questions
    survey = Survey(questions=[question1, question2])
    
    # Add skip logic: only ask question2 if question1 answer is "Other"
    survey = survey.add_skip_rule(
        question2, 
        "sector != 'Other'"
    )

    # Show the survey flow before running
    print("\n=== Survey Flow ===")
    survey.show_flow()
    print("==================\n")

    # Choose a model (same one used in the working healthcare script)
    model = Model("gemini-1.5-flash-8b")

    # Execute the survey with job description
    job = survey.by(agent).by(scenarios).by(model)
    results = job.run(
        remote_inference_description="Competitor Clustering Tests"
    )

    # Extract results manually (mirroring the working script)
    rows = []
    for idx, result in enumerate(results):
        scenario_data = scenarios[idx]  # matches order
        
        # Get the main sector answer
        sector_answer = result.answer["sector"]
        
        # Handle the conditional "Other" sector question
        sector_other_answer = sector_answer  # Default to original answer
        sector_other_confidence = None  # Default confidence for non-"Other" cases
        
        # If "Other" was selected, try to get the refined answer
        if sector_answer == "Other" and "sector_other" in result.answer:
            try:
                # Parse the JSON response from the conditional question
                import json
                other_response = result.answer["sector_other"]
                # Extract the sector name and confidence from the JSON response
                if "sector" in other_response:
                    # Try to parse as JSON first
                    try:
                        parsed = json.loads(other_response)
                        sector_other_answer = parsed.get("sector", "Other")
                        sector_other_confidence = parsed.get("confidence", None)
                    except json.JSONDecodeError:
                        # If not valid JSON, try to extract sector name and confidence from text
                        if '"sector"' in other_response:
                            # Simple extraction - look for "sector": "value"
                            import re
                            sector_match = re.search(r'"sector"\s*:\s*"([^"]+)"', other_response)
                            confidence_match = re.search(r'"confidence"\s*:\s*([0-9]*\.?[0-9]+)', other_response)
                            
                            if sector_match:
                                sector_other_answer = sector_match.group(1)
                            else:
                                sector_other_answer = "Other"
                                
                            if confidence_match:
                                sector_other_confidence = float(confidence_match.group(1))
                            else:
                                sector_other_confidence = None
                        else:
                            sector_other_answer = "Other"
                            sector_other_confidence = None
            except Exception as e:
                print(f"Warning: Could not parse 'Other' response for company {scenario_data['name']}: {e}")
                sector_other_answer = "Other"
                sector_other_confidence = None
        
        rows.append(
            {
                "uuid": scenario_data["uuid"],
                "name": scenario_data["name"],
                "category_list": scenario_data["category_list"],
                "main_product": scenario_data["main_product"],
                "main_customers": scenario_data["main_customers"],
                "sector": sector_answer,
                "sector_other": sector_other_answer,
                "sector_other_confidence": sector_other_confidence,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EDSL sector survey.")
    parser.add_argument(
        "--input",
        type=pathlib.Path,
        default=pathlib.Path("data/processed/cb_data_main.csv"),
        help="Path to the full Crunchbase-style CSV file.",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("data/temp_output/sector_assignment_sample.csv"),
        help="Destination CSV path for the survey results.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
        help="Number of companies to sample for the survey.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling reproducibility.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ---------------------------------------------------------------------
    # 1. Load data and sample
    # ---------------------------------------------------------------------
    full_df = pd.read_csv(args.input, low_memory=False)

    # Ensure required columns exist
    missing_cols = set(SELECT_COLUMNS) - set(full_df.columns)
    if missing_cols:
        raise ValueError(
            f"Input CSV is missing required columns: {', '.join(sorted(missing_cols))}"
        )

    # Reproducible random sampling
    sample_df = (
        full_df.sample(n=args.sample_size, random_state=args.seed)
        .reset_index(drop=True)
    )

    # Optional: fill NaNs to avoid blank scenario values (EDSL handles None but
    # we make it explicit)
    sample_df = sample_df.fillna("")

    # ---------------------------------------------------------------------
    # 2. Build scenarios & run survey
    # ---------------------------------------------------------------------
    scenarios = build_scenarios(sample_df)
    results_df = run_sector_question(scenarios)

    # ---------------------------------------------------------------------
    # 3. Save combined output
    # ---------------------------------------------------------------------
    combined_df = sample_df.merge(results_df, on=SELECT_COLUMNS, how="left")
    
    # Select only the desired columns for output
    output_columns = SELECT_COLUMNS + ["sector", "sector_other", "sector_other_confidence"]
    final_df = combined_df[output_columns]
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(args.output, index=False)
    print(f"✅ Survey complete – results written to {args.output}")


if __name__ == "__main__":
    main() 