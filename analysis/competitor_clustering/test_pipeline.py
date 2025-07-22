#!/usr/bin/env python3
"""
Test script for the competitor clustering pipeline.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent / "src"))

from competitors.config import CompetitorSettings
from competitors.pipeline import run_competitor_clustering


def test_small_sample():
    """Test the pipeline on a small sample of data."""
    
    # Create test settings with smaller model
    settings = CompetitorSettings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",  # Smaller model for testing
        k=5,
        tau=0.5,
        alpha=0.6,
        seed=42,
        input=Path("../../data/processed/edsl_survey.csv"),
        output_dir=Path("../../data/processed/competitor_clustering_test")
    )
    
    # Load a small sample of data
    print("Loading sample data...")
    df = pd.read_csv(settings.input)
    sample_df = df.head(100)  # Just 100 companies for testing
    
    # Save sample to temporary file
    temp_input = Path("../../data/processed/edsl_survey_sample.csv")
    temp_input.parent.mkdir(parents=True, exist_ok=True)
    sample_df.to_csv(temp_input, index=False)
    
    # Update settings to use sample
    settings.input = temp_input
    
    print(f"Testing with {len(sample_df)} companies...")
    
    try:
        # Run pipeline
        results = run_competitor_clustering(settings)
        
        print("✅ Pipeline completed successfully!")
        print(f"Found {results['metrics']['n_clusters']} clusters")
        print(f"Silhouette score: {results['metrics']['silhouette_score']:.3f}")
        
        # Show some validation samples
        print("\nSample clusters:")
        for i, sample in enumerate(results['validation_samples'][:3]):
            print(f"\nCluster {sample['cluster_id']} ({sample['cluster_size']} companies):")
            for company in sample['companies'][:3]:
                print(f"  - {company['name']}: {company['product']}")
        
        # Clean up
        temp_input.unlink()
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_small_sample()
    sys.exit(0 if success else 1) 