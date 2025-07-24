#!/usr/bin/env python3
"""
Test the full pipeline on a small sample from the actual dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent / "src"))

from competitors.pipeline import run_competitor_clustering
from competitors.config import CompetitorSettings


def test_full_pipeline_small():
    """Test the full pipeline on 100 companies from the actual dataset."""
    
    print("Testing full pipeline on small sample from actual dataset...")
    
    # Create settings for small test
    settings = CompetitorSettings(
        input=Path("../../data/processed/edsl_survey_small_test.csv"),
        output_dir=Path("../../data/processed/competitor_clustering_small_test"),
        model_name="sentence-transformers/all-MiniLM-L6-v2",  # Faster model for testing
        k=10,
        tau=0.6,
        alpha=0.7
    )
    
    # Create output directory
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Input: {settings.input}")
    print(f"Output: {settings.output_dir}")
    print(f"Model: {settings.model_name}")
    
    try:
        # Run the full pipeline
        results = run_competitor_clustering(settings)
        
        print("✅ Full pipeline test completed successfully!")
        
        # Check output files
        output_files = [
            "clustered_companies.csv",
            "metadata.json", 
            "embeddings.npy",
            "adjacency_matrix.npz"
        ]
        
        print("\nOutput files created:")
        for file in output_files:
            file_path = settings.output_dir / file
            if file_path.exists():
                size = file_path.stat().st_size
                print(f"  ✅ {file} ({size:,} bytes)")
            else:
                print(f"  ❌ {file} (missing)")
        
        # Show basic results
        if 'metrics' in results:
            metrics = results['metrics']
            print(f"\nResults:")
            print(f"  Companies: {metrics.get('n_companies', 'N/A')}")
            print(f"  Clusters: {metrics.get('n_clusters', 'N/A')}")
            print(f"  Avg cluster size: {metrics.get('avg_cluster_size', 'N/A'):.1f}")
            print(f"  Silhouette score: {metrics.get('silhouette_score', 'N/A'):.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Full pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_full_pipeline_small()
    sys.exit(0 if success else 1) 