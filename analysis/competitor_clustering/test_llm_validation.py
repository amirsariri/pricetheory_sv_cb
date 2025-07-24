#!/usr/bin/env python3
"""
Test script for LLM-based cluster validation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent / "src"))

from competitors.llm_validate import validate_clusters_with_llm, save_llm_validation_results


def test_llm_validation():
    """Test LLM validation on a small sample."""
    
    # Load the clustered data
    clustered_file = Path("../../data/processed/competitor_clustering_test/clustered_companies.csv")
    
    if not clustered_file.exists():
        print("❌ Clustered data not found. Run test_pipeline.py first.")
        return False
    
    print("Loading clustered data...")
    df = pd.read_csv(clustered_file)
    labels = df['cluster_id'].values
    
    print(f"Loaded {len(df)} companies with {len(set(labels))} clusters")
    
    # Test LLM validation on a few clusters
    print("\nRunning LLM validation on 3 clusters...")
    try:
        results = validate_clusters_with_llm(
            df=df,
            labels=labels,
            n_clusters_to_validate=3,
            model_name="gpt-4o-mini"
        )
        
        print("✅ LLM validation completed successfully!")
        
        # Show results
        print(f"\n=== LLM VALIDATION RESULTS ===")
        print(f"Average cluster quality: {results['overall_metrics']['avg_cluster_quality']:.2f}")
        print(f"High quality clusters: {results['overall_metrics']['high_quality_clusters']}")
        print(f"Low quality clusters: {results['overall_metrics']['low_quality_clusters']}")
        
        # Show cluster summaries
        print(f"\n=== CLUSTER SUMMARIES ===")
        for cluster_id, summary in results['cluster_summaries'].items():
            quality_score = results['cluster_quality_scores'][cluster_id]['score']
            print(f"\nCluster {cluster_id} (Quality: {quality_score:.1f}/10):")
            print(f"Summary: {summary}")
            
            # Show company fit scores
            company_scores = results['company_fit_scores'][cluster_id]
            print("Company fit scores:")
            for company, data in company_scores.items():
                print(f"  - {company}: {data['score']:.1f}/10")
        
        # Save results
        output_dir = Path("../../data/processed/competitor_clustering_test")
        save_llm_validation_results(results, output_dir)
        print(f"\nResults saved to {output_dir}")
        
        return True
        
    except Exception as e:
        print(f"❌ LLM validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_llm_validation()
    sys.exit(0 if success else 1) 