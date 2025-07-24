#!/usr/bin/env python3
"""
Analyze competitor clustering results and provide descriptive statistics
for economic research on product-market structure.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

def load_clustering_results():
    """Load clustering results and metadata."""
    # Load clustered data
    df = pd.read_csv("data/processed/competitor_clustering/clustered_companies.csv")
    
    # Load metadata
    with open("data/processed/competitor_clustering/metadata.json", "r") as f:
        metadata = json.load(f)
    
    return df, metadata

def analyze_cluster_distribution(df):
    """Analyze cluster size distribution."""
    cluster_sizes = df['cluster_id'].value_counts().sort_values(ascending=False)
    
    print("=== CLUSTER SIZE DISTRIBUTION ===")
    print(f"Total number of clusters: {len(cluster_sizes)}")
    print(f"Total number of companies: {len(df)}")
    print(f"Average cluster size: {cluster_sizes.mean():.2f}")
    print(f"Median cluster size: {cluster_sizes.median():.2f}")
    print(f"Standard deviation: {cluster_sizes.std():.2f}")
    print(f"Minimum cluster size: {cluster_sizes.min()}")
    print(f"Maximum cluster size: {cluster_sizes.max()}")
    
    # Percentiles
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    print("\nCluster size percentiles:")
    for p in percentiles:
        print(f"  {p}th percentile: {cluster_sizes.quantile(p/100):.0f}")
    
    # Size categories
    small_clusters = (cluster_sizes <= 5).sum()
    medium_clusters = ((cluster_sizes > 5) & (cluster_sizes <= 50)).sum()
    large_clusters = (cluster_sizes > 50).sum()
    
    print(f"\nCluster size categories:")
    print(f"  Small clusters (≤5 companies): {small_clusters} ({small_clusters/len(cluster_sizes)*100:.1f}%)")
    print(f"  Medium clusters (6-50 companies): {medium_clusters} ({medium_clusters/len(cluster_sizes)*100:.1f}%)")
    print(f"  Large clusters (>50 companies): {large_clusters} ({large_clusters/len(cluster_sizes)*100:.1f}%)")
    
    return cluster_sizes

def analyze_market_concentration(df, cluster_sizes):
    """Analyze market concentration using various metrics."""
    print("\n=== MARKET CONCENTRATION ANALYSIS ===")
    
    # Herfindahl-Hirschman Index (HHI) equivalent for cluster sizes
    total_companies = len(df)
    market_shares = cluster_sizes / total_companies
    hhi = (market_shares ** 2).sum()
    
    print(f"Market concentration (HHI-like): {hhi:.4f}")
    print(f"  - HHI < 0.15: Unconcentrated")
    print(f"  - 0.15 ≤ HHI < 0.25: Moderately concentrated") 
    print(f"  - HHI ≥ 0.25: Highly concentrated")
    
    # Top cluster analysis
    top_10_clusters = cluster_sizes.head(10)
    top_10_share = top_10_clusters.sum() / total_companies
    
    print(f"\nTop 10 clusters contain {top_10_share:.1%} of all companies")
    print("Top 10 largest clusters:")
    for i, (cluster_id, size) in enumerate(top_10_clusters.items(), 1):
        share = size / total_companies
        print(f"  {i}. Cluster {cluster_id}: {size} companies ({share:.1%})")
    
    return hhi, top_10_clusters

def analyze_category_distribution(df):
    """Analyze category distribution across clusters."""
    print("\n=== CATEGORY ANALYSIS ===")
    
    # Clean categories
    df['categories_clean'] = df['category_list'].fillna('Unknown')
    
    # Most common categories
    all_categories = []
    for cats in df['categories_clean']:
        if cats != 'Unknown':
            all_categories.extend([cat.strip() for cat in cats.split(',')])
    
    category_counts = Counter(all_categories)
    print(f"Total unique categories: {len(category_counts)}")
    print("\nTop 20 most common categories:")
    for cat, count in category_counts.most_common(20):
        print(f"  {cat}: {count} companies")
    
    # Category diversity within clusters
    cluster_category_diversity = []
    for cluster_id in df['cluster_id'].unique():
        cluster_companies = df[df['cluster_id'] == cluster_id]
        categories_in_cluster = []
        for cats in cluster_companies['categories_clean']:
            if cats != 'Unknown':
                categories_in_cluster.extend([cat.strip() for cat in cats.split(',')])
        unique_categories = len(set(categories_in_cluster))
        cluster_category_diversity.append(unique_categories)
    
    print(f"\nCategory diversity within clusters:")
    print(f"  Average unique categories per cluster: {np.mean(cluster_category_diversity):.1f}")
    print(f"  Median unique categories per cluster: {np.median(cluster_category_diversity):.1f}")
    
    return category_counts

def analyze_cluster_characteristics(df):
    """Analyze characteristics of different cluster types."""
    print("\n=== CLUSTER CHARACTERISTICS ===")
    
    # Analyze singleton clusters (clusters with only 1 company)
    cluster_sizes = df['cluster_id'].value_counts()
    singleton_clusters = cluster_sizes[cluster_sizes == 1].index
    singleton_companies = df[df['cluster_id'].isin(singleton_clusters)]
    
    print(f"Singleton clusters (unique companies): {len(singleton_clusters)}")
    print(f"Companies in singleton clusters: {len(singleton_companies)} ({len(singleton_companies)/len(df)*100:.1f}%)")
    
    # Analyze large clusters (top 10%)
    large_cluster_threshold = cluster_sizes.quantile(0.9)
    large_clusters = cluster_sizes[cluster_sizes >= large_cluster_threshold].index
    large_cluster_companies = df[df['cluster_id'].isin(large_clusters)]
    
    print(f"Large clusters (top 10%): {len(large_clusters)}")
    print(f"Companies in large clusters: {len(large_cluster_companies)} ({len(large_cluster_companies)/len(df)*100:.1f}%)")
    
    return singleton_companies, large_cluster_companies

def generate_summary_statistics(df, metadata):
    """Generate comprehensive summary statistics."""
    print("=" * 60)
    print("COMPETITOR CLUSTERING ANALYSIS SUMMARY")
    print("=" * 60)
    
    print(f"\nDataset Overview:")
    print(f"  Total companies analyzed: {len(df):,}")
    print(f"  Total product-markets identified: {df['cluster_id'].nunique():,}")
    print(f"  Average companies per product-market: {len(df)/df['cluster_id'].nunique():.1f}")
    
    print(f"\nClustering Parameters:")
    print(f"  Model: {metadata['settings']['model_name']}")
    print(f"  Similarity threshold (tau): {metadata['settings']['tau']}")
    print(f"  Text vs category weighting (alpha): {metadata['settings']['alpha']}")
    print(f"  Graph density: {metadata['metrics']['graph_density']:.4f}")
    print(f"  Silhouette score: {metadata['metrics']['silhouette_score']:.3f}")
    
    # Cluster distribution
    cluster_sizes = analyze_cluster_distribution(df)
    
    # Market concentration
    hhi, top_clusters = analyze_market_concentration(df, cluster_sizes)
    
    # Category analysis
    category_counts = analyze_category_distribution(df)
    
    # Cluster characteristics
    singleton_companies, large_cluster_companies = analyze_cluster_characteristics(df)
    
    print("\n" + "=" * 60)
    print("KEY INSIGHTS FOR ECONOMIC RESEARCH")
    print("=" * 60)
    
    print(f"\n1. Market Structure:")
    print(f"   - The dataset reveals {df['cluster_id'].nunique():,} distinct product-markets")
    print(f"   - Market concentration is {'high' if hhi >= 0.25 else 'moderate' if hhi >= 0.15 else 'low'}")
    print(f"   - {len(singleton_companies)/len(df)*100:.1f}% of companies operate in unique product-markets")
    
    print(f"\n2. Competitive Dynamics:")
    print(f"   - {len(large_cluster_companies)/len(df)*100:.1f}% of companies face high competition (large clusters)")
    print(f"   - Average cluster size of {cluster_sizes.mean():.1f} suggests moderate competitive intensity")
    print(f"   - Top 10 clusters contain {top_clusters.sum()/len(df)*100:.1f}% of all companies")
    
    print(f"\n3. Industry Diversity:")
    print(f"   - {len(category_counts)} unique industry categories identified")
    print(f"   - Most common categories: {', '.join([cat for cat, _ in category_counts.most_common(5)])}")
    
    return {
        'total_companies': len(df),
        'total_clusters': df['cluster_id'].nunique(),
        'avg_cluster_size': cluster_sizes.mean(),
        'market_concentration': hhi,
        'singleton_share': len(singleton_companies)/len(df),
        'large_cluster_share': len(large_cluster_companies)/len(df)
    }

if __name__ == "__main__":
    df, metadata = load_clustering_results()
    summary_stats = generate_summary_statistics(df, metadata) 