from typing import List, Dict, Any
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics import silhouette_score
from loguru import logger

from .config import CompetitorSettings


def calculate_clustering_metrics(
    embeddings: np.ndarray,
    labels: np.ndarray,
    adjacency_matrix: csr_matrix
) -> Dict[str, float]:
    """
    Calculate clustering quality metrics.
    
    Args:
        embeddings: Company embeddings
        labels: Cluster labels
        adjacency_matrix: Similarity graph
        
    Returns:
        Dictionary of metrics
    """
    logger.info("Calculating clustering metrics")
    
    metrics = {}
    
    # Silhouette score
    if len(set(labels)) > 1:
        metrics['silhouette_score'] = silhouette_score(embeddings, labels)
    else:
        metrics['silhouette_score'] = 0.0
    
    # Cluster statistics
    unique_labels, counts = np.unique(labels, return_counts=True)
    metrics['n_clusters'] = len(unique_labels)
    metrics['avg_cluster_size'] = np.mean(counts)
    metrics['min_cluster_size'] = np.min(counts)
    metrics['max_cluster_size'] = np.max(counts)
    
    # Graph statistics
    n_edges = adjacency_matrix.nnz // 2
    n_nodes = adjacency_matrix.shape[0]
    metrics['graph_density'] = n_edges / (n_nodes * (n_nodes - 1) / 2)
    
    # Intra-cluster density
    intra_cluster_edges = 0
    for label in unique_labels:
        cluster_nodes = np.where(labels == label)[0]
        if len(cluster_nodes) > 1:
            for i in range(len(cluster_nodes)):
                for j in range(i + 1, len(cluster_nodes)):
                    if adjacency_matrix[cluster_nodes[i], cluster_nodes[j]] > 0:
                        intra_cluster_edges += 1
    
    total_possible_intra_edges = sum(count * (count - 1) // 2 for count in counts)
    metrics['intra_cluster_density'] = intra_cluster_edges / total_possible_intra_edges if total_possible_intra_edges > 0 else 0.0
    
    logger.info(f"Metrics: {metrics}")
    return metrics


def generate_validation_samples(
    df: pd.DataFrame,
    labels: np.ndarray,
    n_samples: int = 10
) -> List[Dict[str, Any]]:
    """
    Generate manual validation samples for cluster review.
    
    Args:
        df: Company dataframe
        labels: Cluster labels
        n_samples: Number of clusters to sample
        
    Returns:
        List of cluster samples for manual review
    """
    logger.info(f"Generating {n_samples} validation samples")
    
    unique_labels = np.unique(labels)
    if len(unique_labels) <= n_samples:
        selected_clusters = unique_labels
    else:
        selected_clusters = np.random.choice(unique_labels, n_samples, replace=False)
    
    samples = []
    for cluster_id in selected_clusters:
        cluster_mask = labels == cluster_id
        cluster_companies = df[cluster_mask]
        
        sample = {
            'cluster_id': int(cluster_id),
            'cluster_size': len(cluster_companies),
            'companies': []
        }
        
        # Sample up to 10 companies from this cluster
        n_show = min(10, len(cluster_companies))
        sample_companies = cluster_companies.sample(n=n_show, random_state=42)
        
        for _, company in sample_companies.iterrows():
            sample['companies'].append({
                'name': company['company_name'],
                'customers': company['main_customers'],
                'product': company['main_product'],
                'categories': company['category_list'],
                'description': company.get('description', 'N/A')
            })
        
        samples.append(sample)
    
    return samples


def log_clustering_summary(
    df: pd.DataFrame,
    labels: np.ndarray,
    metrics: Dict[str, float]
) -> None:
    """Log a comprehensive clustering summary."""
    
    logger.info("=== CLUSTERING SUMMARY ===")
    logger.info(f"Total companies: {len(df)}")
    logger.info(f"Number of clusters: {metrics['n_clusters']}")
    logger.info(f"Average cluster size: {metrics['avg_cluster_size']:.1f}")
    logger.info(f"Silhouette score: {metrics['silhouette_score']:.3f}")
    logger.info(f"Graph density: {metrics['graph_density']:.4f}")
    logger.info(f"Intra-cluster density: {metrics['intra_cluster_density']:.3f}")
    
    # Show largest clusters
    unique_labels, counts = np.unique(labels, return_counts=True)
    largest_clusters = sorted(zip(unique_labels, counts), key=lambda x: x[1], reverse=True)[:5]
    
    logger.info("Top 5 largest clusters:")
    for cluster_id, size in largest_clusters:
        logger.info(f"  Cluster {cluster_id}: {size} companies")
