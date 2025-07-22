from typing import List, Tuple
import numpy as np
import igraph as ig
import leidenalg as la
from loguru import logger

from .config import CompetitorSettings


def find_clusters(
    adjacency_matrix, 
    settings: CompetitorSettings
) -> np.ndarray:
    """
    Find competitor clusters using Leiden algorithm.
    
    Args:
        adjacency_matrix: Sparse adjacency matrix from graph.py
        settings: Configuration settings
        
    Returns:
        Cluster labels array
    """
    logger.info("Finding clusters using Leiden algorithm")
    
    # Convert to igraph
    edges = adjacency_matrix.nonzero()
    weights = adjacency_matrix.data
    
    # Create igraph object
    g = ig.Graph(
        n=adjacency_matrix.shape[0],
        edges=list(zip(edges[0], edges[1])),
        directed=False
    )
    g.es['weight'] = weights
    
    # Set random seed for reproducibility
    np.random.seed(settings.seed)
    
    # Run Leiden clustering
    partition = la.find_partition(
        g,
        la.ModularityVertexPartition,
        weights='weight'
    )
    
    # Extract cluster labels
    labels = np.array(partition.membership)
    
    # Log clustering results
    n_clusters = len(set(labels))
    cluster_sizes = [sum(labels == i) for i in range(n_clusters)]
    
    logger.info(f"Found {n_clusters} clusters")
    logger.info(f"Cluster sizes - min: {min(cluster_sizes)}, max: {max(cluster_sizes)}, mean: {np.mean(cluster_sizes):.1f}")
    
    return labels
