"""
Competitor clustering package.

This package provides tools for identifying competitor clusters among companies
based on their product and market descriptions.
"""

from .config import CompetitorSettings
from .pipeline import run_competitor_clustering
from .embed import get_embeddings
from .graph import build_similarity_graph
from .cluster import find_clusters
from .validate import calculate_clustering_metrics, generate_validation_samples

__all__ = [
    "CompetitorSettings",
    "run_competitor_clustering", 
    "get_embeddings",
    "build_similarity_graph",
    "find_clusters",
    "calculate_clustering_metrics",
    "generate_validation_samples"
]
