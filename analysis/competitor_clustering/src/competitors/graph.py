from typing import List, Tuple
import numpy as np
import faiss
from scipy.sparse import csr_matrix
from loguru import logger

from .config import CompetitorSettings
from .text_prep import clean_text


def _calculate_category_similarity(categories: List[str]) -> np.ndarray:
    """Calculate Jaccard similarity matrix for categories."""
    n = len(categories)
    similarity_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i, n):
            if i == j:
                similarity_matrix[i, j] = 1.0
                continue
                
            # Parse categories (comma-separated)
            cats_i = set(cat.strip() for cat in categories[i].split(',')) if categories[i] else set()
            cats_j = set(cat.strip() for cat in categories[j].split(',')) if categories[j] else set()
            
            # Jaccard similarity
            if cats_i and cats_j:
                intersection = len(cats_i & cats_j)
                union = len(cats_i | cats_j)
                similarity_matrix[i, j] = similarity_matrix[j, i] = intersection / union
            else:
                similarity_matrix[i, j] = similarity_matrix[j, i] = 0.0
    
    return similarity_matrix


def build_similarity_graph(
    embeddings: np.ndarray,
    categories: List[str],
    settings: CompetitorSettings
) -> csr_matrix:
    """
    Build similarity graph using direct cosine similarity computation.
    
    Args:
        embeddings: Company embeddings array
        categories: List of category strings
        settings: Configuration settings
        
    Returns:
        Sparse adjacency matrix
    """
    logger.info(f"Building similarity graph for {len(embeddings)} companies")
    
    n_companies = len(embeddings)
    
    # Compute cosine similarities directly (avoiding FAISS for now)
    similarities = np.dot(embeddings, embeddings.T)
    
    # Build adjacency matrix
    rows, cols, data = [], [], []
    
    for i in range(n_companies):
        for j in range(i + 1, n_companies):  # Only upper triangle
            sim = similarities[i, j]
            
            # Add edge if above threshold
            if sim >= settings.tau:
                rows.extend([i, j])  # Add both directions
                cols.extend([j, i])
                data.extend([sim, sim])
    
    # Create sparse matrix
    adjacency_matrix = csr_matrix(
        (data, (rows, cols)), 
        shape=(n_companies, n_companies)
    )
    
    # Get graph statistics
    n_edges = adjacency_matrix.nnz // 2  # Divide by 2 since symmetric
    density = n_edges / (n_companies * (n_companies - 1) / 2)
    
    logger.info(f"Built graph with {n_edges} edges, density: {density:.4f}")
    
    return adjacency_matrix
