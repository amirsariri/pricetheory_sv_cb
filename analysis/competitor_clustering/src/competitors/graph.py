from typing import List, Tuple
import numpy as np
import pandas as pd
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
                
            # Handle NaN values and convert to string
            cat_i = str(categories[i]) if categories[i] and not pd.isna(categories[i]) else ""
            cat_j = str(categories[j]) if categories[j] and not pd.isna(categories[j]) else ""
            
            # Parse categories (comma-separated)
            cats_i = set(cat.strip() for cat in cat_i.split(',')) if cat_i else set()
            cats_j = set(cat.strip() for cat in cat_j.split(',')) if cat_j else set()
            
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
    Build similarity graph using memory-efficient approach with batch processing.
    
    Args:
        embeddings: Company embeddings array
        categories: List of category strings
        settings: Configuration settings
        
    Returns:
        Sparse adjacency matrix
    """
    logger.info(f"Building similarity graph for {len(embeddings)} companies")
    
    n_companies = len(embeddings)
    
    # Normalize embeddings for cosine similarity
    embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Calculate category similarities (this is memory-intensive but necessary)
    logger.info("Calculating category similarities...")
    category_similarities = _calculate_category_similarity(categories)
    
    # Build adjacency matrix with memory-efficient batch processing
    rows, cols, data = [], [], []
    batch_size = 1000  # Process in batches to manage memory
    
    logger.info(f"Processing similarity matrix in batches of {batch_size}")
    
    for start_idx in range(0, n_companies, batch_size):
        end_idx = min(start_idx + batch_size, n_companies)
        logger.info(f"Processing batch {start_idx//batch_size + 1}/{(n_companies + batch_size - 1)//batch_size}")
        
        # Compute cosine similarities for this batch
        batch_embeddings = embeddings_norm[start_idx:end_idx]
        batch_similarities = np.dot(batch_embeddings, embeddings_norm.T)
        
        # Process each company in the batch
        for i in range(end_idx - start_idx):
            global_i = start_idx + i
            
            # Find similar companies (only upper triangle to avoid duplicates)
            for j in range(global_i + 1, n_companies):
                text_sim = batch_similarities[i, j]
                cat_sim = category_similarities[global_i, j]
                
                # Combined similarity: 80% text, 20% category
                combined_sim = 0.8 * text_sim + 0.2 * cat_sim
                
                # Smart filtering: use text similarity as primary, category as secondary
                if text_sim >= settings.tau:  # Primary filter
                    # Only add edge if categories are reasonably similar OR text similarity is very high
                    if cat_sim >= 0.1 or text_sim >= 0.8:
                        rows.extend([global_i, j])  # Add both directions
                        cols.extend([j, global_i])
                        data.extend([combined_sim, combined_sim])
        
        # Clear batch memory
        del batch_similarities
    
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
