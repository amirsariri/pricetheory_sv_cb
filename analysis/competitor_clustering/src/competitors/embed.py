from functools import lru_cache
from typing import List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from loguru import logger

from .config import CompetitorSettings


@lru_cache(maxsize=1)
def _get_model(model_name: str) -> SentenceTransformer:
    """Get cached sentence transformer model."""
    logger.info(f"Loading sentence transformer model: {model_name}")
    return SentenceTransformer(model_name)


def get_embeddings(
    products: List[str], 
    markets: List[str], 
    settings: CompetitorSettings
) -> np.ndarray:
    """
    Generate weighted embeddings from product and market descriptions.
    
    Args:
        products: List of product descriptions
        markets: List of market/customer descriptions  
        settings: Configuration settings
        
    Returns:
        Weighted embeddings array of shape (n_companies, embedding_dim)
    """
    logger.info(f"Generating embeddings for {len(products)} companies")
    
    # Get cached model
    model = _get_model(settings.model_name)
    
    # Generate separate embeddings
    product_embeddings = model.encode(products, show_progress_bar=True)
    market_embeddings = model.encode(markets, show_progress_bar=True)
    
    # Weighted combination
    alpha = settings.alpha
    weighted_embeddings = alpha * product_embeddings + (1 - alpha) * market_embeddings
    
    # Normalize
    norms = np.linalg.norm(weighted_embeddings, axis=1, keepdims=True)
    weighted_embeddings = weighted_embeddings / (norms + 1e-8)
    
    logger.info(f"Generated embeddings shape: {weighted_embeddings.shape}")
    return weighted_embeddings
