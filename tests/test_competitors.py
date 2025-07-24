"""
Unit tests for competitor clustering modules.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent / "analysis/competitor_clustering/src"))

from competitors.config import CompetitorSettings
from competitors.text_prep import clean_text
from competitors.embed import get_embeddings
from competitors.graph import build_similarity_graph
from competitors.cluster import find_clusters


@pytest.fixture
def sample_settings():
    """Sample configuration settings for testing."""
    return CompetitorSettings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",  # Smaller model for testing
        k=5,
        tau=0.5,
        alpha=0.6,
        seed=42
    )


@pytest.fixture
def sample_data():
    """Sample company data for testing."""
    return pd.DataFrame({
        'company_name': ['Company A', 'Company B', 'Company C'],
        'main_customers': [
            'Tech-savvy homeowners seeking automation',
            'Businesses and independent freelancers', 
            'Tech startups and enterprises'
        ],
        'main_product': [
            'Smart home automation platform',
            'Online freelance marketplace platform',
            'Custom software development solutions'
        ],
        'category_list': [
            'Internet of Things,Smart Home',
            'Freelance,Marketplace,Recruiting',
            'Web Development,Software'
        ]
    })


def test_clean_text():
    """Test text cleaning functionality."""
    # Test basic cleaning
    assert clean_text("Tech-Savvy Homeowners Inc.") == "tech-savvy homeowners"
    assert clean_text("AI-Driven Platform LLC") == "ai-driven platform"
    
    # Test empty/null inputs
    assert clean_text("") == ""
    assert clean_text(None) == ""
    
    # Test with special characters
    assert clean_text("E-commerce & SaaS Solutions") == "e-commerce & saas solutions"


def test_embeddings_generation(sample_settings, sample_data):
    """Test embedding generation."""
    # Clean text
    products = [clean_text(text) for text in sample_data['main_product']]
    markets = [clean_text(text) for text in sample_data['main_customers']]
    
    # Generate embeddings
    embeddings = get_embeddings(products, markets, sample_settings)
    
    # Check output shape
    assert embeddings.shape == (3, 384)  # MiniLM-L6-v2 has 384 dimensions
    
    # Check normalization
    norms = np.linalg.norm(embeddings, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-6)


def test_similarity_graph_construction(sample_settings, sample_data):
    """Test similarity graph construction."""
    # Generate embeddings first
    products = [clean_text(text) for text in sample_data['main_product']]
    markets = [clean_text(text) for text in sample_data['main_customers']]
    embeddings = get_embeddings(products, markets, sample_settings)
    
    # Build graph
    adjacency_matrix = build_similarity_graph(
        embeddings=embeddings,
        categories=sample_data['category_list'].tolist(),
        settings=sample_settings
    )
    
    # Check output properties
    assert adjacency_matrix.shape == (3, 3)
    assert adjacency_matrix.nnz > 0  # Should have some edges
    assert adjacency_matrix.dtype == np.float64


def test_clustering_deterministic(sample_settings, sample_data):
    """Test that clustering is deterministic."""
    # Generate embeddings
    products = [clean_text(text) for text in sample_data['main_product']]
    markets = [clean_text(text) for text in sample_data['main_customers']]
    embeddings = get_embeddings(products, markets, sample_settings)
    
    # Build graph
    adjacency_matrix = build_similarity_graph(
        embeddings=embeddings,
        categories=sample_data['category_list'].tolist(),
        settings=sample_settings
    )
    
    # Run clustering twice
    labels1 = find_clusters(adjacency_matrix, sample_settings)
    labels2 = find_clusters(adjacency_matrix, sample_settings)
    
    # Should be identical
    np.testing.assert_array_equal(labels1, labels2)


def test_config_validation():
    """Test configuration validation."""
    # Valid settings
    settings = CompetitorSettings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        k=10,
        tau=0.6,
        alpha=0.5,
        seed=42
    )
    assert settings.k == 10
    assert settings.tau == 0.6
    
    # Test validation
    with pytest.raises(ValueError):
        CompetitorSettings(k=0)  # k must be > 0
    
    with pytest.raises(ValueError):
        CompetitorSettings(tau=1.5)  # tau must be <= 1.0
    
    with pytest.raises(ValueError):
        CompetitorSettings(alpha=-0.1)  # alpha must be >= 0.0 