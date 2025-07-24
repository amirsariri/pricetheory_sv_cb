import json
from pathlib import Path
import pandas as pd
import numpy as np
from loguru import logger

from .config import CompetitorSettings
from .text_prep import clean_text
from .embed import get_embeddings
from .graph import build_similarity_graph
from .cluster import find_clusters
from .validate import calculate_clustering_metrics, generate_validation_samples, log_clustering_summary
from .llm_validate import validate_clusters_with_llm, save_llm_validation_results


def run_competitor_clustering(settings: CompetitorSettings) -> dict:
    """
    Run the complete competitor clustering pipeline.
    
    Args:
        settings: Configuration settings
        
    Returns:
        Dictionary containing results and metadata
    """
    logger.info("Starting competitor clustering pipeline")
    
    # Create output directory
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load and clean data
    logger.info(f"Loading data from {settings.input}")
    df = pd.read_csv(settings.input)
    logger.info(f"Loaded {len(df)} companies")
    
    # Load original company data to get descriptions
    logger.info("Loading original company data for descriptions")
    # Try different possible paths for the original file
    possible_paths = [
        Path("../../data/processed/orgs_2012_2018_survived.csv"),  # From competitor_clustering/
        Path("../../../data/processed/orgs_2012_2018_survived.csv"),  # From competitor_clustering/scripts/
        Path("data/processed/orgs_2012_2018_survived.csv"),  # From project root
    ]
    
    original_file = None
    for path in possible_paths:
        if path.exists():
            original_file = path
            break
    
    if original_file:
        original_df = pd.read_csv(original_file, usecols=['uuid', 'description'])
        df = df.merge(original_df, on='uuid', how='left')
        logger.info(f"Merged descriptions for {df['description'].notna().sum()} companies")
    else:
        logger.warning(f"Original file not found. Tried paths: {[str(p) for p in possible_paths]}")
        df['description'] = None
    
    # Clean text descriptions
    logger.info("Cleaning text descriptions")
    df['main_customers_clean'] = df['main_customers'].apply(clean_text)
    df['main_product_clean'] = df['main_product'].apply(clean_text)
    
    # Remove companies with empty descriptions
    mask = (df['main_customers_clean'] != '') & (df['main_product_clean'] != '')
    df = df[mask].reset_index(drop=True)
    logger.info(f"After cleaning: {len(df)} companies")
    
    # Step 2: Generate embeddings
    logger.info("Generating embeddings")
    embeddings = get_embeddings(
        products=df['main_product_clean'].tolist(),
        markets=df['main_customers_clean'].tolist(),
        settings=settings
    )
    
    # Step 3: Build similarity graph
    logger.info("Building similarity graph")
    adjacency_matrix = build_similarity_graph(
        embeddings=embeddings,
        categories=df['category_list'].tolist(),
        settings=settings
    )
    
    # Step 4: Find clusters
    logger.info("Finding clusters")
    labels = find_clusters(adjacency_matrix, settings)
    
    # Step 5: Calculate metrics
    logger.info("Calculating clustering metrics")
    metrics = calculate_clustering_metrics(embeddings, labels, adjacency_matrix)
    
    # Step 6: Generate validation samples
    logger.info("Generating validation samples")
    validation_samples = generate_validation_samples(df, labels, n_samples=10)
    
    # Initialize metadata
    metadata = {
        'settings': settings.model_dump(),
        'metrics': metrics,
        'validation_samples': validation_samples,
        'output_files': {
            'clustered_data': str(settings.output_dir / "clustered_companies.csv"),
            'adjacency_matrix': str(settings.output_dir / "adjacency_matrix.npz"),
            'embeddings': str(settings.output_dir / "embeddings.npy")
        }
    }
    
    # Step 7: LLM-based cluster validation (disabled for now)
    logger.info("LLM validation disabled - will be run as separate stage")
    metadata['llm_validation'] = {'disabled': 'will be run separately'}
    
    # Step 8: Save results
    logger.info("Saving results")
    
    # Add cluster labels to dataframe
    df['cluster_id'] = labels
    
    # Save clustered data
    output_file = settings.output_dir / "clustered_companies.csv"
    df.to_csv(output_file, index=False)
    logger.info(f"Saved clustered data to {output_file}")
    
    # Save metadata (already initialized above)
    
    metadata_file = settings.output_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info(f"Saved metadata to {metadata_file}")
    
    # Save sparse matrices
    np.save(settings.output_dir / "embeddings.npy", embeddings)
    from scipy.sparse import save_npz
    save_npz(settings.output_dir / "adjacency_matrix.npz", adjacency_matrix)
    
    # Step 8: Log summary
    log_clustering_summary(df, labels, metrics)
    
    logger.info("Competitor clustering pipeline completed successfully")
    
    return metadata
