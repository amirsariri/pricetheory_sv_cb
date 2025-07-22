#!/usr/bin/env python3
"""
Run the competitor clustering pipeline.
"""

import hydra
from omegaconf import DictConfig
from loguru import logger
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from competitors.pipeline import run_competitor_clustering
from competitors.config import CompetitorSettings


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig) -> None:
    """Run the competitor clustering pipeline with Hydra configuration."""
    
    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    # Convert Hydra config to Pydantic settings
    settings = CompetitorSettings(
        model_name=cfg.get("model_name", "sentence-transformers/all-mpnet-base-v2"),
        k=cfg.get("k", 20),
        tau=cfg.get("tau", 0.55),
        alpha=cfg.get("alpha", 0.6),
        seed=cfg.get("seed", 42),
        input=Path(cfg.get("input", "data/processed/edsl_survey.csv")),
        output_dir=Path(cfg.get("output_dir", "data/processed/competitor_clustering"))
    )
    
    logger.info("Starting competitor clustering with settings:")
    logger.info(f"  Model: {settings.model_name}")
    logger.info(f"  k: {settings.k}")
    logger.info(f"  tau: {settings.tau}")
    logger.info(f"  alpha: {settings.alpha}")
    logger.info(f"  Input: {settings.input}")
    logger.info(f"  Output: {settings.output_dir}")
    
    # Run the pipeline
    try:
        results = run_competitor_clustering(settings)
        logger.info("Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
