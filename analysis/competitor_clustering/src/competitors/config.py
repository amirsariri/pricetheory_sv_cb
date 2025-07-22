from pathlib import Path
from pydantic import BaseModel, Field

class CompetitorSettings(BaseModel):
    """Settings for competitor clustering pipeline."""

    model_name: str = Field(
        "sentence-transformers/all-mpnet-base-v2",
        description="Sentence-transformer model identifier",
    )

    k: int = Field(20, description="# nearest neighbours", gt=0)

    tau: float = Field(0.55, description="Cosine-similarity threshold", ge=0.0, le=1.0)

    alpha: float = Field(0.6, description="Product vs market weighting", ge=0.0, le=1.0)

    seed: int = Field(42, description="Random seed")

    input: Path = Path("data/processed/edsl_survey.csv")  # ~100K companies
    output_dir: Path = Path("data/processed/competitor_clustering")
