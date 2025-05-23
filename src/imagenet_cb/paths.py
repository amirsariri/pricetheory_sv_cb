"""Path management for the ImageNet-Crunchbase project."""

from pathlib import Path

# Project root directory (parent of src/)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

def get_raw_file_path(filename: str) -> Path:
    """Get the full path for a raw data file.
    
    Args:
        filename: Name of the file in the raw data directory
        
    Returns:
        Path object pointing to the file
    """
    return RAW_DATA_DIR / filename

def get_processed_file_path(filename: str) -> Path:
    """Get the full path for a processed data file.
    
    Args:
        filename: Name of the file in the processed data directory
        
    Returns:
        Path object pointing to the file
    """
    return PROCESSED_DATA_DIR / filename 