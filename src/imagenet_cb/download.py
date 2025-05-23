"""Download Crunchbase data from Kaggle or Crunchbase S3."""

import os
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

import click
import requests
from dotenv import load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi
from tqdm import tqdm

from .paths import RAW_DATA_DIR

# Load environment variables
load_dotenv()

def download_from_kaggle() -> None:
    """Download Crunchbase data from Kaggle."""
    api = KaggleApi()
    api.authenticate()
    
    # Download the dataset
    click.echo("Downloading Crunchbase data from Kaggle...")
    api.dataset_download_files(
        "chhinna/crunchbase-data",
        path=str(RAW_DATA_DIR),
        unzip=True
    )
    click.echo("Download complete!")

def download_from_s3() -> None:
    """Download Crunchbase data from S3."""
    # Get credentials from environment
    key = os.getenv("CB_S3_KEY")
    secret = os.getenv("CB_S3_SECRET")
    
    if not key or not secret:
        raise click.ClickException(
            "CB_S3_KEY and CB_S3_SECRET must be set in .env file"
        )
    
    # Calculate yesterday's date for the export
    yesterday = datetime.now() - timedelta(days=1)
    date_str = yesterday.strftime("%Y-%m-%d")
    
    # S3 URL template
    base_url = f"https://crunchbase-data-exports.s3.amazonaws.com/{date_str}/"
    
    # List of files to download
    files = [
        "organizations.csv.gz",
        "funding_rounds.csv.gz",
        "acquisitions.csv.gz",
        "investments.csv.gz",
        "people.csv.gz",
    ]
    
    # Download each file
    for filename in tqdm(files, desc="Downloading files"):
        url = base_url + filename
        response = requests.get(url, auth=(key, secret), stream=True)
        response.raise_for_status()
        
        # Save the file
        output_path = RAW_DATA_DIR / filename
        total_size = int(response.headers.get("content-length", 0))
        
        with open(output_path, "wb") as f, tqdm(
            desc=filename,
            total=total_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                pbar.update(size)

@click.command()
@click.option(
    "--source",
    type=click.Choice(["kaggle", "s3"]),
    default="kaggle",
    help="Data source to use (kaggle or s3)",
)
def main(source: Literal["kaggle", "s3"]) -> None:
    """Download Crunchbase data from either Kaggle or Crunchbase S3."""
    if source == "kaggle":
        download_from_kaggle()
    else:
        download_from_s3()

if __name__ == "__main__":
    main() 