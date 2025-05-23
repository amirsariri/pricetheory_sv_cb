# ImageNet-2012 Crunchbase Dataset

This repository contains tools for downloading and processing Crunchbase data for companies in the ImageNet-2012 dataset. The project supports both Kaggle and official Crunchbase API data sources.

## Quick Start

### Prerequisites

- Python 3.9 or higher
- Poetry for dependency management
- Kaggle account and API credentials (for Kaggle data source)
- Crunchbase API credentials (for S3 data source)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/imagenet_2012_cb.git
cd imagenet_2012_cb
```

2. Install dependencies using Poetry:
```bash
poetry install
```

### Using Kaggle Data Source

1. Set up Kaggle credentials:
```bash
export KAGGLE_USERNAME=<your-kaggle-username>
export KAGGLE_KEY=<your-kaggle-api-key>
```

2. Download the data:
```bash
poetry run python -m imagenet_cb.download --source kaggle
```

### Using Crunchbase S3 Data Source

1. Create a `.env` file with your Crunchbase credentials:
```bash
echo "CB_S3_KEY=your-key" > .env
echo "CB_S3_SECRET=your-secret" >> .env
```

2. Download the data:
```bash
poetry run python -m imagenet_cb.download --source s3
```

## Project Structure

```
imagenet_2012_cb/
├── data/
│   ├── raw/           # Original Crunchbase data
│   └── processed/     # Processed datasets
├── src/
│   └── imagenet_cb/   # Python package
│       ├── paths.py   # Path management
│       ├── download.py # Data download utilities
│       ├── filter.py  # Data filtering
│       └── milestones.py # Milestone processing
├── pyproject.toml     # Project dependencies
└── README.md         # This file
```

## Development

- Code style is enforced using Black and isort
- Tests can be run using `poetry run pytest`
- GitHub Actions workflow runs tests on push

## License

[Your chosen license] 