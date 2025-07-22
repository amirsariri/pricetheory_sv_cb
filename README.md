# Startup Competition Analysis: Crunchbase Dataset

This repository contains a comprehensive analysis of startup competition and entrepreneurship effects using Crunchbase data. The project analyzes how entry into entrepreneurship affects existing startups' success and wages, with a focus on identifying competitors and understanding competitive dynamics in the startup ecosystem.

## Quick Start

### Prerequisites

- Python 3.12 or higher
- Poetry for dependency management
- R and Stata for statistical analysis
- OpenAI API key for Codex CLI and EDSL analysis
- Crunchbase data access (Kaggle or S3)

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

### Data Setup

1. **Crunchbase Data**: Place your Crunchbase data files in `data/250612_cb_data/`
   - `organizations.csv` - Company information
   - `organization_descriptions.csv` - Company descriptions

2. **Environment Setup**: Set up your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key"
```

3. **Install Dependencies**:
```bash
poetry install
```

### Data Processing Pipeline

1. **Filter Companies**: Run the data preparation script to filter US startups (2012-2018):
```bash
cd analysis/data_preparation
python 01_import_filter_orgs.py
```

2. **Generate EDSL Analysis**: Use ExpectedParrot to analyze company descriptions:
```bash
cd analysis/expectedparrot
python 04_full_dataset_analysis.py
```

3. **Run Clustering**: Identify competitors using the clustering algorithm:
```bash
cd analysis/competitor_clustering/scripts
python run_pipeline.py
```

## Project Structure

```
startup-competition-analysis/
├── analysis/
│   ├── competitor_clustering/ # Competitor identification algorithms
│   ├── data_preparation/     # Data filtering and processing
│   ├── cost_estimation/      # SEMrush API cost analysis
│   ├── expectedparrot/       # EDSL-based company analysis
│   ├── 00master.do          # Main Stata analysis script
│   ├── 10globals.do         # Stata global variables
│   └── 30clean.do           # Data cleaning procedures
├── data/
│   ├── 250612_cb_data/      # Raw Crunchbase data
│   ├── processed/           # Processed datasets
│   │   ├── orgs_2012_2018_survived.csv  # Filtered companies (~103K)
│   │   ├── edsl_survey.csv              # EDSL analysis results (~100K)
│   │   └── urlcheck.csv                 # Website validation results
│   └── temp_output/         # Temporary analysis outputs
├── src/
│   └── imagenet_cb/         # Legacy utilities (to be refactored)
├── references/              # Research papers and documentation
├── tests/                   # Test files
├── pyproject.toml          # Python dependencies
└── README.md              # This file
```

## Research Components

### Data Processing
- **Company Filtering**: Identifies US startups founded 2012-2018 that survived ≥12 months
- **Website Validation**: Checks company websites for active status vs. parked domains
- **Description Analysis**: Uses EDSL to extract product/market descriptions from company text

### Competitor Analysis
- **Text Embeddings**: Converts EDSL-generated descriptions to semantic vectors using sentence transformers
- **Similarity Graphs**: Builds k-NN similarity graphs with category overlap using FAISS
- **Leiden Clustering**: Graph-based clustering algorithm for competitor identification
- **Validation Framework**: Comprehensive metrics and manual validation samples

### Cost Estimation
- **SEMrush Analysis**: Estimates API costs for competitive intelligence data collection
- **Company-years Calculation**: Determines total data points needed for analysis

### Statistical Analysis
- **Stata Integration**: Master scripts for econometric analysis
- **R Support**: Additional statistical modeling capabilities

## Research Focus

This project investigates the effects of entrepreneurial entry on existing startups' performance and wages. Key research questions include:

1. **Competitive Dynamics**: How do new entrants affect incumbent startup success?
2. **Market Concentration**: What is the relationship between market entry and competitive intensity?
3. **Wage Effects**: How does entrepreneurial competition impact employee compensation?
4. **Survival Analysis**: What factors predict startup survival in competitive markets?

## Development

- **Python**: Data processing, clustering algorithms, and API integrations
- **Stata**: Econometric analysis and statistical modeling
- **Code Style**: Enforced using Black and isort
- **Testing**: Run tests using `poetry run pytest`
- **Version Control**: GitHub Actions workflow for automated testing

## Key Datasets

- **~103,000 US startups** (2012-2018, survived ≥12 months)
- **~100,000 companies** with EDSL-generated product/market descriptions
- **Website validation data** for active vs. inactive companies
- **Crunchbase categories** for industry classification
