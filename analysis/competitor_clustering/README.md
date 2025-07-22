# Competitor Clustering Pipeline

This directory contains a complete implementation of a competitor clustering algorithm that identifies competitive groups among ~100,000 companies based on their product and market descriptions.

## Overview

The pipeline uses EDSL-generated descriptions (5 words or less) to create semantic embeddings, builds similarity graphs, and applies Leiden clustering to identify competitor clusters. The approach is designed to be simple, effective, and scalable.

## Architecture

### Core Components

1. **Text Preprocessing** (`text_prep.py`): Cleans company descriptions by removing legal suffixes and normalizing text
2. **Embedding Generation** (`embed.py`): Creates weighted embeddings from product and market descriptions using sentence transformers
3. **Graph Construction** (`graph.py`): Builds k-NN similarity graphs with category overlap using FAISS
4. **Clustering** (`cluster.py`): Applies Leiden algorithm to find competitor clusters
5. **Validation** (`validate.py`): Provides clustering diagnostics and manual validation samples
6. **Pipeline** (`pipeline.py`): Orchestrates the complete workflow

### Key Features

- **Weighted embeddings**: Combines product and market descriptions with configurable weighting (`alpha`)
- **Multi-modal similarity**: Combines text similarity with category overlap
- **Adaptive thresholds**: Uses percentile-based similarity thresholds
- **Deterministic results**: Reproducible clustering with seed control
- **Comprehensive validation**: Metrics, diagnostics, and manual review samples

## Configuration

The pipeline uses Pydantic for configuration validation and Hydra for configuration management:

```python
class CompetitorSettings(BaseModel):
    model_name: str = "sentence-transformers/all-mpnet-base-v2"
    k: int = 20  # Number of nearest neighbors
    tau: float = 0.55  # Similarity threshold
    alpha: float = 0.6  # Product vs market weighting
    seed: int = 42  # Random seed
    input: Path = Path("data/processed/edsl_survey.csv")
    output_dir: Path = Path("data/processed/competitor_clustering")
```

## Usage

### Quick Start

1. **Install dependencies**:
```bash
poetry install
```

2. **Run the pipeline**:
```bash
cd analysis/competitor_clustering/scripts
python run_pipeline.py
```

3. **Run with custom settings**:
```bash
python run_pipeline.py k=30 tau=0.6 alpha=0.7
```

### Programmatic Usage

```python
from competitors import CompetitorSettings, run_competitor_clustering

settings = CompetitorSettings(
    k=25,
    tau=0.6,
    alpha=0.7
)

results = run_competitor_clustering(settings)
```

## Output

The pipeline generates several output files:

- `clustered_companies.csv`: Companies with cluster assignments
- `embeddings.npy`: Company embeddings array
- `adjacency_matrix.npz`: Sparse similarity graph
- `metadata.json`: Complete pipeline metadata and metrics

### Validation Samples

The pipeline automatically generates validation samples for manual review:

```json
{
  "cluster_id": 0,
  "cluster_size": 45,
  "companies": [
    {
      "name": "Company A",
      "customers": "Tech-savvy homeowners seeking automation",
      "product": "Smart home automation platform",
      "categories": "Internet of Things,Smart Home"
    }
  ]
}
```

## Testing

Run the test suite:

```bash
poetry run pytest tests/test_competitors.py -v
```

Tests cover:
- Text cleaning functionality
- Embedding generation
- Graph construction
- Deterministic clustering
- Configuration validation

## Performance

- **Dataset size**: ~100,000 companies
- **Embedding model**: sentence-transformers/all-mpnet-base-v2 (768 dimensions)
- **Memory usage**: ~2-3GB for embeddings + graph
- **Processing time**: ~30-60 minutes for full dataset

## Algorithm Details

### Similarity Computation

1. **Text similarity**: Cosine similarity between weighted embeddings
2. **Category overlap**: Jaccard similarity on Crunchbase categories
3. **Combined score**: 0.8 × text_similarity + 0.2 × category_similarity

### Graph Construction

1. Build FAISS index on normalized embeddings
2. Query k nearest neighbors for each company
3. Keep edges above similarity threshold `tau`
4. Create symmetric sparse adjacency matrix

### Clustering

1. Convert graph to igraph format
2. Apply Leiden algorithm with modularity optimization
3. Extract cluster labels

## Validation Metrics

- **Silhouette score**: Measures cluster cohesion
- **Graph density**: Overall connectivity
- **Intra-cluster density**: Within-cluster connectivity
- **Cluster size distribution**: Balance of cluster sizes

## Future Enhancements

Potential improvements (if needed):
- Multi-resolution clustering
- Ensemble methods
- Geographic/temporal proximity
- Advanced visualization tools
- Performance optimizations for larger datasets

## Dependencies

- sentence-transformers: Text embeddings
- faiss-cpu: Fast similarity search
- leidenalg: Graph clustering
- igraph: Graph operations
- scipy: Sparse matrices
- pandas: Data manipulation
- hydra-core: Configuration management
- loguru: Logging
- pydantic: Configuration validation
