# Competitor Clustering Algorithm

This directory contains the clustering algorithm to identify competitors among the 37,500+ companies in our dataset.

## Data Source
- **Input**: `data/processed/edsl_survey.csv` (~37,500 companies)
- **Key Fields**: 
  - `main_customers`: 5-word description of target customers
  - `main_product`: 5-word description of main product/service
  - `category_list`: Crunchbase categories
  - `uuid`: Unique company identifier

## Clustering Approach

### 1. Text Embeddings
- Convert `main_customers` and `main_product` descriptions to embeddings
- Use sentence-transformers for high-quality semantic similarity

### 2. Multi-Modal Similarity
- **Text Similarity**: Cosine similarity between embeddings
- **Category Overlap**: Jaccard similarity on Crunchbase categories
- **Combined Score**: Weighted combination of text + category similarity

### 3. Clustering Algorithm
- **Hierarchical Clustering**: Build competitive clusters at different levels
- **DBSCAN**: Density-based clustering for core competitor groups
- **Validation**: Manual review of sample clusters

### 4. Output
- Competitor clusters with similarity scores
- Cluster summaries and representative companies
- Interactive visualization for exploration

## Files Structure
```
clustering/
├── 01_text_embeddings.py      # Generate embeddings from EDSL output
├── 02_similarity_matrix.py    # Calculate multi-modal similarity
├── 03_clustering.py           # Apply clustering algorithms
├── 04_validation.py           # Validate and analyze clusters
├── 05_visualization.py        # Create interactive visualizations
└── config.py                  # Configuration parameters
```

## Usage
```bash
# Run the complete pipeline
python 01_text_embeddings.py
python 02_similarity_matrix.py
python 03_clustering.py
python 04_validation.py
python 05_visualization.py
``` 