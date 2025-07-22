"""
Configuration parameters for competitor clustering algorithm
"""

# Data paths
INPUT_FILE = "../../data/processed/edsl_survey.csv"
OUTPUT_DIR = "../../data/processed/clustering"

# Text embedding parameters
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fast, good quality embeddings
EMBEDDING_BATCH_SIZE = 32

# Similarity weights
TEXT_SIMILARITY_WEIGHT = 0.7
CATEGORY_SIMILARITY_WEIGHT = 0.3

# Clustering parameters
MIN_CLUSTER_SIZE = 3
EPSILON_DBSCAN = 0.3  # Distance threshold for DBSCAN
MIN_SAMPLES_DBSCAN = 2

# Hierarchical clustering
N_CLUSTERS_HIERARCHICAL = 100  # Target number of clusters
LINKAGE_METHOD = "ward"  # Linkage method for hierarchical clustering

# Output parameters
TOP_K_SIMILAR = 10  # Number of most similar companies to show
SIMILARITY_THRESHOLD = 0.6  # Minimum similarity to consider as competitor

# Validation parameters
SAMPLE_SIZE_VALIDATION = 50  # Number of clusters to manually validate 