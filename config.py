"""
Financial RAG Analyst - Configuration Module
Centralized parameters for chunking, vector indexing, and model inference.
"""

from pathlib import Path
import os

# Base paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "sample_data"
CHROMA_DIR = BASE_DIR / "chroma_db"

# Ensure data directory exists
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Default sample SEC filing path
SAMPLE_FILING_PATH = DATA_DIR / "sec_10k_sample.txt"

# Text chunking configuration
# SEC filings are dense, so 500-char chunks with 100-char overlap preserve context
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# Vector retrieval settings
DEFAULT_TOP_K = 3
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_COLLECTION_NAME = "financial_filings"

# App Metadata
APP_TITLE = "Financial & SEC 10-K RAG Analyst"
APP_SUBTITLE = "Verifiable Document Intelligence with Exact Page & Section Citations"
AUTHOR = "Gaurav"
GITHUB_REPO = "https://github.com/grvv-vv/financial-rag-analyst"
