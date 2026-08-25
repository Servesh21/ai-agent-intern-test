"""Configuration and settings for the Aster & Row support agent."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "knowledge-base"
ORDERS_FILE = PROJECT_ROOT / "data" / "orders.json"
CHROMA_DB_DIR = PROJECT_ROOT / "chroma_db"

# --- API ---
def get_gemini_api_key() -> str:
    """Get the Gemini API key, failing fast with a clear message if missing."""
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        raise ValueError(
            "GEMINI_API_KEY environment variable is required. "
            "Copy .env.example to .env and add your key."
        )
    return key

# --- Models ---
GENERATION_MODEL = "gemini-3.6-flash"
EMBEDDING_MODEL = "gemini-embedding-001"

# --- Retrieval ---
CHROMA_COLLECTION_NAME = "aster_row_kb"
RETRIEVAL_TOP_K = 8  # Number of chunks to retrieve per query
EMBEDDING_DIMENSION = 3072  # gemini-embedding-001 default

# --- Debug ---
DEBUG = os.getenv("DEBUG", "0") == "1"
