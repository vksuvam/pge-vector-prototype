import os

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
QDRANT_STORAGE_PATH = os.path.join(BASE_DIR, "qdrant_storage")

GREENBOOK_PDF = os.path.join(DATA_DIR, "greenbook.pdf")
TARIFF_DIR = os.path.join(DATA_DIR, "tariffs")   # folder containing all tariff PDFs
MAX_TARIFF_DOCS = 50                               # sorted by filename, first N used

# --- Qdrant ---
QDRANT_COLLECTION_NAME = "spd_knowledge_base"
EMBEDDING_DIM = 384  # matches all-MiniLM-L6-v2; change if you switch model

# --- Embedding Model ---
# Use BAAI/bge-large-en-v1.5 for better quality on technical docs (dim=1024, update EMBEDDING_DIM)
# Use all-MiniLM-L6-v2 for faster local runs (dim=384)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# --- Chunking ---
CHUNK_SIZE = 512        # tokens per chunk
CHUNK_OVERLAP = 64      # overlap between consecutive chunks

# --- Retrieval ---
TOP_K = 6               # number of chunks to retrieve

# --- Groq Models ---
GROQ_MODELS = {
    "llama-3.3-70b-versatile": "Best quality, general Q&A",
    "llama-3.1-8b-instant": "Fast, lightweight responses",
    "mixtral-8x7b-32768": "Long context, multi-chunk synthesis",
    "gemma2-9b-it": "Alternative for benchmarking",
}
DEFAULT_MODEL = "llama-3.3-70b-versatile"

# --- RAG ---
RAG_APPROACH = "vector_search"  # hardcoded per requirement

# --- Process/Steps Detection ---
# Keywords that trigger verbatim full-process return (no summarization)
PROCESS_KEYWORDS = [
    "steps", "procedure", "process", "how to", "how do i", "how do we",
    "instructions", "guide me", "walk me through", "what do i need to do",
    "apply for", "connect to", "install", "submit", "request", "sign up",
    "get started", "begin", "initiate", "setup", "set up",
]

# --- Groq API ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")  # set in environment
