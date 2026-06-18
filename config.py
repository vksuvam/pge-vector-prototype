import os
from dotenv import load_dotenv
load_dotenv()

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
QDRANT_STORAGE_PATH = os.path.join(BASE_DIR, "qdrant_storage")
IMAGES_DIR = os.path.join(DATA_DIR, "images")   # extracted PDF images stored here

GREENBOOK_PDF = os.path.join(DATA_DIR, "greenbook-manual-full.pdf")
TARIFF_DIR = os.path.join(DATA_DIR, "tariffs")
MAX_TARIFF_DOCS = 50

# --- Qdrant ---
QDRANT_COLLECTION_NAME = "spd_knowledge_base"
QDRANT_IMAGE_COLLECTION_NAME = "spd_images"   # separate collection for image captions
EMBEDDING_DIM = 384  # matches all-MiniLM-L6-v2

# --- Embedding Model ---
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# --- Chunking ---
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

# --- Retrieval ---
TOP_K = 6
TOP_K_IMAGES = 3    # max images to return per response

# --- Groq Models (text) ---
GROQ_MODELS = {
    "llama-3.3-70b-versatile": "Best quality, general Q&A",
    "llama-3.1-8b-instant": "Fast, lightweight responses",
    "mixtral-8x7b-32768": "Long context, multi-chunk synthesis",
    "gemma2-9b-it": "Alternative for benchmarking",
}
DEFAULT_MODEL = "llama-3.3-70b-versatile"

# --- Groq Vision Model (for image captioning during ingestion) ---
GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
# GROQ_VISION_MODEL = "llama-3.2-90b-vision-preview"  # or another vision model Groq supports
# IMAGE_SIMILARITY_THRESHOLD = 0.35  # (or tune based on your testing)

# --- RAG ---
RAG_APPROACH = "vector_search"

# --- Process/Steps Detection ---
PROCESS_KEYWORDS = [
    "steps", "procedure", "process", "how to", "how do i", "how do we",
    "instructions", "guide me", "walk me through", "what do i need to do",
    "apply for", "connect to", "install", "submit", "request", "sign up",
    "get started", "begin", "initiate", "setup", "set up",
]

# --- Groq API ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
