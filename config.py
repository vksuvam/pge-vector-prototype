import os
from dotenv import load_dotenv
load_dotenv()

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
# QDRANT_STORAGE_PATH = os.path.join(BASE_DIR, "qdrant_storage")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

if not QDRANT_URL:
    raise ValueError("QDRANT_URL environment variable not set")

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
    "GPT-OSS-120B": "Open-weight reasoning model",
    "llama-4-scout-17b-16e-instruct": "Multimodal MoE assistant",
    "qwen3.6-27b" : "Dense coding & reasoning model"
}
DEFAULT_MODEL = "llama-3.3-70b-versatile"

# --- Groq Vision Model (for image captioning during ingestion) ---
# GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
# GROQ_VISION_MODEL = "qwen/qwen3.6-27b"
# GROQ_VISION_MODEL = "openai/gpt-oss-120b"  
# GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"  

IMAGE_SIMILARITY_THRESHOLD = 0.3  

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
