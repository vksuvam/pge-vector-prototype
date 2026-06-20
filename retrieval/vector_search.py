"""
vector_search.py
Retrieves top-k relevant chunks from Qdrant for a given query.
"""

from typing import List, Dict, Any

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from config import QDRANT_URL, QDRANT_API_KEY

from config import (
    QDRANT_COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    TOP_K
)

# Module-level singletons — loaded once, reused across requests
_qdrant_client: QdrantClient = None
_embedding_model: SentenceTransformer = None

def _get_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        print(f"[VectorSearch] Connecting to Qdrant Cloud: {QDRANT_URL}")
        _qdrant_client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            timeout=30  # Add timeout for cloud requests
        )
        print("[VectorSearch] ✓ Connected to Qdrant Cloud")
    return _qdrant_client

def _get_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model

TEXT_SIMILARITY_THRESHOLD = 0.25  # Minimum cosine similarity for text chunks

def retrieve(query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    """
    Embed the query and retrieve top-k chunks from Qdrant.

    Returns a list of dicts, each containing:
    {
        "text":     str,
        "page_no":  int,
        "doc_name": str,
        "source":   str,
        "url":      str,
        "chunk_id": str,
        "score":    float,   # cosine similarity score
    }
    Sorted by score descending (most relevant first).
    """
    model = _get_model()
    client = _get_client()

    query_vector = model.encode(query, normalize_embeddings=True).tolist()

    results = client.search(
        collection_name=QDRANT_COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k,
        with_payload=True,
    )

    chunks = []
    for hit in results:
        if hit.score < TEXT_SIMILARITY_THRESHOLD:
            print(f"[Retrieval] Skipping low-score chunk (score={hit.score})")
            continue

        payload = hit.payload
        chunks.append({
            "text": payload.get("text", ""), 
            "page_no": payload.get("page_no", 0),
            "doc_name": payload.get("doc_name", ""),
            "source": payload.get("source", ""),
            "url": payload.get("url", ""),
            "chunk_id": payload.get("chunk_id", ""),
            "score": round(hit.score, 4),
        })

    return chunks
