"""
vector_search.py
Retrieves top-k relevant chunks from Qdrant for a given query.
"""

from typing import List, Dict, Any

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

from config import (
    QDRANT_STORAGE_PATH,
    QDRANT_COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    TOP_K,
)

# Module-level singletons — loaded once, reused across requests
_qdrant_client: QdrantClient = None
_embedding_model: SentenceTransformer = None


def _get_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(path=QDRANT_STORAGE_PATH)
    return _qdrant_client


def _get_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


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
