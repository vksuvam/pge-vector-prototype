"""
image_search.py
Caption-based semantic image retrieval from Qdrant.

Flow:
1. Embed the user query
2. Search spd_images collection for captions semantically similar to the query
3. For matching captions, load the image file from disk
4. Encode image as base64 data URI
5. Return list of ImageResult-ready dicts

This means images are only returned when the query is genuinely relevant
to what the image depicts — not just because it's on the same page.
"""

import os
import base64
from typing import List, Dict, Any

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from config import QDRANT_URL, QDRANT_API_KEY

from config import (
    QDRANT_IMAGE_COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    TOP_K_IMAGES
)

# Module-level singletons
_qdrant_client: QdrantClient = None
_embedding_model: SentenceTransformer = None

# Minimum similarity score for an image to be returned
# Below this threshold, the image caption isn't relevant enough to the query
IMAGE_SIMILARITY_THRESHOLD = 0.35    


def _get_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            timeout=30
        )
    return _qdrant_client


def _get_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def _image_collection_exists(client: QdrantClient) -> bool:
    return any(
        c.name == QDRANT_IMAGE_COLLECTION_NAME
        for c in client.get_collections().collections
    )


def _encode_image_file(image_path: str) -> str:
    """Read image from disk and return base64 data URI."""
    with open(image_path, "rb") as f:
        raw = f.read()
    b64 = base64.b64encode(raw).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def retrieve_images(query: str, top_k: int = TOP_K_IMAGES) -> List[Dict[str, Any]]:
    """
    Search image captions semantically and return matching images as base64.

    Returns list of dicts:
    {
        "image_base64": str,   # "data:image/png;base64,..."
        "caption":      str,
        "page_no":      int,
        "doc_name":     str,
        "score":        float,
    }

    Returns empty list if:
    - Image collection doesn't exist yet (ingestion not run)
    - No captions are similar enough to the query
    - Image file missing from disk
    """
    client = _get_client()

    # Gracefully handle case where image ingestion hasn't been run
    if not _image_collection_exists(client):
        print("[ImageSearch] Image collection not found — skipping image retrieval")
        return []

    model = _get_model()
    query_vector = model.encode(query, normalize_embeddings=True).tolist()

    results = client.search(
        collection_name=QDRANT_IMAGE_COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k,
        with_payload=True,
    )

    images = []
    for hit in results:
        # Filter by similarity threshold
        if hit.score < IMAGE_SIMILARITY_THRESHOLD:
            continue

        payload = hit.payload
        image_path = payload.get("image_path", "")

        # Skip if image file was deleted or moved
        if not os.path.exists(image_path):
            print(f"[ImageSearch] Image file not found: {image_path}")
            continue

        try:
            image_base64 = _encode_image_file(image_path)
            images.append({
                "image_base64": image_base64,
                "caption": payload.get("caption", ""),
                "page_no": payload.get("page_no", 0),
                "doc_name": payload.get("doc_name", ""),
                "score": round(hit.score, 4),
            })
        except Exception as e:
            print(f"[ImageSearch] Failed to encode image {image_path}: {e}")
            continue

    print(f"[ImageSearch] Returning {len(images)} relevant images for query")
    return images
