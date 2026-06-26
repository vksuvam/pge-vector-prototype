"""
image_search.py
Caption-based semantic image retrieval from Qdrant.

Flow:
1. Embed the user query
2. Search spd_images collection for captions semantically similar to the query
3. For matching captions, load the image file from disk
4. Encode image as base64 data URI
5. Return list of ImageResult-ready dicts

Images are only returned when the query is genuinely relevant to what
the image depicts — not just because it's on the same page.
"""

import os
import base64
from typing import List, Dict, Any

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

from config import (
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_IMAGE_COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    TOP_K_IMAGES,
    IMAGE_SIMILARITY_THRESHOLD,
    BASE_DIR,  # absolute path to project root
)

# Module-level singletons
_qdrant_client: QdrantClient = None
_embedding_model: SentenceTransformer = None


def _get_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            timeout=30,
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


def _resolve_image_path(stored_path: str) -> str:
    """
    Convert stored relative path like "data/images/greenbook_p3_img1.png"
    to absolute path anchored at BASE_DIR.

    Why: image_path is stored as relative during ingestion. At query time,
    the working directory may differ (e.g., on HuggingFace), so
    os.path.exists("data/images/...") fails silently. Anchoring to BASE_DIR
    makes the lookup reliable regardless of cwd.

    Args:
        stored_path: relative or absolute path from Qdrant payload

    Returns:
        absolute path to the image file
    """
    if os.path.isabs(stored_path):
        return stored_path
    return os.path.join(BASE_DIR, stored_path)


def _encode_image_file(abs_path: str) -> str:
    """Read image from disk and return base64 data URI."""
    with open(abs_path, "rb") as f:
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
    - No captions are similar enough to the query (below IMAGE_SIMILARITY_THRESHOLD)
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
        score = hit.score

        # Filter by similarity threshold (from config.py)
        if score < IMAGE_SIMILARITY_THRESHOLD:
            print(
                f"[ImageSearch] Skipping low-score image "
                f"(score={score:.4f} < threshold={IMAGE_SIMILARITY_THRESHOLD})"
            )
            continue

        payload = hit.payload
        stored_path = payload.get("image_path", "")

        # Resolve relative path to absolute (critical for HuggingFace compatibility)
        abs_path = _resolve_image_path(stored_path)

        # Skip if image file was deleted or moved
        if not os.path.exists(abs_path):
            print(f"[ImageSearch] Image file not found on disk: {abs_path}")
            continue

        try:
            image_base64 = _encode_image_file(abs_path)
            images.append({
                "image_base64": image_base64,
                "caption": payload.get("caption", ""),
                "page_no": payload.get("page_no", 0),
                "doc_name": payload.get("doc_name", ""),
                "score": round(score, 4),
            })
        except Exception as e:
            print(f"[ImageSearch] Failed to encode image {abs_path}: {e}")
            continue

    print(f"[ImageSearch] Returning {len(images)} relevant images for query")
    return images