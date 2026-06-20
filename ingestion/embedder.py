"""
embedder.py
Embeds text chunks AND image captions into separate Qdrant collections.

Collection 1: spd_knowledge_base  — text chunks from PDF pages
Collection 2: spd_images          — image captions (searchable by semantic similarity)
"""

from pathlib import Path
from typing import List, Dict, Any

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from config import QDRANT_URL, QDRANT_API_KEY

from config import (
    QDRANT_COLLECTION_NAME,
    QDRANT_IMAGE_COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DIM,
)

BATCH_SIZE = 64


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=30
    )

def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def _collection_exists(client: QdrantClient, name: str) -> bool:
    return any(c.name == name for c in client.get_collections().collections)


def _create_collection(client: QdrantClient, name: str):
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )
    print(f"[Embedder] Collection '{name}' created")


# ── Text Chunks ───────────────────────────────────────────────────────────────

def embed_and_store(chunks: List[Dict[str, Any]], force_reingest: bool = False):
    """Embed text chunks and store in spd_knowledge_base collection."""
    client = get_qdrant_client()
    model = get_embedding_model()

    if _collection_exists(client, QDRANT_COLLECTION_NAME):
        if force_reingest:
            print(f"[Embedder] force_reingest=True — dropping text collection")
            client.delete_collection(QDRANT_COLLECTION_NAME)
            _create_collection(client, QDRANT_COLLECTION_NAME)
        else:
            print(f"[Embedder] Text collection already exists. Skipping.")
            print(f"[Embedder] Pass force_reingest=True to re-ingest.")
            return
    else:
        _create_collection(client, QDRANT_COLLECTION_NAME)

    print(f"[Embedder] Embedding {len(chunks)} text chunks...")

    for batch_start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[batch_start: batch_start + BATCH_SIZE]
        texts = [c["text"] for c in batch]
        embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)

        points = []
        for i, (chunk, vector) in enumerate(zip(batch, embeddings)):
            points.append(PointStruct(
                id=batch_start + i,
                vector=vector.tolist(),
                payload={
                    "text": chunk["text"],
                    "page_no": chunk["page_no"],
                    "source": chunk["source"],
                    "doc_name": chunk["doc_name"],
                    "url": chunk["url"],
                    "chunk_id": chunk["chunk_id"],
                },
            ))

        client.upsert(collection_name=QDRANT_COLLECTION_NAME, points=points)
        processed = min(batch_start + BATCH_SIZE, len(chunks))
        print(f"[Embedder] Stored {processed}/{len(chunks)} text chunks")

    print(f"[Embedder] Text ingestion complete.")


# ── Image Captions ────────────────────────────────────────────────────────────

def embed_and_store_images(image_records: List[Dict[str, Any]], force_reingest: bool = False):
    """
    Embed image captions and store in spd_images collection.
    At query time, captions are searched semantically — matching captions
    tell us which image files to load and return.
    """
    if not image_records:
        print("[Embedder] No image records to store.")
        return

    client = get_qdrant_client()
    model = get_embedding_model()

    if _collection_exists(client, QDRANT_IMAGE_COLLECTION_NAME):
        if force_reingest:
            print(f"[Embedder] force_reingest=True — dropping image collection")
            client.delete_collection(QDRANT_IMAGE_COLLECTION_NAME)
            _create_collection(client, QDRANT_IMAGE_COLLECTION_NAME)
        else:
            print(f"[Embedder] Image collection already exists. Skipping.")
            return
    else:
        _create_collection(client, QDRANT_IMAGE_COLLECTION_NAME)

    print(f"[Embedder] Embedding {len(image_records)} image captions...")

    captions = [r["caption"] for r in image_records]
    embeddings = model.encode(captions, show_progress_bar=False, normalize_embeddings=True)

    points = []
    for i, (record, vector) in enumerate(zip(image_records, embeddings)):
        points.append(PointStruct(
            id=i,
            vector=vector.tolist(),
            payload={
                "image_path": record["image_path"],
                "caption": record["caption"],
                "page_no": record["page_no"],
                "doc_name": record["doc_name"],
                "image_id": record["image_id"],
            },
        ))

    client.upsert(collection_name=QDRANT_IMAGE_COLLECTION_NAME, points=points)
    print(f"[Embedder] Image caption ingestion complete. {len(points)} images stored.")
