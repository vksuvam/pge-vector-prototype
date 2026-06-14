"""
embedder.py
Embeds text chunks using sentence-transformers and stores them in local Qdrant.

Local Qdrant (no Docker): uses qdrant_client with path= pointing to a local directory.
Data persists between runs. Re-running this script will skip if collection exists,
unless force_reingest=True.
"""

from typing import List, Dict, Any
from pathlib import Path

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)

from config import (
    QDRANT_STORAGE_PATH,
    QDRANT_COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DIM,
)

# Batch size for embedding + uploading — reduce if you hit memory limits
BATCH_SIZE = 64


def get_qdrant_client() -> QdrantClient:
    """Returns a local persistent Qdrant client (no Docker required)."""
    Path(QDRANT_STORAGE_PATH).mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=QDRANT_STORAGE_PATH)


def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def collection_exists(client: QdrantClient) -> bool:
    collections = client.get_collections().collections
    return any(c.name == QDRANT_COLLECTION_NAME for c in collections)


def create_collection(client: QdrantClient):
    client.create_collection(
        collection_name=QDRANT_COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )
    print(f"[Embedder] Collection '{QDRANT_COLLECTION_NAME}' created")


def embed_and_store(
    chunks: List[Dict[str, Any]],
    force_reingest: bool = False,
):
    """
    Embeds all chunks and upserts into Qdrant.

    Args:
        chunks: list of chunk dicts from chunker.py
        force_reingest: if True, drops and recreates the collection
    """
    client = get_qdrant_client()
    model = get_embedding_model()

    # Handle existing collection
    if collection_exists(client):
        if force_reingest:
            print(f"[Embedder] force_reingest=True — dropping existing collection")
            client.delete_collection(QDRANT_COLLECTION_NAME)
            create_collection(client)
        else:
            print(f"[Embedder] Collection already exists. Skipping ingestion.")
            print(f"[Embedder] Pass force_reingest=True to re-ingest.")
            return
    else:
        create_collection(client)

    print(f"[Embedder] Embedding {len(chunks)} chunks with '{EMBEDDING_MODEL_NAME}'...")

    for batch_start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[batch_start: batch_start + BATCH_SIZE]
        texts = [c["text"] for c in batch]

        embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)

        points = []
        for i, (chunk, vector) in enumerate(zip(batch, embeddings)):
            point_id = batch_start + i  # sequential integer ID

            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector.tolist(),
                    payload={
                        "text": chunk["text"],
                        "page_no": chunk["page_no"],
                        "source": chunk["source"],
                        "doc_name": chunk["doc_name"],
                        "url": chunk["url"],
                        "chunk_id": chunk["chunk_id"],
                    },
                )
            )

        client.upsert(collection_name=QDRANT_COLLECTION_NAME, points=points)

        processed = min(batch_start + BATCH_SIZE, len(chunks))
        print(f"[Embedder] Stored {processed}/{len(chunks)} chunks")

    print(f"[Embedder] Ingestion complete. {len(chunks)} chunks in Qdrant.")
