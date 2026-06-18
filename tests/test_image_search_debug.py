"""Debug image search with similarity scores."""

from retrieval.image_search import _get_client, _get_model
from config import QDRANT_IMAGE_COLLECTION_NAME

client = _get_client()
model = _get_model()

query = "meter cabinet enclosure clearance diagram"
query_vector = model.encode(query, normalize_embeddings=True).tolist()

results = client.search(
    collection_name=QDRANT_IMAGE_COLLECTION_NAME,
    query_vector=query_vector,
    limit=10,
    with_payload=True,
)

print(f"Query: {query}\n")
print(f"Top 10 image matches:\n")

for i, hit in enumerate(results, 1):
    payload = hit.payload
    print(f"{i}. Score: {hit.score:.4f}")
    print(f"   Image ID: {payload.get('image_id')}")
    print(f"   Page: {payload.get('page_no')}")
    print(f"   Caption: {payload.get('caption')[:80]}...")
    print() 