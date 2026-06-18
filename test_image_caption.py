"""Debug: Print all stored image captions."""

from retrieval.image_search import _get_client
from config import QDRANT_IMAGE_COLLECTION_NAME

client = _get_client()

try:
    results = client.scroll(
        collection_name=QDRANT_IMAGE_COLLECTION_NAME,
        limit=10,
    )
    
    points = results[0]  # List of points
    
    print(f"Sample image captions:\n")
    for i, point in enumerate(points, 1):
        payload = point.payload
        print(f"{i}. Image ID: {payload.get('image_id')}")
        print(f"   Page: {payload.get('page_no')}")
        print(f"   Caption: {payload.get('caption')[:100]}...")  # First 100 chars
        print()
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()