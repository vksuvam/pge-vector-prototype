"""Test RAG with detailed image retrieval debugging."""

import requests
import json

query = "meter cabinet enclosure clearance diagram figure"

# Step 1: Test image search directly
print("="*60)
print("STEP 1: Direct Image Search")
print("="*60)

from retrieval.image_search import retrieve_images

images = retrieve_images(query, top_k=5)
print(f"Images returned: {len(images)}")
for i, img in enumerate(images, 1):
    print(f"  {i}. Score: {img['score']}, Page: {img['page_no']}")

# Step 2: Test full RAG API
print("\n" + "="*60)
print("STEP 2: Full RAG API")
print("="*60)

response = requests.post(
    "http://localhost:8000/rag",
    json={
        "query": query,
        "model": "llama-3.3-70b-versatile",
        "ragapproach": "vector_search"
    }
)

data = response.json()
print(data)
# # print(f"Status: {data['status']}")
# print(f"Sources: {len(data['sources'])}")
# print(f"Images: {len(data['images'])}")

if data['images']:
    print("Images returned:")
    for i, img in enumerate(data['images'], 1):
        b64_len = len(img.get('image_base64', ''))
        print(f"  {i}. Base64 length: {b64_len} chars")
else:
    print("⚠️  No images in API response!")

print("\nFull response:")
print(json.dumps(data, indent=2, default=str))