#!/bin/bash

echo "[Startup] ===== SP&D RAG Startup ====="

# Create necessary directories
mkdir -p data/images

# Check if Qdrant Cloud has collections
echo "[Startup] Checking Qdrant Cloud collections..."

python -c "
import sys
from qdrant_client import QdrantClient
from config import QDRANT_URL, QDRANT_API_KEY, QDRANT_COLLECTION_NAME

try:
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=10)
    collections = [c.name for c in client.get_collections().collections]
    
    if QDRANT_COLLECTION_NAME in collections:
        print('[Startup] ✓ Qdrant collections found')
        sys.exit(0)
    else:
        print('[Startup] Collections missing, will ingest')
        sys.exit(1)
except Exception as e:
    print(f'[Startup] Cannot reach Qdrant: {e}')
    sys.exit(1)
" || {
    echo "[Startup] Running ingestion pipeline..."
    python run_ingestion.py --force
    
    if [ $? -eq 0 ]; then
        echo "[Startup] ✓ Ingestion completed successfully"
    else
        echo "[Startup] ✗ Ingestion failed"
    fi
}

echo "[Startup] ===== Starting Uvicorn Server ====="
exec uvicorn api.main:app --host 0.0.0.0 --port 7860