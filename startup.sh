#!/bin/bash

echo "[Startup] ===== SP&D RAG Startup ====="

# Create necessary directories
mkdir -p data/images qdrant_storage

# Check if qdrant collections exist
if [ ! -d "qdrant_storage/collections" ] || [ ! -d "qdrant_storage/collections/spd_knowledge_base" ]; then
    echo "[Startup] Qdrant collections not found"
    echo "[Startup] Running ingestion pipeline..."
    
    # Check if PDFs exist
    if [ ! -f "data/greenbook-manual-full.pdf" ]; then
        echo "[Startup] ERROR: data/greenbook-manual-full.pdf not found!"
        echo "[Startup] Ingestion requires PDFs in data/ directory"
        echo "[Startup] Continuing without ingestion..."
    else
        echo "[Startup] Starting ingestion..."
        python run_ingestion.py --force
        
        if [ $? -eq 0 ]; then
            echo "[Startup] ✓ Ingestion completed successfully"
        else
            echo "[Startup] ✗ Ingestion failed - will continue with empty database"
        fi
    fi
else
    echo "[Startup] ✓ Qdrant collections found, skipping ingestion"
fi

echo "[Startup] ===== Starting Uvicorn Server ====="

# Start the server
exec uvicorn api.main:app --host 0.0.0.0 --port 7860