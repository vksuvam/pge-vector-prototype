---
<<<<<<< HEAD
title: VectorRAG
emoji: 🚀
=======
title: Vector RAG
emoji: 🦀
>>>>>>> 14e99041456010573ddf8a60471e92de3d070145
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---
<<<<<<< HEAD

# SP&D RAG Prototype — Vector Search
=======
>>>>>>> 14e99041456010573ddf8a60471e92de3d070145

PG&E SP&D AI Platform POC  
RAG approach: **Vector Search** (Qdrant + sentence-transformers + Groq)

---

## Setup

```bash
# 1. Clone / navigate to project
cd suvam_rag

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your Groq API key in env
GROQ_API_KEY=your_key_here

# 5. Place PDFs in data/
   data/greenbook.pdf
   data/tariff.pdf
```

---

## Run Order

### Step 1 — Ingest PDFs (run once)
```bash
python run_ingestion.py

# Force re-ingest if PDFs changed:
python run_ingestion.py --force
```

### Step 2 — Start API Server
```bash
uvicorn api.main:app --reload
# API available at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### Step 3 — Test manually
```bash
curl -X POST http://localhost:8000/rag \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How do I apply for new electric service?",
    "model": "llama-3.3-70b-versatile",
    "ragapproach": "vector_search"
  }'
```

### Step 4 — Run Red Team Evaluation
```bash
# With API server running in another terminal:
python -m redteam.runner

# Run only adversarial cases:
python -m redteam.runner --category adversarial

# Run only process/steps cases:
python -m redteam.runner --category process_steps
```
Report saved to `redteam/report.json`

---

## Supported Models

| Model ID | Best For |
|---|---|
| `llama-3.3-70b-versatile` | Best quality, general Q&A (default) |
| `llama-3.1-8b-instant` | Fast, lightweight responses |
| `mixtral-8x7b-32768` | Long context, multi-chunk synthesis |
| `gemma2-9b-it` | Benchmarking comparison |

---

## API Contract

**POST /rag**

Request:
```json
{
  "query": "string",
  "model": "string",
  "ragapproach": "string"
}
```

Response:
```json
{
  "status": "success",
  "query": "string",
  "answer": "string",
  "sources": [
    { "title": "string", "url": "string", "pageno": "string" }
  ],
  "metadata": {
    "retrievaltimems": 0,
    "generationtimems": 0,
    "totaltimems": 0,
    "generatedat": "string",
    "inputtokens": 0,
    "outputtokens": 0,
    "totaltokens": 0,
    "modelused": "string",
    "retrievalmethod": "vector_search"
  }
}
```

---

## Notes

- Qdrant runs locally in `qdrant_storage/` — no Docker needed
- Process/steps queries return full verbatim content (no summarization)
- CORS is open for all origins during development — tighten before sharing