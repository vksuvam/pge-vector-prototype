"""
main.py
FastAPI application. Single endpoint: POST /rag
Accepts RAGRequest, returns RAGResponse.

Static file mounts:
  /docs  → serves data/ folder (PDFs accessible as clickable links)
"""

import time
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.schemas import RAGRequest, RAGResponse, Source, Metadata, ImageResult
from retrieval.vector_search import retrieve
from retrieval.image_search import retrieve_images
from generation.prompt_builder import build_prompt
from generation.llm_client import generate
from config import RAG_APPROACH, GROQ_MODELS, DATA_DIR

app = FastAPI(
    title="SP&D RAG API",
    description="Vector Search RAG chatbot for PG&E Service Planning & Design documents",
    version="1.0.0",
)

# CORS — allow React/Next.js frontend during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve PDFs and images from data/ folder
# /docs/greenbook-manual-full.pdf#page=45  → opens PDF at page 45 in browser
app.mount("/docs", StaticFiles(directory=DATA_DIR), name="docs")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/models")
async def list_models():
    """Returns supported models for the UI model selector dropdown."""
    return {
        "models": [
            {"id": model_id, "description": desc}
            for model_id, desc in GROQ_MODELS.items()
        ]
    }


@app.post("/rag", response_model=RAGResponse)
async def rag_retrieval(request: RAGRequest):
    """
    Main RAG endpoint.

    Flow:
    1. Retrieve top-k text chunks from Qdrant (vector search)
    2. Retrieve relevant images via caption-based semantic search
    3. Build prompt (detects process/steps queries)
    4. Generate answer via Groq
    5. Return RAGResponse with answer + sources + images
    """
    total_start = time.time()

    # ── Step 1: Retrieve text chunks ─────────────────────────────────────────
    retrieval_start = time.time()
    try:
        chunks = retrieve(request.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text retrieval failed: {str(e)}")

    retrieval_ms = int((time.time() - retrieval_start) * 1000)

    # ── Step 2: Retrieve relevant images ─────────────────────────────────────
    try:
        image_hits = retrieve_images(request.query)
    except Exception as e:
        print(f"[API] Image retrieval failed (non-fatal): {e}")
        image_hits = []

    # ── Handle empty retrieval ────────────────────────────────────────────────
    if not chunks:
        return RAGResponse(
            status="success",
            query=request.query,
            answer="I could not find relevant information in the PG&E documents for your query.",
            sources=[],
            images=[],
            metadata=Metadata(
                retrievaltimems=retrieval_ms,
                generationtimems=0,
                totaltimems=retrieval_ms,
                generatedat=datetime.now(timezone.utc).isoformat(),
                inputtokens=0,
                outputtokens=0,
                totaltokens=0,
                modelused=request.model,
                retrievalmethod=RAG_APPROACH,
            ),
        )

    # ── Step 3: Build prompt ──────────────────────────────────────────────────
    system_prompt, user_message = build_prompt(request.query, chunks)

    # ── Step 4: Generate answer ───────────────────────────────────────────────
    generation_start = time.time()
    try:
        llm_result = generate(
            system_prompt=system_prompt,
            user_message=user_message,
            model=request.model,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

    generation_ms = int((time.time() - generation_start) * 1000)
    total_ms = int((time.time() - total_start) * 1000)

    # ── Step 5: Build sources (deduplicated by page) ──────────────────────────
    seen = set()
    sources = []
    for chunk in chunks:
        key = (chunk["doc_name"], chunk["page_no"])
        if key not in seen:
            seen.add(key)
            doc_label = chunk["doc_name"].replace("-", " ").replace("_", " ").title()
            sources.append(Source(
                title=f"PG&E {doc_label} - Page {chunk['page_no']}",
                url=chunk["url"],
                pageno=str(chunk["page_no"]),
            ))

    # ── Step 6: Build image results ───────────────────────────────────────────
    images = [
        ImageResult(image_base64=hit["image_base64"])
        for hit in image_hits
    ]

    # ── Step 7: Assemble response ─────────────────────────────────────────────
    return RAGResponse(
        status="success",
        query=request.query,
        answer=llm_result["answer"],
        sources=sources,
        images=images,
        metadata=Metadata(
            retrievaltimems=retrieval_ms,
            generationtimems=generation_ms,
            totaltimems=total_ms,
            generatedat=datetime.now(timezone.utc).isoformat(),
            inputtokens=llm_result["input_tokens"],
            outputtokens=llm_result["output_tokens"],
            totaltokens=llm_result["total_tokens"],
            modelused=llm_result["model_used"],
            retrievalmethod=RAG_APPROACH,
        ),
    )
