"""
main.py
FastAPI application. Single endpoint: POST /rag
Accepts RAGRequest, returns RAGResponse — schema is the shared contract with the UI.
"""

import time
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import RAGRequest, RAGResponse, Source, Metadata
from retrieval.vector_search import retrieve
from generation.prompt_builder import build_prompt
from generation.llm_client import generate
from config import RAG_APPROACH, GROQ_MODELS

app = FastAPI(
    title="SP&D RAG API",
    description="Vector Search RAG chatbot for PG&E Service Planning & Design documents",
    version="1.0.0",
)

# Allow React/Next.js frontend on any origin during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # tighten this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/models")
async def list_models():
    """Returns supported models so the UI can populate the model selector."""
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
    1. Retrieve top-k chunks from Qdrant (vector search)
    2. Build prompt (detects process/steps queries)
    3. Generate answer via Groq
    4. Return structured RAGResponse
    """
    total_start = time.time()

    # ── Step 1: Retrieve ────────────────────────────────────────────────────
    retrieval_start = time.time()
    try:
        chunks = retrieve(request.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")

    retrieval_ms = int((time.time() - retrieval_start) * 1000)

    if not chunks:
        # No relevant chunks found — return graceful empty response
        return RAGResponse(
            status="success",
            query=request.query,
            answer="I could not find relevant information in the PG&E documents for your query.",
            sources=[],
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

    # ── Step 2: Build Prompt ────────────────────────────────────────────────
    system_prompt, user_message = build_prompt(request.query, chunks)

    # ── Step 3: Generate ────────────────────────────────────────────────────
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

    # ── Step 4: Build Sources ───────────────────────────────────────────────
    # Deduplicate by page_no + doc_name so we don't list the same page twice
    seen = set()
    sources = []
    for chunk in chunks:
        key = (chunk["doc_name"], chunk["page_no"])
        if key not in seen:
            seen.add(key)
            doc_label = chunk["doc_name"].replace("_", " ").title()
            sources.append(
                Source(
                    title=f"PG&E {doc_label} - Page {chunk['page_no']}",
                    url=chunk["url"],
                    pageno=str(chunk["page_no"]),
                )
            )

    # ── Step 5: Assemble Response ───────────────────────────────────────────
    return RAGResponse(
        status="success",
        query=request.query,
        answer=llm_result["answer"],
        sources=sources,
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
