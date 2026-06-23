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
import os

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
    allow_origins=["https://main.d1u5v42dfn8m2v.amplifyapp.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("data", exist_ok=True)
app.mount("/docs", StaticFiles(directory="data"), name="docs")

# Serve PDFs and images from data/ folder
# /docs/greenbook-manual-full.pdf#page=45  → opens PDF at page 45 in browser
app.mount("/docs", StaticFiles(directory=DATA_DIR), name="docs")

@app.get("/")
def root():
    return {
        "status": "healthy",
        "service": "PGE Vector RAG"
    }

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


# @app.post("/rag", response_model=RAGResponse)
# async def rag_retrieval(request: RAGRequest):
#     """
#     Main RAG endpoint.

#     Flow:
#     1. Retrieve top-k text chunks from Qdrant (vector search)
#     2. Retrieve relevant images via caption-based semantic search
#     3. Build prompt (detects process/steps queries)
#     4. Generate answer via Groq
#     5. Return RAGResponse with answer + sources + images
#     """
#     total_start = time.time()

#     # ── Step 1: Retrieve text chunks ─────────────────────────────────────────
#     retrieval_start = time.time()
#     try:
#         chunks = retrieve(request.query)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Text retrieval failed: {str(e)}")

#     retrieval_ms = int((time.time() - retrieval_start) * 1000)

#     # ── Step 2: Retrieve relevant images ─────────────────────────────────────
#     try:
#         image_hits = retrieve_images(request.query)
#     except Exception as e:
#         print(f"[API] Image retrieval failed (non-fatal): {e}")
#         image_hits = []

#     # ── Handle empty retrieval ────────────────────────────────────────────────
#     if not chunks:
#         return RAGResponse(
#             status="success",
#             query=request.query,
#             answer="I could not find relevant information in the PG&E documents for your query.",
#             sources=[],
#             images=[],
#             metadata=Metadata(
#                 retrievaltimems=retrieval_ms,
#                 generationtimems=0,
#                 totaltimems=retrieval_ms,
#                 generatedat=datetime.now(timezone.utc).isoformat(),
#                 inputtokens=0,
#                 outputtokens=0,
#                 totaltokens=0,
#                 modelused=request.model,
#                 retrievalmethod=RAG_APPROACH,
#             ),
#         )

#     # ── Step 3: Build prompt ──────────────────────────────────────────────────
#     system_prompt, user_message = build_prompt(request.query, chunks)

#     # ── Step 4: Generate answer ───────────────────────────────────────────────
#     generation_start = time.time()
#     try:
#         llm_result = generate(
#             system_prompt=system_prompt,
#             user_message=user_message,
#             model=request.model,
#         )
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

#     generation_ms = int((time.time() - generation_start) * 1000)
#     total_ms = int((time.time() - total_start) * 1000)

#     # ── Step 5: Build sources (deduplicated by page) ──────────────────────────
#     seen = set()
#     sources = []
#     for chunk in chunks:
#         key = (chunk["doc_name"], chunk["page_no"])
#         if key not in seen:
#             seen.add(key)
#             doc_label = chunk["doc_name"].replace("-", " ").replace("_", " ").title()
#             sources.append(Source(
#                 title=f"PG&E {doc_label} - Page {chunk['page_no']}",
#                 url=chunk["url"],
#                 pageno=str(chunk["page_no"]),
#             ))

#     # ── Step 6: Build image results ───────────────────────────────────────────
#     # ── Step 6: Build image results ───────────────────────────────────────────
#     images = []
#     print(f"[API] Processing {len(image_hits)} image hits...")
#     for i, hit in enumerate(image_hits):
#         try:
#             print(f"[API] Hit {i+1}: keys={list(hit.keys())}")
#             b64 = hit.get("image_base64", "")
#             if not b64:
#                 print(f"[API] Warning: No image_base64 for hit {i+1}")
#                 continue
#             print(f"[API] Building ImageResult with {len(b64)} chars of base64")
#             images.append(ImageResult(image_base64=b64))
#         except Exception as e:
#             print(f"[API] Error on hit {i+1}: {type(e).__name__}: {e}")
#             import traceback
#             traceback.print_exc()
#             continue

#     print(f"[API] Successfully built {len(images)} ImageResult objects")

#     # ── Step 7: Assemble response ─────────────────────────────────────────────
#     return RAGResponse(
#         status="success",
#         query=request.query,
#         answer=llm_result["answer"],
#         sources=sources,
#         images=images,
#         metadata=Metadata(
#             retrievaltimems=retrieval_ms,
#             generationtimems=generation_ms,
#             totaltimems=total_ms,
#             generatedat=datetime.now(timezone.utc).isoformat(),
#             inputtokens=llm_result["input_tokens"],
#             outputtokens=llm_result["output_tokens"],
#             totaltokens=llm_result["total_tokens"],
#             modelused=llm_result["model_used"],
#             retrievalmethod=RAG_APPROACH,
#         ),
#     )


@app.post("/rag", response_model=RAGResponse)
async def rag_retrieval(request: RAGRequest):
    """Main RAG endpoint with detailed logging."""
    total_start = time.time()
    
    try:
        print(f"\n[API] ===== NEW REQUEST =====")
        print(f"[API] Query: {request.query}")
        print(f"[API] Model: {request.model}")

        # ── Step 1: Retrieve text chunks ─────────────────────────────────────────
        print(f"[API] Step 1: Retrieving text chunks...")
        retrieval_start = time.time()
        try:
            chunks = retrieve(request.query)
            print(f"[API] ✓ Text retrieval succeeded: {len(chunks)} chunks")
        except Exception as e:
            print(f"[API] ✗ Text retrieval failed: {e}")
            raise HTTPException(status_code=500, detail=f"Text retrieval failed: {str(e)}")

        retrieval_ms = int((time.time() - retrieval_start) * 1000)

        # ── Step 2: Retrieve relevant images ─────────────────────────────────────
        print(f"[API] Step 2: Retrieving images...")
        image_hits = []
        try:
            # Try to get images, but don't fail if Qdrant is locked
            image_hits = retrieve_images(request.query)
            print(f"[API] ✓ Image retrieval succeeded: {len(image_hits)} images")
        except Exception as e:
            error_msg = str(e)
            if "already accessed by another instance" in error_msg:
                print(f"[API] ⚠️  Image retrieval: Qdrant locked (expected during dev), skipping images")
            else:
                print(f"[API] ✗ Image retrieval failed (non-fatal): {e}")
            image_hits = []

        # ── Handle empty retrieval ────────────────────────────────────────────────
        print(f"[API] Step 3: Checking if we have content...")
        if not chunks:
            print(f"[API] No chunks found, returning empty response")
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

        # ── Step 4: Build prompt ──────────────────────────────────────────────────
        print(f"[API] Step 4: Building prompt...")
        try:
            system_prompt, user_message = build_prompt(request.query, chunks)
            print(f"[API] ✓ Prompt built successfully")
        except Exception as e:
            print(f"[API] ✗ Prompt building failed: {e}")
            raise HTTPException(status_code=500, detail=f"Prompt building failed: {str(e)}")

        # ── Step 5: Generate answer ───────────────────────────────────────────────
        print(f"[API] Step 5: Generating answer...")
        generation_start = time.time()
        try:
            llm_result = generate(
                system_prompt=system_prompt,
                user_message=user_message,
                model=request.model,
            )
            print(f"[API] ✓ Generation succeeded")
        except Exception as e:
            print(f"[API] ✗ Generation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

        generation_ms = int((time.time() - generation_start) * 1000)
        total_ms = int((time.time() - total_start) * 1000)

        # ── Step 6: Build sources ─────────────────────────────────────────────────
        print(f"[API] Step 6: Building sources...")
        try:
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
            print(f"[API] ✓ Built {len(sources)} sources")
        except Exception as e:
            print(f"[API] ✗ Source building failed: {e}")
            raise HTTPException(status_code=500, detail=f"Source building failed: {str(e)}")

        # ── Step 7: Build image results ───────────────────────────────────────────
        print(f"[API] Step 7: Building image results from {len(image_hits)} hits...")
        try:
            images = []
            for i, hit in enumerate(image_hits):
                print(f"[API]   Image {i+1}: keys={list(hit.keys())}")
                b64 = hit.get("image_base64")
                if not b64:
                    print(f"[API]   ⚠️  No image_base64 for hit {i+1}, skipping")
                    continue
                print(f"[API]   ✓ Creating ImageResult ({len(b64)} chars)")
                images.append(ImageResult(image_base64=b64))
            print(f"[API] ✓ Built {len(images)} ImageResult objects")
        except Exception as e:
            print(f"[API] ✗ Image building failed: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Image building failed: {str(e)}")

        # ── Step 8: Assemble response ─────────────────────────────────────────────
        print(f"[API] Step 8: Assembling final response...")
        try:
            response = RAGResponse(
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
            print(f"[API] ✓ Response assembled successfully")
            print(f"[API] ===== REQUEST COMPLETE =====\n")
            return response
        except Exception as e:
            print(f"[API] ✗ Response assembly failed: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Response assembly failed: {str(e)}")
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"[API] ✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")