from pydantic import BaseModel
from typing import List


# ── Request ──────────────────────────────────────────────────────────────────

class RAGRequest(BaseModel):
    query: str
    model: str
    ragapproach: str = "vector_search"   # hardcoded default; UI can send anything, backend ignores it

    class Config:
        json_schema_extra = {
            "example": {
                "query": "How do I apply for new electric service?",
                "model": "llama-3.3-70b-versatile",
                "ragapproach": "vector_search"
            }
        }


# ── Response ──────────────────────────────────────────────────────────────────

class Source(BaseModel):
    title: str       # Document name + section heading if available
    url: str         # Page-level deep link: "file:///abs/path/to/doc.pdf#page=42" or HTTP URL
    pageno: str      # Page number as string (e.g. "42" or "42-43" for multi-page chunks)


class Metadata(BaseModel):
    retrievaltimems: int
    generationtimems: int
    totaltimems: int
    generatedat: str       # ISO 8601 datetime string
    inputtokens: int
    outputtokens: int
    totaltokens: int
    modelused: str
    retrievalmethod: str   # always "vector_search" for this prototype


class RAGResponse(BaseModel):
    status: str            # "success" | "error"
    query: str
    answer: str
    sources: List[Source]
    metadata: Metadata

    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "query": "How do I apply for new electric service?",
                "answer": "To apply for new electric service, follow these steps: ...",
                "sources": [
                    {
                        "title": "PG&E Greenbook - Chapter 3: New Service Applications",
                        "url": "file:///data/greenbook.pdf#page=45",
                        "pageno": "45"
                    }
                ],
                "metadata": {
                    "retrievaltimems": 120,
                    "generationtimems": 890,
                    "totaltimems": 1010,
                    "generatedat": "2026-06-12T10:30:00Z",
                    "inputtokens": 1200,
                    "outputtokens": 350,
                    "totaltokens": 1550,
                    "modelused": "llama-3.3-70b-versatile",
                    "retrievalmethod": "vector_search"
                }
            }
        }
