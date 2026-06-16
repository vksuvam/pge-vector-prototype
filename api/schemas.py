from pydantic import BaseModel
from typing import List


# ── Request ──────────────────────────────────────────────────────────────────

class RAGRequest(BaseModel):
    query: str
    model: str
    ragapproach: str = "vector_search"

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What are the clearance requirements for transformers?",
                "model": "llama-3.3-70b-versatile",
                "ragapproach": "vector_search"
            }
        }


# ── Response ──────────────────────────────────────────────────────────────────

class Source(BaseModel):
    title: str
    url: str        # e.g. "/docs/greenbook-manual-full.pdf#page=45"
    pageno: str


class ImageResult(BaseModel):
    image_base64: str   # "data:image/png;base64,..." — drop directly into <img src>


class Metadata(BaseModel):
    retrievaltimems: int
    generationtimems: int
    totaltimems: int
    generatedat: str
    inputtokens: int
    outputtokens: int
    totaltokens: int
    modelused: str
    retrievalmethod: str


class RAGResponse(BaseModel):
    status: str
    query: str
    answer: str
    sources: List[Source]
    images: List[ImageResult]   # empty list if no relevant images found
    metadata: Metadata
