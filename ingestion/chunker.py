"""
chunker.py
Splits page text into overlapping chunks for embedding.

Strategy: sentence-aware chunking with token-based size control.
Why not fixed character splits? PG&E Greenbook has dense technical paragraphs
where cutting mid-sentence loses critical context (e.g., clearance requirements,
voltage specs). Sentence boundaries are respected first, then token limits applied.
"""

from typing import List, Dict, Any
import re

from config import CHUNK_SIZE, CHUNK_OVERLAP


def _split_into_sentences(text: str) -> List[str]:
    """
    Splits text into sentences using a simple regex.
    Handles abbreviations poorly (e.g. "Fig. 3") — acceptable trade-off for now.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _approx_token_count(text: str) -> int:
    """Rough token estimate: 1 token ≈ 4 characters (good enough for chunking)."""
    return len(text) // 4


def chunk_page(page: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Takes a single page record and returns a list of chunk records.

    Each chunk record:
    {
        "text":      str,
        "page_no":   int,
        "source":    str,
        "doc_name":  str,
        "url":       str,
        "chunk_id":  str,   # e.g. "greenbook_p45_c2"
    }
    """
    sentences = _split_into_sentences(page["text"])
    chunks = []
    current_sentences = []
    current_tokens = 0
    chunk_index = 0

    for sentence in sentences:
        sentence_tokens = _approx_token_count(sentence)

        # If adding this sentence exceeds chunk size, flush current chunk
        if current_tokens + sentence_tokens > CHUNK_SIZE and current_sentences:
            chunk_text = " ".join(current_sentences)
            chunks.append(_make_chunk(chunk_text, page, chunk_index))
            chunk_index += 1

            # Overlap: keep last N tokens worth of sentences
            overlap_sentences = []
            overlap_tokens = 0
            for s in reversed(current_sentences):
                t = _approx_token_count(s)
                if overlap_tokens + t <= CHUNK_OVERLAP:
                    overlap_sentences.insert(0, s)
                    overlap_tokens += t
                else:
                    break
            current_sentences = overlap_sentences
            current_tokens = overlap_tokens

        current_sentences.append(sentence)
        current_tokens += sentence_tokens

    # Flush remaining
    if current_sentences:
        chunk_text = " ".join(current_sentences)
        chunks.append(_make_chunk(chunk_text, page, chunk_index))

    return chunks


def _make_chunk(text: str, page: Dict[str, Any], index: int) -> Dict[str, Any]:
    return {
        "text": text,
        "page_no": page["page_no"],
        "source": page["source"],
        "doc_name": page["doc_name"],
        "url": page["url"],
        "chunk_id": f"{page['doc_name']}_p{page['page_no']}_c{index}",
    }


def chunk_pages(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Chunk all pages from all documents."""
    all_chunks = []
    for page in pages:
        all_chunks.extend(chunk_page(page))
    print(f"[Chunker] Total chunks created: {len(all_chunks)}")
    return all_chunks
