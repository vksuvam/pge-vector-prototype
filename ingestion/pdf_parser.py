"""
pdf_parser.py
Extracts text page-by-page from PDFs, preserving page numbers and source metadata.
"""

import os
from pathlib import Path
from typing import List, Dict, Any

import pdfplumber


def _make_page_url(pdf_path: str, page_number: int) -> str:
    """
    Builds a relative URL for the page, served via FastAPI static files mount.
    Greenbook:  /docs/greenbook-manual-full.pdf#page=45
    Tariffs:    /docs/tariffs/E-1.pdf#page=3
    """
    filename = Path(pdf_path).name
    parent = Path(pdf_path).parent.name

    if parent == "tariffs":
        return f"/docs/tariffs/{filename}#page={page_number}"
    else:
        return f"/docs/{filename}#page={page_number}"


def parse_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Parse a PDF and return a list of page records.

    Each record:
    {
        "text":      str,
        "page_no":   int,   # 1-based
        "source":    str,   # absolute file path
        "doc_name":  str,   # filename without extension
        "url":       str,   # /docs/... clickable link
    }
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc_name = Path(pdf_path).stem
    pages = []

    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        print(f"[Parser] {doc_name}: {total} pages found")

        for i, page in enumerate(pdf.pages):
            page_no = i + 1
            text = page.extract_text()

            if not text or not text.strip():
                print(f"[Parser] Skipping page {page_no} (no extractable text)")
                continue

            pages.append({
                "text": text.strip(),
                "page_no": page_no,
                "source": str(Path(pdf_path).resolve()),
                "doc_name": doc_name,
                "url": _make_page_url(pdf_path, page_no),
            })

    print(f"[Parser] {doc_name}: {len(pages)} pages extracted")
    return pages


def parse_all_pdfs(pdf_paths: List[str]) -> List[Dict[str, Any]]:
    """Parse multiple PDFs and return combined page list."""
    all_pages = []
    for path in pdf_paths:
        all_pages.extend(parse_pdf(path))
    return all_pages


def get_tariff_pdfs(tariff_dir: str, max_docs: int) -> List[str]:
    """
    Returns sorted list of tariff PDF paths, capped at max_docs.
    """
    if not os.path.exists(tariff_dir):
        raise FileNotFoundError(
            f"Tariff directory not found: {tariff_dir}\n"
            f"Create data/tariffs/ and place your tariff PDFs there."
        )

    all_pdfs = sorted([
        os.path.join(tariff_dir, f)
        for f in os.listdir(tariff_dir)
        if f.lower().endswith(".pdf")
    ])

    selected = all_pdfs[:max_docs]
    print(f"[Parser] Found {len(all_pdfs)} tariff PDFs, using first {len(selected)} (sorted by name)")
    return selected
