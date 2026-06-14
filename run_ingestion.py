"""
run_ingestion.py  (project root)
Run this ONCE to ingest all documents into Qdrant.

Usage:
    python run_ingestion.py              # skips if already ingested
    python run_ingestion.py --force      # re-ingests from scratch
"""

import sys
from ingestion.pdf_parser import parse_all_pdfs, get_tariff_pdfs
from ingestion.chunker import chunk_pages
from ingestion.embedder import embed_and_store
from config import GREENBOOK_PDF, TARIFF_DIR, MAX_TARIFF_DOCS


def run(force: bool = False):
    print("=" * 60)
    print("SP&D RAG Ingestion Pipeline")
    print("=" * 60)

    # Step 1: Collect PDF paths
    print("\n[Step 1] Collecting documents...")
    tariff_pdfs = get_tariff_pdfs(TARIFF_DIR, MAX_TARIFF_DOCS)
    all_pdfs = [GREENBOOK_PDF] + tariff_pdfs
    print(f"         Greenbook: 1 file")
    print(f"         Tariffs:   {len(tariff_pdfs)} files")
    print(f"         Total:     {len(all_pdfs)} documents")

    # Step 2: Parse PDFs
    print("\n[Step 2] Parsing PDFs...")
    pages = parse_all_pdfs(all_pdfs)
    print(f"         Total pages extracted: {len(pages)}")

    # Step 3: Chunk
    print("\n[Step 3] Chunking pages...")
    chunks = chunk_pages(pages)
    print(f"         Total chunks: {len(chunks)}")

    # Step 4: Embed + store in Qdrant
    print("\n[Step 4] Embedding and storing in Qdrant...")
    embed_and_store(chunks, force_reingest=force)

    print("\nIngestion complete. Start the API server with:")
    print("   uvicorn api.main:app --reload")


if __name__ == "__main__":
    force_flag = "--force" in sys.argv
    run(force=force_flag)
