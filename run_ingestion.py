"""
run_ingestion.py  (project root)
Run this ONCE to ingest all documents into Qdrant.

Usage:
    python run_ingestion.py                # skips if already ingested
    python run_ingestion.py --force        # re-ingests text + images from scratch
    python run_ingestion.py --force-text   # re-ingests text only
    python run_ingestion.py --force-images # re-ingests images only
"""

import sys
from ingestion.pdf_parser import parse_all_pdfs, get_tariff_pdfs
from ingestion.chunker import chunk_pages
from ingestion.embedder import embed_and_store, embed_and_store_images
from ingestion.image_extractor import extract_images_from_all_pdfs
from config import GREENBOOK_PDF, TARIFF_DIR, MAX_TARIFF_DOCS


def run(force_text: bool = False, force_images: bool = False):
    print("=" * 60)
    print("SP&D RAG Ingestion Pipeline")
    print("=" * 60)

    # Collect PDF paths
    print("\n[Step 1] Collecting documents...")
    tariff_pdfs = get_tariff_pdfs(TARIFF_DIR, MAX_TARIFF_DOCS)
    all_pdfs = [GREENBOOK_PDF] + tariff_pdfs
    print(f"         Greenbook: 1 file")
    print(f"         Tariffs:   {len(tariff_pdfs)} files")
    print(f"         Total:     {len(all_pdfs)} documents")

    # Parse PDFs (text)
    print("\n[Step 2] Parsing PDFs for text...")
    pages = parse_all_pdfs(all_pdfs)
    print(f"         Total pages extracted: {len(pages)}")

    # Chunk text
    print("\n[Step 3] Chunking pages...")
    chunks = chunk_pages(pages)
    print(f"         Total chunks: {len(chunks)}")

    # Embed + store text chunks
    print("\n[Step 4] Embedding text chunks into Qdrant...")
    embed_and_store(chunks, force_reingest=force_text)

    # Extract images + captions
    print("\n[Step 5] Extracting images from PDFs and generating captions...")
    print("         Note: This makes Groq API calls for each image.")
    print("         Run with --force-images only when needed.")
    image_records = extract_images_from_all_pdfs(all_pdfs)
    print(f"         Total images extracted: {len(image_records)}")

    # Embed + store image captions
    print("\n[Step 6] Embedding image captions into Qdrant...")
    embed_and_store_images(image_records, force_reingest=force_images)

    print("\n" + "=" * 60)
    print("Ingestion complete. Start the API server with:")
    print("   uvicorn api.main:app --reload")
    print("=" * 60)


if __name__ == "__main__":
    args = sys.argv[1:]
    force_all = "--force" in args
    force_text = "--force-text" in args or force_all
    force_images = "--force-images" in args or force_all
    run(force_text=force_text, force_images=force_images)
