"""
image_extractor.py

Extracts images from PDF pages and generates captions using Groq vision.
Each extracted image is:
  1. Saved to disk at data/images/{doc_name}_p{page_no}_img{idx}.png
  2. Captioned via Groq vision model
  3. Returned as a record for embedding into Qdrant image collection

Why caption-based?
  We embed the caption text (not the image pixels) into Qdrant.
  At query time, we search captions semantically — if the query matches
  a caption, we load the image file, encode it as base64, and return it.
"""

import os
import base64
import time
from pathlib import Path
from typing import List, Dict, Any

import pdfplumber
from groq import Groq

from config import GROQ_API_KEY, GROQ_VISION_MODEL, IMAGES_DIR


def _encode_image_to_base64(image_path: str) -> str:
    """Read image file and return base64 data URI string."""
    with open(image_path, "rb") as f:
        raw = f.read()
    b64 = base64.b64encode(raw).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _caption_image(image_path: str, doc_name: str, page_no: int) -> str:
    """
    Send image to Groq vision model and get a technical caption.
    Returns caption string, or a fallback if vision call fails.
    """
    client = Groq(api_key=GROQ_API_KEY)

    try:
        b64 = _encode_image_to_base64(image_path)

        response = client.chat.completions.create(
            model=GROQ_VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": b64},
                        },
                        {
                            "type": "text",
                            "text": (
                                "This image is from a PG&E electrical engineering document "
                                f"({doc_name}, page {page_no}). "
                                "Describe what this image shows in 1-3 sentences. "
                                "Focus on: what type of diagram/figure it is, "
                                "what electrical components or standards it depicts, "
                                "and what a reader would use it for. "
                                "Be specific and technical."
                            ),
                        },
                    ],
                }
            ],
            max_tokens=200,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"[ImageExtractor] Vision caption failed for {image_path}: {e}")
        return f"Technical diagram from {doc_name} page {page_no}"


def extract_images_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract all images from a PDF and caption them.

    Returns list of image records:
    {
        "image_path":  str,   # absolute path to saved PNG
        "caption":     str,   # Groq-generated caption
        "page_no":     int,
        "doc_name":    str,
        "image_id":    str,   # e.g. "greenbook-manual-full_p45_img1"
    }
    """
    os.makedirs(IMAGES_DIR, exist_ok=True)

    doc_name = Path(pdf_path).stem
    records = []

    print(f"[ImageExtractor] Processing {doc_name}...")

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_no = i + 1

            # pdfplumber exposes images as dicts with pixel data
            page_images = page.images
            if not page_images:
                continue

            for img_idx, img in enumerate(page_images, 1):
                try:
                    # Crop the image region from the page and save as PNG
                    # pdfplumber image dict has x0, top, x1, bottom bbox
                    bbox = (img["x0"], img["top"], img["x1"], img["bottom"])

                    # Skip tiny images (likely decorative icons, bullets)
                    width = img["x1"] - img["x0"]
                    height = img["bottom"] - img["top"]
                    if width < 50 or height < 50:
                        continue

                    cropped = page.crop(bbox)
                    pil_image = cropped.to_image(resolution=150)

                    image_id = f"{doc_name}_p{page_no}_img{img_idx}"
                    image_path = os.path.join(IMAGES_DIR, f"{image_id}.png")
                    pil_image.save(image_path)

                    print(f"[ImageExtractor] Captioning {image_id}...")
                    caption = _caption_image(image_path, doc_name, page_no)

                    records.append({
                        "image_path": image_path,
                        "caption": caption,
                        "page_no": page_no,
                        "doc_name": doc_name,
                        "image_id": image_id,
                    })

                    # Small delay to avoid Groq rate limits
                    time.sleep(0.5)

                except Exception as e:
                    print(f"[ImageExtractor] Failed on {doc_name} p{page_no} img{img_idx}: {e}")
                    continue

    print(f"[ImageExtractor] {doc_name}: {len(records)} images extracted and captioned")
    return records


def extract_images_from_all_pdfs(pdf_paths: List[str]) -> List[Dict[str, Any]]:
    """Extract images from multiple PDFs."""
    all_records = []
    for path in pdf_paths:
        all_records.extend(extract_images_from_pdf(path))
    return all_records
