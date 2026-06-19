"""
image_extractor.py
Extracts images from PDFs and associates them with captions from nearby text.
 
Strategy:
1. For each page, find all images via pdfplumber
2. Look at text above/below the image bbox — that's the caption
3. Crop image to bbox, save as PNG, encode as base64
4. Store (image_path, caption, page_no, doc_name, image_id) in image_records
5. These records are later embedded and stored in Qdrant for caption-based search
 
Caption extraction uses PDF text positioning only — NO Groq vision calls.
"""
 
import os
from pathlib import Path
from typing import List, Dict, Any
import base64
 
import pdfplumber
from PIL import Image
import io
 
 
def _extract_nearby_text(page, image_bbox: tuple, max_distance: float = 100) -> str:
    """
    Find text lines near the image bbox and return them as a caption.
 
    Strategy:
    1. Get all text on the page with character-level bboxes
    2. Filter for text that's above the image (between image.y0 - max_distance and image.y0)
       or below the image (between image.y1 and image.y1 + max_distance)
    3. Sort by y position and join into caption
    """
    image_y0, image_y1 = image_bbox[1], image_bbox[3]  # top, bottom
    image_x0, image_x1 = image_bbox[0], image_bbox[2]  # left, right
 
    try:
        chars = page.chars
    except Exception:
        return None
 
    # Find text lines near the image
    nearby_lines = []
 
    for char in chars:
        char_y0, char_y1 = char["top"], char["bottom"]
        char_x0, char_x1 = char["x0"], char["x1"]
 
        # Above the image
        if image_y0 - max_distance <= char_y1 <= image_y0:
            nearby_lines.append((char_y0, char_x0, char["text"]))
        # Below the image
        elif image_y1 <= char_y0 <= image_y1 + max_distance:
            nearby_lines.append((char_y0, char_x0, char["text"]))
 
    if not nearby_lines:
        return None
 
    # Sort by y position, then x position (left to right)
    nearby_lines.sort(key=lambda x: (x[0], x[1]))
 
    # Join characters into a caption (simple concatenation)
    caption = "".join([item[2] for item in nearby_lines]).strip()
 
    # Clean up — remove excessive whitespace
    caption = " ".join(caption.split())
 
    return caption if len(caption) > 5 else None  # Only use if non-trivial
 
 
def _crop_and_save_image(page, image_bbox: tuple, doc_name: str, page_no: int, img_index: int, images_dir: str) -> tuple[str, str]:
    """
    Crops an image from the page PDF and saves it as PNG, encodes to base64.
    Returns (image_path, image_base64).
    """
    try:
        bbox = (image_bbox[0], image_bbox[1], image_bbox[2], image_bbox[3])  # (x0, top, x1, bottom)
        cropped = page.crop(bbox)
        pil_image = cropped.to_image(resolution=150).original
 
        filename = f"{doc_name}_p{page_no}_img{img_index}.png"
        filepath = os.path.join(images_dir, filename)
        pil_image.save(filepath, format="PNG")
 
        # Also encode to base64 for JSON response
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
 
        return f"data/images/{filename}", f"data:image/png;base64,{img_base64}"
 
    except Exception as e:
        print(f"[ImageExtractor] Could not extract image {img_index} from page {page_no}: {e}")
        return None, None
 
 
def extract_images_from_pdf(pdf_path: str, images_dir: str) -> List[Dict[str, Any]]:
    """
    Extract images from a single PDF using nearby text as captions.
 
    Returns list of image records:
    {
        "image_path":      str,      # data/images/greenbook_p45_img1.png
        "image_base64":    str,      # data:image/png;base64,... (full data URI)
        "caption":         str,      # extracted from text near image
        "page_no":         int,
        "doc_name":        str,
        "image_id":        str,      # greenbook_p45_img1
    }
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
 
    os.makedirs(images_dir, exist_ok=True)
    doc_name = Path(pdf_path).stem
    image_records = []
 
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        print(f"[ImageExtractor] {doc_name}: scanning {total_pages} pages for images...")
 
        for page_idx, page in enumerate(pdf.pages):
            page_no = page_idx + 1
 
            try:
                page_images = page.images  # list of image dicts from pdfplumber
            except Exception:
                continue
 
            for img_index, img in enumerate(page_images, 1):
                # Extract caption from nearby text (no Groq calls)
                bbox = (img["x0"], img["top"], img["x1"], img["bottom"])
                caption = _extract_nearby_text(page, bbox)
 
                if not caption:
                    # Fallback: use a generic caption
                    caption = f"Figure on page {page_no}"
 
                # Crop and save image
                image_path, image_base64 = _crop_and_save_image(
                    page, bbox, doc_name, page_no, img_index, images_dir
                )
 
                if not image_path:
                    continue
 
                image_id = f"{doc_name}_p{page_no}_img{img_index}"
 
                image_records.append({
                    "image_path": image_path,
                    "image_base64": image_base64,
                    "caption": caption,
                    "page_no": page_no,
                    "doc_name": doc_name,
                    "image_id": image_id,
                })
 
                print(f"[ImageExtractor] {doc_name} page {page_no}: extracted image {img_index}")
                print(f"                Caption: {caption[:80]}...")
 
    print(f"[ImageExtractor] {doc_name}: {len(image_records)} images extracted")
    return image_records
 
 
def extract_images_from_all_pdfs(pdf_paths: List[str], images_dir: str = "data/images") -> List[Dict[str, Any]]:
    """
    Extract images from multiple PDFs using nearby text captions.
    """
    all_records = []
    for pdf_path in pdf_paths:
        try:
            records = extract_images_from_pdf(pdf_path, images_dir)
            all_records.extend(records)
        except Exception as e:
            print(f"[ImageExtractor] Error processing {pdf_path}: {e}")
            continue
    return all_records