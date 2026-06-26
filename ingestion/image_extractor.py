"""
image_extractor.py
Extracts images from PDFs and associates them with captions from nearby text.
 
Strategy:
1. For each page, find all images via pdfplumber
2. Skip images in header/footer zones (spatial filtering)
3. Look at text above/below the image bbox — that's the caption
4. Crop image to bbox, save as PNG, encode as base64
5. Store (image_path, caption, page_no, doc_name, image_id) in image_records
6. These records are later embedded and stored in Qdrant for caption-based search
 
Caption extraction uses PDF text positioning only — NO Groq vision calls.
Header/footer filtering uses page dimensions — headers are in top ~10%, footers in bottom ~10%.
"""
 
import os
from pathlib import Path
from typing import List, Dict, Any
import base64
 
import pdfplumber
from PIL import Image
import io

# ─────────────────────────────────────────────────────────────────────────────
# Header/Footer Zones
#
# Most PDF headers/footers occupy the top and bottom ~10% of the page.
# Images in these zones are typically logos, page numbers, company branding.
# We filter them out spatially rather than by keyword to avoid false positives.
# ─────────────────────────────────────────────────────────────────────────────

HEADER_ZONE_PERCENT = 10  # Top 10% of page is header
FOOTER_ZONE_PERCENT = 10  # Bottom 10% of page is footer


def _is_in_header_or_footer(image_bbox: tuple, page_height: float) -> bool:
    """
    Returns True if the image is in the header or footer zone.
    
    Args:
        image_bbox: (x0, top, x1, bottom) in PDF coordinates
        page_height: height of the page in points
    
    Returns:
        True if image center is in top HEADER_ZONE_PERCENT or bottom FOOTER_ZONE_PERCENT of page
    """
    _, top, _, bottom = image_bbox
    image_center_y = (top + bottom) / 2
    
    header_threshold = page_height * (HEADER_ZONE_PERCENT / 100)
    footer_threshold = page_height * (1 - FOOTER_ZONE_PERCENT / 100)
    
    in_header = image_center_y < header_threshold
    in_footer = image_center_y > footer_threshold
    
    return in_header or in_footer
 
 
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
            page_height = page.height  # PDF coordinate space height
 
            try:
                page_images = page.images  # list of image dicts from pdfplumber
            except Exception:
                continue
 
            for img_index, img in enumerate(page_images, 1):
                bbox = (img["x0"], img["top"], img["x1"], img["bottom"])
                
                # Skip images in header/footer zones (spatial filtering)
                if _is_in_header_or_footer(bbox, page_height):
                    print(f"[ImageExtractor] Skipping image in header/footer on page {page_no}")
                    continue
 
                # Extract caption from nearby text (no Groq calls)
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