"""
PDF text extraction pipeline (priority order):
  1. PyMuPDF    — renders PDF pages to images (replaces pdf2image; needs no poppler)
  2. Apple Vision framework — native macOS OCR, zero system dependencies, high accuracy
  3. Tesseract  — cross-platform OCR fallback (requires: brew install tesseract)
  4. Ollama     — local vision LLM fallback for low-confidence results
"""

import base64
import io
import re
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
import pdfplumber
from loguru import logger
from PIL import Image

from .config import load_config

# Disable PIL's decompression-bomb guard entirely.
# _resize_for_ocr() enforces our own pixel-count cap before any OCR call,
# so PIL's built-in check is redundant and only clutters the logs.
Image.MAX_IMAGE_PIXELS = None

# Maximum pixels per image fed to any OCR backend.
# Apple Vision hard-fails above ~178 MP; Tesseract slows badly above ~50 MP.
# 80 MP gives good OCR quality while staying well within all limits.
_MAX_OCR_PIXELS = 80_000_000


def _resize_for_ocr(img: Image.Image) -> Image.Image:
    """
    Downscale *img* so its total pixel count stays at or below _MAX_OCR_PIXELS.
    Returns the original image unchanged if it is already small enough.
    """
    w, h = img.size
    pixels = w * h
    if pixels <= _MAX_OCR_PIXELS:
        return img
    scale = (_MAX_OCR_PIXELS / pixels) ** 0.5
    new_w, new_h = int(w * scale), int(h * scale)
    logger.debug(f"  Resizing {w}×{h} ({pixels // 1_000_000} MP) → {new_w}×{new_h} for OCR")
    return img.resize((new_w, new_h), Image.LANCZOS)

# ---------------------------------------------------------------------------
# Optional imports – degrade gracefully if not installed
# ---------------------------------------------------------------------------

# Apple Vision framework (macOS native – most accurate, no external tools needed)
try:
    import Vision
    from Foundation import NSData
    APPLE_VISION_AVAILABLE = True
except ImportError:
    APPLE_VISION_AVAILABLE = False

# Tesseract (cross-platform fallback)
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

# Ollama local LLM (optional enhanced fallback)
try:
    import ollama as ollama_client
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _image_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _clean_text(text: str) -> str:
    """Normalize whitespace and remove control characters."""
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _render_pdf_pages(pdf_path: Path, dpi: int = 300) -> list[Image.Image]:
    """Render all pages of a PDF to PIL Images using PyMuPDF (no poppler needed)."""
    doc = fitz.open(str(pdf_path))
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    images = []
    for page in doc:
        pix = page.get_pixmap(matrix=mat)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        images.append(img)
    return images


# ---------------------------------------------------------------------------
# Primary extraction: pdfplumber (embedded text – fast path)
# ---------------------------------------------------------------------------

def extract_text_pdfplumber(pdf_path: Path) -> list[dict]:
    """Returns list of page dicts: {page, text, method}. Text may be empty for scanned PDFs.

    Uses layout=True so pdfplumber preserves column structure rather than
    interleaving left/right column text — this fixes physician-info and vitals
    fields that OCR reads in the wrong order on multi-column form pages.
    """
    pages = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text(layout=True) or ""
            text = _clean_text(text)
            pages.append({"page": i, "text": text, "method": "pdfplumber"})
    return pages


# ---------------------------------------------------------------------------
# OCR: Apple Vision framework (macOS native – best quality, no system deps)
# ---------------------------------------------------------------------------

def _run_apple_vision_request(png_bytes: bytes):
    """Run Vision text recognition and return the raw observations list."""
    request_handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(
        NSData.dataWithBytes_length_(png_bytes, len(png_bytes)),
        {}
    )
    request = Vision.VNRecognizeTextRequest.alloc().init()
    request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    request.setUsesLanguageCorrection_(True)
    request_handler.performRequests_error_([request], None)
    return request.results() or []


def extract_text_apple_vision(img: Image.Image) -> str:
    """OCR a PIL Image using macOS Vision framework. Returns extracted text."""
    if not APPLE_VISION_AVAILABLE:
        raise RuntimeError("Apple Vision framework not available")

    img = _resize_for_ocr(img)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    lines = []
    for obs in _run_apple_vision_request(png_bytes):
        candidates = obs.topCandidates_(1)
        if candidates and len(candidates) > 0:
            lines.append(candidates[0].string())

    return _clean_text("\n".join(lines))


def extract_ocr_observations(img: Image.Image) -> list[dict]:
    """OCR a PIL Image using Apple Vision. Returns list of observations with
    text, confidence, and bounding box in PIL pixel coordinates (origin top-left).

    Each entry: {text, confidence, x, y, w, h}
    Vision coordinate origin is bottom-left; we flip y so (0,0) is top-left.
    """
    if not APPLE_VISION_AVAILABLE:
        return []

    img = _resize_for_ocr(img)
    img_w, img_h = img.size
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    observations = []
    for obs in _run_apple_vision_request(png_bytes):
        candidates = obs.topCandidates_(1)
        if not (candidates and len(candidates) > 0):
            continue
        text = candidates[0].string()
        confidence = float(candidates[0].confidence())

        # boundingBox() → CGRect in normalized Vision coords (origin bottom-left)
        try:
            bbox = obs.boundingBox()
            vx = float(bbox.origin.x)
            vy = float(bbox.origin.y)
            vw = float(bbox.size.width)
            vh = float(bbox.size.height)
        except Exception:
            try:
                # Some pyobjc versions return ((x,y),(w,h)) tuple
                (vx, vy), (vw, vh) = obs.boundingBox()
                vx, vy, vw, vh = float(vx), float(vy), float(vw), float(vh)
            except Exception:
                # Fall back: no spatial info; store full-page bbox
                observations.append({"text": text, "confidence": confidence,
                                      "x": 0.0, "y": 0.0, "w": float(img_w), "h": float(img_h)})
                continue

        # Flip y: Vision (0,0)=bottom-left → PIL (0,0)=top-left
        x = vx * img_w
        y = (1.0 - vy - vh) * img_h
        w = vw * img_w
        h = vh * img_h
        observations.append({"text": text, "confidence": confidence,
                              "x": x, "y": y, "w": w, "h": h})
    return observations


def extract_text_vision_pdf(pdf_path: Path) -> list[dict]:
    """Render each PDF page with PyMuPDF and OCR with Apple Vision.
    Each page dict includes 'ocr_observations' with per-word bounding boxes."""
    images = _render_pdf_pages(pdf_path)
    pages = []
    for i, img in enumerate(images, start=1):
        try:
            obs_list = extract_ocr_observations(img)
            text = _clean_text("\n".join(o["text"] for o in obs_list))
            pages.append({"page": i, "text": text, "method": "apple_vision",
                          "image": img, "ocr_observations": obs_list})
        except Exception:
            text = extract_text_apple_vision(img)
            pages.append({"page": i, "text": text, "method": "apple_vision", "image": img})
    return pages


# ---------------------------------------------------------------------------
# OCR: Tesseract (cross-platform fallback)
# ---------------------------------------------------------------------------

def extract_text_tesseract_pdf(pdf_path: Path, lang: str = "eng") -> list[dict]:
    """Render each page with PyMuPDF and OCR with Tesseract."""
    if not TESSERACT_AVAILABLE:
        raise RuntimeError("pytesseract not installed")
    images = _render_pdf_pages(pdf_path)
    pages = []
    for i, img in enumerate(images, start=1):
        img = _resize_for_ocr(img)
        data = pytesseract.image_to_data(img, lang=lang, output_type=pytesseract.Output.DICT)
        words = [
            data["text"][j]
            for j in range(len(data["text"]))
            if int(data["conf"][j]) >= 30 and data["text"][j].strip()
        ]
        text = _clean_text(" ".join(words))
        pages.append({"page": i, "text": text, "method": "tesseract", "image": img})
    return pages


# ---------------------------------------------------------------------------
# Tertiary extraction: Ollama vision model
# ---------------------------------------------------------------------------

OLLAMA_MEDICAL_PROMPT = """You are extracting data from a missionary medical/dental evaluation form image.
Return ONLY the raw text you can read from this form page, preserving field labels and values.
Include all text: typed, handwritten, checkbox states (marked/unmarked), and field labels.
For checkboxes write: CHECKED or UNCHECKED before the label.
Do not summarize, interpret, or add any text that is not on the form."""


def extract_text_ollama(image: Image.Image, model: str = "llava") -> str:
    """Use a local Ollama vision model to extract text from an image."""
    if not OLLAMA_AVAILABLE:
        raise RuntimeError("ollama Python library not installed")

    image = _resize_for_ocr(image)
    img_b64 = _image_to_base64(image)
    response = ollama_client.generate(
        model=model,
        prompt=OLLAMA_MEDICAL_PROMPT,
        images=[img_b64],
    )
    return _clean_text(response["response"])


# ---------------------------------------------------------------------------
# Quality check
# ---------------------------------------------------------------------------

def _text_has_content(text: str, min_words: int = 20) -> bool:
    """Return True if the text appears to contain enough readable content."""
    words = [w for w in text.split() if re.match(r"[a-zA-Z]{2,}", w)]
    return len(words) >= min_words


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def extract_pdf_text(pdf_path: Path) -> list[dict]:
    """
    Full extraction pipeline for a single PDF.

    Priority:
      1. pdfplumber  (fast; works if PDF has selectable text)
      2. Apple Vision (macOS native OCR; no system deps; high accuracy)
      3. Tesseract   (cross-platform OCR; requires: brew install tesseract)
      4. Ollama      (local vision LLM; requires Ollama + a vision model)

    Returns a list of page dicts: {page, text, method, image(optional)}
    """
    config = load_config()
    pdf_path = Path(pdf_path)
    logger.info(f"Extracting text from: {pdf_path.name}")

    # Step 1: Try pdfplumber for embedded text (instant for digital PDFs)
    try:
        pages = extract_text_pdfplumber(pdf_path)
    except Exception as e:
        logger.warning(f"pdfplumber failed: {e}")
        pages = []

    needs_ocr = [p for p in pages if not _text_has_content(p["text"])]

    if not needs_ocr:
        logger.info(f"pdfplumber extracted text from all {len(pages)} pages")
        return pages

    logger.info(f"{len(needs_ocr)}/{len(pages)} pages need OCR")

    # Step 2: Apple Vision (macOS native – preferred, no external dependencies)
    if APPLE_VISION_AVAILABLE:
        logger.info("Using Apple Vision framework for OCR …")
        try:
            vision_results = extract_text_vision_pdf(pdf_path)
            # Replace sparse pages with Vision results
            vision_map = {r["page"]: r for r in vision_results}
            for p in pages:
                if not _text_has_content(p["text"]):
                    vr = vision_map.get(p["page"])
                    if vr and _text_has_content(vr["text"]):
                        p.update(vr)
            needs_ocr = [p for p in pages if not _text_has_content(p["text"])]
            if not needs_ocr:
                logger.info("Apple Vision OCR succeeded for all pages")
                return pages
        except Exception as e:
            logger.warning(f"Apple Vision OCR failed: {e}")

    # Step 3: Tesseract fallback
    if needs_ocr and TESSERACT_AVAILABLE:
        logger.info(f"Falling back to Tesseract OCR for {len(needs_ocr)} pages")
        try:
            tess_results = extract_text_tesseract_pdf(pdf_path, lang=config["tesseract_lang"])
            tess_map = {r["page"]: r for r in tess_results}
            for p in pages:
                if not _text_has_content(p["text"]):
                    tr = tess_map.get(p["page"])
                    if tr and _text_has_content(tr["text"]):
                        p.update(tr)
            needs_ocr = [p for p in pages if not _text_has_content(p["text"])]
        except Exception as e:
            logger.warning(f"Tesseract OCR failed: {e}")

    # Step 4: Ollama vision model for any remaining sparse pages
    if needs_ocr and config.get("ollama_enabled") and OLLAMA_AVAILABLE:
        model = config.get("ollama_model", "llava")
        logger.info(f"Trying Ollama ({model}) on {len(needs_ocr)} remaining pages")
        # Ensure we have rendered images for these pages
        try:
            all_images = _render_pdf_pages(pdf_path)
        except Exception:
            all_images = []
        for p in needs_ocr:
            idx = p["page"] - 1
            img = p.get("image") or (all_images[idx] if idx < len(all_images) else None)
            if img:
                try:
                    text = extract_text_ollama(img, model=model)
                    if _text_has_content(text):
                        p["text"] = text
                        p["method"] = "ollama"
                except Exception as e:
                    logger.warning(f"  Page {p['page']}: Ollama failed – {e}")

    for p in pages:
        logger.debug(f"  Page {p['page']} ({p['method']}): {len(p['text'].split())} words")

    return pages


def combine_pages(pages: list[dict]) -> str:
    """Combine all page texts with page separators."""
    parts = []
    for p in pages:
        parts.append(f"\n--- PAGE {p['page']} ---\n{p['text']}")
    return "\n".join(parts)


def detect_form_types(filename: str, full_text: str) -> tuple[bool, bool]:
    """
    Determine whether a PDF contains medical and/or dental forms.
    Returns (has_medical, has_dental).
    """
    name_upper = filename.upper()
    text_upper = full_text.upper()

    # Check filename keywords first
    file_medical = bool(re.search(r"MEDICAL", name_upper))
    file_dental = bool(re.search(r"DENTAL", name_upper))

    # Cross-check with text content if filename is ambiguous
    text_medical = bool(re.search(r"PHYSICIAN.{0,30}HEALTH EVALUATION|MEDICAL EVALUATION", text_upper))
    text_dental = bool(re.search(r"DENTAL EVALUATION|ORAL EXAMINATION|DENTIST", text_upper))

    has_medical = file_medical or text_medical
    has_dental = file_dental or text_dental

    # If filename has neither keyword, rely on text content
    if not file_medical and not file_dental:
        has_medical = text_medical
        has_dental = text_dental

    return has_medical, has_dental


def parse_missionary_name_from_filename(filename: str) -> str:
    """
    Extract missionary name from common filename patterns:
      "Tyler Fisher Dental Medical - Dave Miner.pdf"
      "Britta Nielsen-Medical - Dave Miner.pdf"
      "Kaden Hansen Missionary Medical Forms - Joshua G C.pdf"
      "2025.07.20 Missionary Health Forms For Ellie Easton - Scott Poelman.pdf"
      "2025.07.27 Missionary Health Forms For Brent And Shannon Robinson.pdf"
    """
    # Strip extension
    name = Path(filename).stem

    # Normalize underscores and multiple spaces to single space
    name = name.replace("_", " ")
    name = re.sub(r"\s+", " ", name).strip()

    # Strip leading date prefix like "2025.07.20 "
    name = re.sub(r"^\d{4}\.\d{2}\.\d{2}\s+", "", name)

    # Handle "... Health Forms For {Name}" or "... Forms For {Name}" pattern
    m = re.search(r"\bHealth Forms For (.+)", name, re.IGNORECASE)
    if not m:
        m = re.search(r"\bForms For (.+)", name, re.IGNORECASE)
    if m:
        name = m.group(1)

    # Remove everything after " - " (doctor name)
    name = re.split(r"\s+-\s+", name)[0]

    # Normalize inline hyphens used as word separators (e.g. "Nielsen-Dental")
    # to spaces so the keyword patterns below can match them.
    name = re.sub(r"(\w)-(\w)", r"\1 \2", name).strip()

    # Remove trailing type keywords; run multiple passes so compound phrases
    # like "Dental Evaluation" are stripped completely (first "Evaluation",
    # then "Dental" on the next pass).
    type_keywords = [
        r"\s+Missionary Medical Forms$",
        r"\s+Physician.{0,30}Health Evaluation$",
        r"\s+Dental Evaluation$",
        r"\s+Medical Evaluation$",
        r"\s+Personal Health History$",
        r"\s+Medical Dental$",
        r"\s+Dental Medical$",
        r"\s+Medical Forms?$",
        r"\s+Dental Forms?$",
        r"\s+Dental Form$",
        r"\s+Mission Docs?$",
        r"\s+Medical Docs?$",
        r"\s+Dental Docs?$",
        r"\s+Dental Doc$",
        r"\s+Medical Doc$",
        r"\s+Medical$",
        r"\s+Dental$",
        r"\s+Missionary$",
        r"\s+Docs?$",
        r"\s+Evaluation$",
    ]
    for _ in range(3):  # multi-pass so compound phrases are fully stripped
        for pattern in type_keywords:
            name = re.sub(pattern, "", name, flags=re.IGNORECASE).strip()

    return name.title()
