"""
Claude Vision-based parser for LDS missionary health evaluation forms.

Uses Anthropic's Claude Vision API to extract structured form
data from PDF page images.

Falls back gracefully: returns None forms if API is unavailable or errors.
Caller (form_parser.py) then uses regex parsing instead.
"""

import base64
import io
import json
import re
import time
from pathlib import Path
from typing import Optional

from loguru import logger
from PIL import Image

from .models import DentalForm, MedicalForm

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    anthropic = None
    ANTHROPIC_AVAILABLE = False

# Reuse prompts, image helpers, JSON extractor, and model builders from vision_shared
from .vision_shared import (
    DENTAL_PROMPT,
    MEDICAL_CHECKBOX_PROMPT,
    _build_dental_form,
    _build_medical_form,
    _extract_json,
    _image_to_base64,
    render_pages,
)
from .pdf_processor import _resize_for_ocr


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "claude-3-5-sonnet-20241022"
# Also available: "claude-3-opus-20240229", "claude-3-haiku-20240307"
# ---------------------------------------------------------------------------
# JSON Schemas (for reference - Claude uses prompt-based constraints)
# ---------------------------------------------------------------------------

_SN = {"type": ["STRING", "NULL"]}   # nullable string shorthand

MEDICAL_CHECKBOX_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "functional_level": _SN,
        "chronic_condition": _SN,
        "driving_medication": _SN,
        "system_exam": {
            "type": "OBJECT",
            "properties": {k: _SN for k in [
                "general", "skin", "eyes", "ears", "nose_throat",
                "chest", "heart", "abdomen", "genitalia", "back",
                "extremities", "neurological",
            ]},
            "required": [
                "general", "skin", "eyes", "ears", "nose_throat",
                "chest", "heart", "abdomen", "genitalia", "back",
                "extremities", "neurological",
            ],
            "additionalProperties": False,
        },
        "labs": {
            "type": "OBJECT",
            "properties": {
                "tb_exposure_risk": _SN,
                "igra_result": _SN,
                "chest_xray": _SN,
            },
            "required": ["tb_exposure_risk", "igra_result", "chest_xray"],
            "additionalProperties": False,
        },
    },
    "required": [
        "functional_level", "chronic_condition", "driving_medication",
        "system_exam", "labs",
    ],
    "additionalProperties": False,
}

DENTAL_SCHEMA = {
    "type": "OBJECT",
    "properties": {k: _SN for k in [
        "orthodontic_treatment", "wisdom_teeth_removed",
        "brushing_frequency", "flossing_frequency",
        "pain_or_bleeding", "bleeding_comments",
        "dentist_name", "dentist_address", "dentist_city",
        "dentist_state", "dentist_postal", "dentist_phone",
        "dentist_email", "exam_date",
        "oral_exam_with_bitewing", "third_molars_assessed",
        "panoramic_done", "active_orthodontic", "dental_decay",
        "bonded_retainers", "oral_hygiene_acceptable",
        "dental_comments",
    ]},
    "required": [
        "orthodontic_treatment", "wisdom_teeth_removed",
        "brushing_frequency", "flossing_frequency",
        "pain_or_bleeding", "bleeding_comments",
        "dentist_name", "dentist_address", "dentist_city",
        "dentist_state", "dentist_postal", "dentist_phone",
        "dentist_email", "exam_date",
        "oral_exam_with_bitewing", "third_molars_assessed",
        "panoramic_done", "active_orthodontic", "dental_decay",
        "bonded_retainers", "oral_hygiene_acceptable",
        "dental_comments",
    ],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# Example loading (few-shot)
# ---------------------------------------------------------------------------

_EXAMPLES_CACHE = None

def _load_examples():
    """Load few-shot examples from logs/examples/manifest.json"""
    global _EXAMPLES_CACHE
    if _EXAMPLES_CACHE is not None:
        return _EXAMPLES_CACHE

    try:
        from .config import LOGS_DIR
        manifest_path = LOGS_DIR / "examples" / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                _EXAMPLES_CACHE = json.load(f)
            logger.info(f"Loaded {len(_EXAMPLES_CACHE)} few-shot examples for Claude")
        else:
            _EXAMPLES_CACHE = []
    except Exception as e:
        logger.warning(f"Could not load examples for Claude: {e}")
        _EXAMPLES_CACHE = []
    return _EXAMPLES_CACHE


def _build_few_shot_content(examples, form_type: str):
    """Build few-shot examples for the prompt."""
    if not examples:
        return []

    content = []
    for ex in examples:
        if form_type == "medical" and ex["key"].startswith("med_"):
            img_path = Path(__file__).parent.parent / "logs" / "examples" / ex["image"]
            if img_path.exists():
                img = Image.open(img_path)
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": _image_to_base64(img)
                    }
                })
                content.append({
                    "type": "text",
                    "text": json.dumps(ex["correct_values"])
                })
    return content

# ---------------------------------------------------------------------------
# Core Extraction Functions
# ---------------------------------------------------------------------------

def _configure_claude(api_key: str):
    """Configure the Anthropic API."""
    if not ANTHROPIC_AVAILABLE:
        raise RuntimeError("anthropic not installed. Run: pip install anthropic")
    return anthropic.Anthropic(api_key=api_key)


def _images_to_claude_parts(images: list[Image.Image], max_dimension: int = 1600) -> list[dict]:
    """Convert PIL images to Claude message parts format."""
    parts = []
    for img in images:
        if max(img.size) > max_dimension:
            ratio = max_dimension / max(img.size)
            new_size = (int(img.width * ratio), int(img.height * ratio))
            img = img.resize(new_size, Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        parts.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": base64.b64encode(buf.getvalue()).decode()
            }
        })
    return parts


def _run_claude_extraction(
    images: list[Image.Image],
    prompt: str,
    model_name: str,
    form_type: str,
    api_key: str,
    examples=None
) -> Optional[dict]:
    """Run Claude extraction with prompt-based constraints."""
    client = _configure_claude(api_key)

    # Build message content
    content = [{"type": "text", "text": prompt}]
    content.extend(_images_to_claude_parts(images))

    # Add few-shot examples if available
    if examples:
        few_shot = _build_few_shot_content(examples, form_type)
        if few_shot:
            # Insert examples before the actual images
            content = [{"type": "text", "text": prompt}] + few_shot + content[1:]

    try:
        response = client.messages.create(
            model=model_name,
            max_tokens=8192,
            temperature=0.0,
            messages=[{"role": "user", "content": content}]
        )
        if response.content and response.content[0].text:
            return _extract_json(response.content[0].text)
    except Exception as e:
        logger.error(f"Claude {form_type} extraction failed: {e}")
        raise

    return None


def _run_claude_batched(
    images: list[Image.Image],
    prompt: str,
    model_name: str,
    form_type: str,
    api_key: str,
    examples=None
) -> Optional[dict]:
    """Try batched multi-image call first."""
    return _run_claude_extraction(images, prompt, model_name, form_type, api_key, examples)


def _run_claude_page_by_page(
    images: list[Image.Image],
    prompt: str,
    model_name: str,
    form_type: str,
    api_key: str,
    examples=None
) -> Optional[dict]:
    """Fallback: process pages one at a time and merge results."""
    logger.info(f"  Falling back to page-by-page extraction ({len(images)} pages)...")
    merged = {}

    for i, img in enumerate(images):
        try:
            result = _run_claude_extraction([img], prompt, model_name, form_type, api_key)
            if result:
                # Deep merge
                for key, value in result.items():
                    if key not in merged or merged[key] is None:
                        merged[key] = value
                    elif isinstance(value, dict) and isinstance(merged[key], dict):
                        for k, v in value.items():
                            if merged[key].get(k) is None:
                                merged[key][k] = v
                    elif isinstance(value, list) and isinstance(merged[key], list):
                        # For lists like covid_doses, extend
                        merged[key].extend(v for v in value if v not in merged[key])
        except Exception as e:
            logger.warning(f"  Page {i+1} failed: {e}")
            continue

    return merged if merged else None

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_with_claude(
    pages: list[dict],
    has_medical: bool,
    has_dental: bool,
    api_key: str,
    model: str = DEFAULT_MODEL,
) -> tuple[Optional[MedicalForm], Optional[DentalForm], list[str]]:
    """
    Use Claude Vision to extract structured form data from rendered PDF page images.

    Pages are classified by content (medical vs dental sections) so that
    each Claude pass only receives the relevant subset of pages.

    Returns (medical_form, dental_form, errors).
    If both forms are None, caller should fall back to regex parsing.
    """
    if not ANTHROPIC_AVAILABLE:
        return None, None, ["anthropic package not installed — run: pip install anthropic"]

    if not any(p.get("image") for p in pages):
        return None, None, ["No page images found — cannot use Claude Vision"]

    if not api_key:
        return None, None, ["Claude API key not configured"]

    from .field_extractor import medical_page_indices, dental_page_indices

    errors: list[str] = []
    medical_form: Optional[MedicalForm] = None
    dental_form: Optional[DentalForm] = None

    examples = _load_examples()

    if has_medical:
        med_idx = medical_page_indices(pages)
        med_pages = [pages[i] for i in med_idx]
        images = [p["image"] for p in med_pages]
        logger.info(f"Claude Vision (medical): {len(images)} pages (classified from {len(pages)} total), model={model}")

        try:
            logger.info("Claude: extracting medical form...")
            # Try checkbox-focused extraction first (more reliable for checkboxes)
            data = _run_claude_batched(images, MEDICAL_CHECKBOX_PROMPT, model, "medical", api_key, examples)
            if data is None:
                # Fallback to page-by-page
                data = _run_claude_page_by_page(images, MEDICAL_CHECKBOX_PROMPT, model, "medical", api_key, examples)

            medical_form = _build_medical_form(data)
            logger.info("Claude medical extraction succeeded")
        except Exception as e:
            logger.error(f"Claude medical extraction failed: {e}")
            errors.append(f"Claude medical error: {e}")

    if has_dental:
        dnt_idx = dental_page_indices(pages)
        dnt_pages = [pages[i] for i in dnt_idx]
        images = [p["image"] for p in dnt_pages]
        logger.info(f"Claude Vision (dental): {len(images)} pages (classified from {len(pages)} total), model={model}")

        try:
            logger.info("Claude: extracting dental form...")
            data = _run_claude_batched(images, DENTAL_PROMPT, model, "dental", api_key, examples)
            if data is None:
                data = _run_claude_page_by_page(images, DENTAL_PROMPT, model, "dental", api_key, examples)

            dental_form = _build_dental_form(data)
            logger.info("Claude dental extraction succeeded")
        except Exception as e:
            logger.error(f"Claude dental extraction failed: {e}")
            errors.append(f"Claude dental error: {e}")

    return medical_form, dental_form, errors
