"""\
Gemini Vision-based parser for LDS missionary health evaluation forms.

Uses Google's Gemini 1.5 Flash via the Generative AI API to extract structured form
data from PDF page images. Free tier: 1,500 requests/day.

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
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    genai = None
    GEMINI_AVAILABLE = False

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

DEFAULT_MODEL = "gemini-1.5-flash"
# Also available: "gemini-1.5-flash-8b" (smaller, faster), "gemini-1.5-pro" (higher quality)


# ---------------------------------------------------------------------------
# JSON Schemas for constrained generation (via response_mime_type + schema)
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
FULL_MEDICAL_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "physician_info": {
            "type": "OBJECT",
            "properties": {k: _SN for k in [
                "name", "address", "city", "state", "postal_code", "phone", "email"
            ]},
            "required": ["name", "address", "city", "state", "postal_code", "phone", "email"],
            "additionalProperties": False,
        },
        "functional_level": _SN,
        "chronic_condition": _SN,
        "chronic_comments": _SN,
        "driving_medication": _SN,
        "driving_medication_explain": _SN,
        "exam_date": _SN,
        "vital_signs": {
            "type": "OBJECT",
            "properties": {k: _SN for k in [
                "height", "weight", "blood_pressure", "pulse", "vision_right", "vision_left"
            ]},
            "required": ["height", "weight", "blood_pressure", "pulse", "vision_right", "vision_left"],
            "additionalProperties": False,
        },
        "system_exam": {
            "type": "OBJECT",
            "properties": {k: _SN for k in [
                "general", "skin", "eyes", "ears", "nose_throat", "chest", "heart",
                "abdomen", "genitalia", "back", "extremities", "neurological",
            ]},
            "required": [
                "general", "skin", "eyes", "ears", "nose_throat", "chest", "heart",
                "abdomen", "genitalia", "back", "extremities", "neurological",
            ],
            "additionalProperties": False,
        },
        "labs": {
            "type": "OBJECT",
            "properties": {k: _SN for k in [
                "urinalysis_blood", "urinalysis_protein", "urinalysis_sugar",
                "hemoglobin", "tb_exposure_risk", "ppd_result",
                "igra_result", "chest_xray",
            ]},
            "required": [
                "urinalysis_blood", "urinalysis_protein", "urinalysis_sugar",
                "hemoglobin", "tb_exposure_risk", "ppd_result",
                "igra_result", "chest_xray",
            ],
            "additionalProperties": False,
        },
        "immunizations": {
            "type": "OBJECT",
            "properties": {k: _SN for k in [
                "tdap", "tdap2", "mmr1", "mmr2", "polio",
                "hep_a1", "hep_a2", "hep_b1", "hep_b2", "hep_b3",
                "influenza", "covid_doses", "covid_brand",
            ]},
            "required": [
                "tdap", "tdap2", "mmr1", "mmr2", "polio",
                "hep_a1", "hep_a2", "hep_b1", "hep_b2", "hep_b3",
                "influenza", "covid_doses", "covid_brand",
            ],
            "additionalProperties": False,
        },
    },
    "required": [
        "physician_info", "functional_level", "chronic_condition",
        "chronic_comments", "driving_medication", "driving_medication_explain",
        "exam_date", "vital_signs", "system_exam", "labs", "immunizations",
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
            logger.info(f"Loaded {len(_EXAMPLES_CACHE)} few-shot examples for Gemini")
        else:
            _EXAMPLES_CACHE = []
    except Exception as e:
        logger.warning(f"Could not load examples for Gemini: {e}")
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
                content.append(_image_to_base64(img))
                content.append(json.dumps(ex["correct_values"]))
    return content


# ---------------------------------------------------------------------------
# Core Extraction Functions
# ---------------------------------------------------------------------------

def _configure_gemini(api_key: str):
    """Configure the Gemini API."""
    if not GEMINI_AVAILABLE:
        raise RuntimeError("google-generativeai not installed. Run: pip install google-generativeai")
    genai.configure(api_key=api_key)


def _images_to_parts(images: list[Image.Image], max_dimension: int = 1600) -> list[dict]:
    """Convert PIL images to Gemini parts format."""
    parts = []
    for img in images:
        if max(img.size) > max_dimension:
            ratio = max_dimension / max(img.size)
            new_size = (int(img.width * ratio), int(img.height * ratio))
            img = img.resize(new_size, Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        parts.append({
            "mime_type": "image/png",
            "data": base64.b64encode(buf.getvalue()).decode()
        })
    return parts


def _run_gemini_extraction(
    images: list[Image.Image],
    prompt: str,
    schema: dict,
    model_name: str,
    form_type: str,
    api_key: str,
    examples=None
) -> Optional[dict]:
    """Run Gemini extraction with schema-constrained output."""
    _configure_gemini(api_key)

    model = genai.GenerativeModel(
        model_name=model_name,
        generation_config={
            "response_mime_type": "application/json",
            "response_schema": schema,
            "temperature": 0.0,
            "top_p": 0.95,
            "max_output_tokens": 8192,
        }
    )

    # Build content: prompt + images
    content = [prompt]
    content.extend(_images_to_parts(images))

    # Add few-shot examples if available
    if examples:
        few_shot = _build_few_shot_content(examples, form_type)
        if few_shot:
            # Insert examples before the actual images
            content = [prompt] + few_shot + content[1:]

    try:
        response = model.generate_content(content)
        if response.text:
            return _extract_json(response.text)
    except Exception as e:
        logger.error(f"Gemini {form_type} extraction failed: {e}")
        raise

    return None


def _run_gemini_batched(
    images: list[Image.Image],
    prompt: str,
    schema: dict,
    model_name: str,
    form_type: str,
    api_key: str,
    examples=None
) -> Optional[dict]:
    """Try batched multi-image call first."""
    return _run_gemini_extraction(images, prompt, schema, model_name, form_type, api_key, examples)


def _run_gemini_page_by_page(
    images: list[Image.Image],
    prompt: str,
    schema: dict,
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
            result = _run_gemini_extraction([img], prompt, schema, model_name, form_type, api_key)
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

def parse_with_gemini(
    pages: list[dict],
    has_medical: bool,
    has_dental: bool,
    api_key: str,
    model: str = DEFAULT_MODEL,
) -> tuple[Optional[MedicalForm], Optional[DentalForm], list[str]]:
    """
    Use Gemini Vision to extract structured form data from rendered PDF page images.

    Pages are classified by content (medical vs dental sections) so that
    each Gemini pass only receives the relevant subset of pages.

    Returns (medical_form, dental_form, errors).
    If both forms are None, caller should fall back to regex parsing.
    """
    if not GEMINI_AVAILABLE:
        return None, None, ["google-generativeai package not installed — run: pip install google-generativeai"]

    if not any(p.get("image") for p in pages):
        return None, None, ["No page images found — cannot use Gemini Vision"]

    if not api_key:
        return None, None, ["Gemini API key not configured"]

    from .field_extractor import medical_page_indices, dental_page_indices

    errors: list[str] = []
    medical_form: Optional[MedicalForm] = None
    dental_form: Optional[DentalForm] = None

    examples = _load_examples()

    if has_medical:
        med_idx = medical_page_indices(pages)
        med_pages = [pages[i] for i in med_idx]
        images = [p["image"] for p in med_pages]
        logger.info(f"Gemini Vision (medical): {len(images)} pages (classified from {len(pages)} total), model={model}")

        try:
            logger.info("Gemini: extracting medical form...")
            # Try checkbox-focused extraction first (more reliable for checkboxes)
            data = _run_gemini_batched(images, MEDICAL_CHECKBOX_PROMPT, MEDICAL_CHECKBOX_SCHEMA,
                                        model, "medical", api_key, examples)
            if data is None:
                # Fallback to page-by-page
                data = _run_gemini_page_by_page(images, MEDICAL_CHECKBOX_PROMPT, MEDICAL_CHECKBOX_SCHEMA,
                                                 model, "medical", api_key, examples)

            medical_form = _build_medical_form(data)
            logger.info("Gemini medical extraction succeeded")
        except Exception as e:
            logger.error(f"Gemini medical extraction failed: {e}")
            errors.append(f"Gemini medical error: {e}")

    if has_dental:
        dnt_idx = dental_page_indices(pages)
        dnt_pages = [pages[i] for i in dnt_idx]
        images = [p["image"] for p in dnt_pages]
        logger.info(f"Gemini Vision (dental): {len(images)} pages (classified from {len(pages)} total), model={model}")

        try:
            logger.info("Gemini: extracting dental form...")
            data = _run_gemini_batched(images, DENTAL_PROMPT, DENTAL_SCHEMA,
                                        model, "dental", api_key, examples)
            if data is None:
                data = _run_gemini_page_by_page(images, DENTAL_PROMPT, DENTAL_SCHEMA,
                                                 model, "dental", api_key, examples)

            dental_form = _build_dental_form(data)
            logger.info("Gemini dental extraction succeeded")
        except Exception as e:
            logger.error(f"Gemini dental extraction failed: {e}")
            errors.append(f"Gemini dental error: {e}")

    return medical_form, dental_form, errors
