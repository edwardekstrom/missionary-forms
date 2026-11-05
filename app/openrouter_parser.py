"""
OpenRouter Vision-based parser for LDS missionary health evaluation forms.
"""

import base64
import io
import json
from pathlib import Path
from typing import Optional

from loguru import logger
from PIL import Image

from .models import DentalForm, MedicalForm

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    openai = None
    OPENAI_AVAILABLE = False

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

DEFAULT_MODEL = "openai/gpt-4o"
# Other options: "anthropic/claude-3.5-sonnet", "google/gemini-pro-1.5", "meta-llama/llama-3.2-90b-vision-instruct"


# ---------------------------------------------------------------------------
# JSON Schemas
# ---------------------------------------------------------------------------

_SN = {"type": ["STRING", "NULL"]}

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
},

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
        manifest_path = Path(__file__).parent.parent / "logs" / "examples" / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                _EXAMPLES_CACHE = json.load(f)
            logger.info(f"Loaded {len(_EXAMPLES_CACHE)} few-shot examples for OpenRouter")
        else:
            _EXAMPLES_CACHE = []
    except Exception as e:
        logger.warning(f"Could not load examples for OpenRouter: {e}")
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
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{_image_to_base64(img)}"
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

def _configure_openrouter(api_key: str, base_url: str = "https://openrouter.ai/api/v1"):
    """Configure the OpenRouter client."""
    if not OPENAI_AVAILABLE:
        raise RuntimeError("openai package not installed. Run: pip install openai")
    return openai.OpenAI(
        api_key=api_key,
        base_url=base_url,
    )


def _run_openrouter_batched(
    images,
    prompt: str,
    schema: dict,
    model_name: str,
    form_type: str,
    api_key: str,
    examples=None
):
    """Try batched multi-image call first."""
    return _run_openrouter_extraction(images, prompt, schema, model_name, form_type, api_key, examples)


def _run_openrouter_page_by_page(
    images,
    prompt: str,
    schema: dict,
    model_name: str,
    form_type: str,
    api_key: str,
    examples=None
):
    """Fallback: process pages one at a time and merge results."""
    logger.info(f"  Falling back to page-by-page extraction ({len(images)} pages)...")
    merged = {}

    for i, img in enumerate(images):
        try:
            result = _run_openrouter_extraction([img], prompt, schema, model_name, form_type, api_key)
            if result:
                for key, value in result.items():
                    if key not in merged or merged[key] is None:
                        merged[key] = value
                    elif isinstance(value, dict) and isinstance(merged[key], dict):
                        for k, v in value.items():
                            if merged[key].get(k) is None:
                                merged[key][k] = v
                    elif isinstance(value, list) and isinstance(merged[key], list):
                        merged[key].extend(v for v in value if v not in merged[key])
        except Exception as e:
            logger.warning(f"  Page {i+1} failed: {e}")
            continue

    return merged if merged else None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_with_openrouter(
    pages: list[dict],
    has_medical: bool,
    has_dental: bool,
    api_key: str,
    model: str = "openai/gpt-4o",
) -> tuple[Optional[MedicalForm], Optional[DentalForm], list[str]]:
    """
    Use OpenRouter Vision to extract structured form data from rendered PDF page images.

    Pages are classified by content (medical vs dental sections) so that
    each OpenRouter pass only receives the relevant subset of pages.

    Returns (medical_form, dental_form, errors).
    If both forms are None, caller should fall back to next parser.
    """
    if not OPENAI_AVAILABLE:
        return None, None, ["openai package not installed — run: pip install openai"]

    if not any(p.get("image") for p in pages):
        return None, None, ["No page images found — cannot use OpenRouter Vision"]

    if not api_key:
        return None, None, ["OpenRouter API key not configured"]

    from .field_extractor import medical_page_indices, dental_page_indices

    errors: list[str] = []
    medical_form: Optional[MedicalForm] = None
    dental_form: Optional[DentalForm] = None

    examples = _load_examples()

    if has_medical:
        med_idx = medical_page_indices(pages)
        med_pages = [pages[i] for i in med_idx]
        images = [p["image"] for p in med_pages]
        logger.info(f"OpenRouter Vision (medical): {len(images)} pages (classified from {len(pages)} total), model={model}")

        try:
            logger.info("OpenRouter: extracting medical form...")
            data = _run_openrouter_batched(images, MEDICAL_CHECKBOX_PROMPT, MEDICAL_CHECKBOX_SCHEMA,
                                            model, "medical", api_key, examples)
            if data is None:
                data = _run_openrouter_page_by_page(images, MEDICAL_CHECKBOX_PROMPT, MEDICAL_CHECKBOX_SCHEMA,
                                                     model, "medical", api_key, examples)

            medical_form = _build_medical_form(data)
            logger.info("OpenRouter medical extraction succeeded")
        except Exception as e:
            logger.error(f"OpenRouter medical extraction failed: {e}")
            errors.append(f"OpenRouter medical error: {e}")

    if has_dental:
        dnt_idx = dental_page_indices(pages)
        dnt_pages = [pages[i] for i in dnt_idx]
        images = [p["image"] for p in dnt_pages]
        logger.info(f"OpenRouter Vision (dental): {len(images)} pages (classified from {len(pages)} total), model={model}")

        try:
            logger.info("OpenRouter: extracting dental form...")
            data = _run_openrouter_batched(images, DENTAL_PROMPT, DENTAL_SCHEMA,
                                            model, "dental", api_key, examples)
            if data is None:
                data = _run_openrouter_page_by_page(images, DENTAL_PROMPT, DENTAL_SCHEMA,
                                                     model, "dental", api_key, examples)

            dental_form = _build_dental_form(data)
            logger.info("OpenRouter dental extraction succeeded")
        except Exception as e:
            logger.error(f"OpenRouter dental extraction failed: {e}")
            errors.append(f"OpenRouter dental error: {e}")

    return medical_form, dental_form, errors
