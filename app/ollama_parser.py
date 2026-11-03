"""
Ollama Vision-based parser for LDS missionary health evaluation forms.

Uses a local vision model (qwen2.5vl) via Ollama to extract structured form
data from PDF page images — fully local, no API key needed.

Three improvements over the original llama3.2-vision approach:
  1. JSON Schema enforcement: passes a full schema as `format=` so the model
     cannot output wrong keys or structure (not just "valid JSON").
  2. Multi-image batching: all form pages are sent in a single call so the
     model has cross-page context (qwen2.5vl supports multiple images per call).
  3. Falls back to page-by-page if the batched call fails or times out.

Falls back gracefully: returns None forms if Ollama is unavailable or errors.
Caller (form_parser.py) then uses regex parsing instead.
"""

import base64
import json
import subprocess
import time
from pathlib import Path
from typing import Optional

from loguru import logger

from .models import DentalForm, MedicalForm

try:
    import ollama
    # Use a Client with a per-request timeout so hung pages don't block forever.
    _OLLAMA_CLIENT = ollama.Client(timeout=180)   # longer for multi-image calls
    OLLAMA_AVAILABLE = True
except ImportError:
    _OLLAMA_CLIENT = None
    OLLAMA_AVAILABLE = False

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
# JSON Schemas for constrained Ollama generation
# ---------------------------------------------------------------------------
# Passing a schema as format= instead of format="json" does constrained
# decoding: the model physically cannot output tokens that violate the schema.
# This eliminates wrong key names, missing nested objects, and type errors.

_SN = {"type": ["string", "null"]}   # nullable string shorthand

MEDICAL_CHECKBOX_SCHEMA = {
    "type": "object",
    "properties": {
        "functional_level": _SN,
        "chronic_condition": _SN,
        "driving_medication": _SN,
        "system_exam": {
            "type": "object",
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
            "type": "object",
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
    "type": "object",
    "properties": {k: _SN for k in [
        "orthodontic_treatment", "wisdom_teeth_removed",
        "brushing_frequency", "flossing_frequency",
        "pain_or_bleeding", "bleeding_comments",
        "dentist_name", "dentist_address", "dentist_city",
        "dentist_state", "dentist_postal", "dentist_phone", "dentist_email",
        "exam_date", "oral_exam_with_bitewing", "third_molars_assessed",
        "panoramic_done", "active_orthodontic", "dental_decay",
        "bonded_retainers", "oral_hygiene_acceptable", "dental_comments",
    ]},
    "required": [
        "orthodontic_treatment", "wisdom_teeth_removed",
        "brushing_frequency", "flossing_frequency", "pain_or_bleeding",
        "dentist_name", "dentist_city", "dentist_state", "dentist_phone",
        "exam_date", "oral_exam_with_bitewing", "third_molars_assessed",
        "panoramic_done", "active_orthodontic", "dental_decay",
        "bonded_retainers", "oral_hygiene_acceptable",
    ],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, update: dict) -> dict:
    """Recursively merge update into base, keeping first non-null value."""
    result = dict(base)
    for k, v in update.items():
        if k not in result or result[k] is None:
            result[k] = v
        elif isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
    return result


_OLLAMA_PAGE_TIMEOUT = 180  # seconds to keep model alive between pages

_EXAMPLES_DIR = Path(__file__).parent.parent / "logs" / "examples"
_MANIFEST_PATH = _EXAMPLES_DIR / "manifest.json"
_EXAMPLES_CACHE: Optional[dict] = None
_EXAMPLES_LOADED: bool = False


def _restart_ollama_runner() -> None:
    """Kill the Ollama runner process so Ollama reloads the model on next request."""
    try:
        result = subprocess.run(
            ["pkill", "-f", "ollama runner"],
            capture_output=True,
        )
        if result.returncode == 0:
            logger.info("  Restarted Ollama runner (was stuck); waiting 15 s …")
            time.sleep(15)
        else:
            logger.debug("  pkill ollama runner: nothing to kill")
    except Exception as exc:
        logger.debug(f"  _restart_ollama_runner: {exc}")


def _pages_to_images_b64(pages: list[dict], max_dimension: int = 1200) -> list[str]:
    """Convert page dicts to base64-encoded image strings for Ollama."""
    result = []
    for p in pages:
        img = p.get("image")
        if img is None:
            continue
        result.append(_image_to_base64(img, max_dimension=max_dimension))
    return result


# ---------------------------------------------------------------------------
# Few-shot example support
# ---------------------------------------------------------------------------

def _load_examples() -> Optional[dict]:
    """
    Load few-shot example crops from logs/examples/manifest.json.
    Pre-encodes all images to base64 at load time and caches permanently.
    Returns None gracefully if manifest missing or corrupt.
    """
    global _EXAMPLES_CACHE, _EXAMPLES_LOADED
    if _EXAMPLES_LOADED:
        return _EXAMPLES_CACHE
    _EXAMPLES_LOADED = True

    if not _MANIFEST_PATH.exists():
        logger.debug(f"Few-shot examples not found at {_MANIFEST_PATH} — skipping")
        return None

    try:
        with open(_MANIFEST_PATH) as f:
            manifest = json.load(f)

        # Pre-encode all example images to base64
        for form_type in ("medical", "dental"):
            for entry in manifest.get(form_type, []):
                img_path = _EXAMPLES_DIR / entry["image"]
                if img_path.exists():
                    with open(img_path, "rb") as f:
                        entry["image_b64"] = base64.b64encode(f.read()).decode()
                else:
                    logger.warning(f"Example image not found: {img_path}")
                    entry["image_b64"] = None

        # Filter out entries whose images couldn't be loaded
        for form_type in ("medical", "dental"):
            manifest[form_type] = [
                e for e in manifest.get(form_type, [])
                if e.get("image_b64") is not None
            ]

        med_n = len(manifest.get("medical", []))
        dnt_n = len(manifest.get("dental", []))
        logger.info(f"Loaded {med_n} medical + {dnt_n} dental few-shot examples")
        _EXAMPLES_CACHE = manifest
        return manifest
    except Exception as e:
        logger.warning(f"Failed to load few-shot examples: {e}")
        return None


def _build_few_shot_messages(
    form_type: str,
    examples: Optional[dict],
    final_images_b64: list[str],
    final_prompt: str,
) -> list[dict]:
    """
    Build the Ollama message list with optional few-shot example context.

    If examples are available, returns a 3-turn list:
      1. user: labeled example crops + guide text
      2. assistant: acknowledgment of checkbox conventions
      3. user: actual form pages + extraction prompt

    Falls back to single-turn (original behavior) if no examples.
    """
    entries = (examples or {}).get(form_type, []) if examples else []
    entries = [e for e in entries if e.get("image_b64")]

    if not entries:
        return [{"role": "user", "content": final_prompt, "images": final_images_b64}]

    # Build the guide text referencing each numbered example image
    lines = [
        f"Here are {len(entries)} labeled reference sections from a real missionary form. "
        "Study these examples to learn what selected vs. unselected marks look like "
        "in this specific form template.\n",
    ]
    for i, entry in enumerate(entries, start=1):
        lines.append(f"Image {i}: {entry['description']}")
        vals = ", ".join(
            f"{k}={v!r}" for k, v in (entry.get("correct_values") or {}).items()
        )
        if vals:
            lines.append(f"  Correct values from this section: {vals}")
        lines.append("")
    lines.append(
        "Use these visual examples as your reference. "
        "A FILLED dark circle or visible checkmark means SELECTED. "
        "An empty circle means NOT SELECTED."
    )

    return [
        {
            "role": "user",
            "content": "\n".join(lines),
            "images": [e["image_b64"] for e in entries],
        },
        {
            "role": "assistant",
            "content": (
                "I understand the form template. I can see the difference between "
                "filled and empty circles. I'll use these reference sections to "
                "accurately read the marks on the actual form pages."
            ),
        },
        {
            "role": "user",
            "content": final_prompt,
            "images": final_images_b64,
        },
    ]


# ---------------------------------------------------------------------------
# Core inference functions
# ---------------------------------------------------------------------------

def _parse_pages_batched(
    images_b64: list[str],
    prompt: str,
    model: str,
    schema: dict,
    form_type: str = "",
    examples: Optional[dict] = None,
) -> Optional[dict]:
    """
    Send all page images in a SINGLE Ollama call with schema-constrained output.

    qwen2.5vl supports multiple images per call, giving the model full cross-page
    context (e.g. can see functional-level checkbox on page 1 and system-exam
    checkboxes on page 3 simultaneously).

    Returns the parsed dict, or None if the call fails (caller should fall back).
    """
    if not images_b64:
        return {}

    n = len(images_b64)
    multi_prompt = (
        f"You are reading {n} page{'s' if n > 1 else ''} of an LDS Missionary form. "
        "All pages are shown together — look across all of them to find every field.\n\n"
        + prompt
    )

    messages = _build_few_shot_messages(form_type, examples, images_b64, multi_prompt)
    has_examples = len(messages) > 1
    num_ctx = 32768 if has_examples else 16384

    logger.info(f"  Ollama: batched call — {n} page(s)" + (" + few-shot examples" if has_examples else "") + " …")
    try:
        response = _OLLAMA_CLIENT.chat(
            model=model,
            messages=messages,
            options={
                "num_ctx": num_ctx,             # larger when few-shot examples present
                "num_predict": 1000,            # full form JSON
                "temperature": 0,
            },
            format=schema,                      # constrained JSON Schema output
            keep_alive=f"{_OLLAMA_PAGE_TIMEOUT}s",
        )
        raw = response.message.content
        logger.debug(f"  Batched raw (first 400 chars): {raw[:400]}")
        data = _extract_json(raw)
        if isinstance(data, list) and data and isinstance(data[0], dict):
            logger.warning("  Batched: model returned list — using first element")
            data = data[0]
        if isinstance(data, dict):
            logger.info("  Ollama: batched call succeeded")
            return data
        logger.warning(f"  Batched: unexpected type {type(data).__name__}")
        return None
    except Exception as e:
        err_str = str(e).lower()
        logger.warning(f"  Batched call failed: {e}")
        if "timed out" in err_str:
            _restart_ollama_runner()
        return None


def _parse_pages_one_at_a_time(
    images_b64: list[str],
    prompt: str,
    model: str,
    schema: dict,
    form_type: str = "",
    examples: Optional[dict] = None,
) -> dict:
    """
    Fallback: send each page individually and deep-merge results.
    Used when the batched call fails (e.g. OOM with many images).
    """
    merged: dict = {}
    has_examples = bool((examples or {}).get(form_type))
    num_ctx = 16384 if has_examples else 4096
    for i, img_b64 in enumerate(images_b64, start=1):
        logger.info(f"  Ollama: processing page {i}/{len(images_b64)} …")
        messages = _build_few_shot_messages(form_type, examples, [img_b64], prompt)
        try:
            response = _OLLAMA_CLIENT.chat(
                model=model,
                messages=messages,
                options={
                    "num_ctx": num_ctx,
                    "num_predict": 600,
                    "temperature": 0,
                },
                format=schema,
                keep_alive=f"{_OLLAMA_PAGE_TIMEOUT}s",
            )
            raw = response.message.content
            logger.debug(f"  Page {i} raw (first 300 chars): {raw[:300]}")
            data = _extract_json(raw)
            if isinstance(data, list) and data and isinstance(data[0], dict):
                logger.warning(f"  Page {i}: Ollama returned a list — using first element")
                data = data[0]
            if not isinstance(data, dict):
                logger.warning(f"  Page {i}: skipped — unexpected JSON type {type(data).__name__}")
                continue
            merged = _deep_merge(merged, data)
            logger.info(f"  Ollama: page {i}/{len(images_b64)} done")
        except Exception as e:
            err_str = str(e).lower()
            logger.warning(f"  Page {i}: skipped — {e}")
            if "timed out" in err_str:
                _restart_ollama_runner()
                logger.info(f"  Retrying page {i}/{len(images_b64)} after runner restart …")
                try:
                    response = _OLLAMA_CLIENT.chat(
                        model=model,
                        messages=messages,
                        options={
                            "num_ctx": num_ctx,
                            "num_predict": 600,
                            "temperature": 0,
                        },
                        format=schema,
                        keep_alive=f"{_OLLAMA_PAGE_TIMEOUT}s",
                    )
                    raw = response.message.content
                    data = _extract_json(raw)
                    if isinstance(data, list) and data and isinstance(data[0], dict):
                        data = data[0]
                    if isinstance(data, dict):
                        merged = _deep_merge(merged, data)
                        logger.info(f"  Retry page {i}/{len(images_b64)} done")
                    else:
                        logger.warning(f"  Retry page {i}: unexpected JSON type {type(data).__name__}")
                except Exception as e2:
                    logger.warning(f"  Retry page {i}: also failed — {e2}")
    return merged


def _run_extraction(
    images_b64: list[str],
    prompt: str,
    model: str,
    schema: dict,
    form_type: str = "",
    examples: Optional[dict] = None,
) -> dict:
    """
    Try batched first; fall back to page-by-page.
    """
    result = _parse_pages_batched(images_b64, prompt, model, schema, form_type, examples)
    if result is not None:
        return result
    logger.info("  Falling back to page-by-page extraction …")
    return _parse_pages_one_at_a_time(images_b64, prompt, model, schema, form_type, examples)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_with_ollama(
    pages: list[dict],
    has_medical: bool,
    has_dental: bool,
    model: str = "qwen2.5vl",
) -> tuple[Optional[MedicalForm], Optional[DentalForm], list[str]]:
    """
    Use a local Ollama vision model to extract structured form data from
    rendered PDF page images.

    Pages are classified by content (medical vs dental sections) so that
    each Ollama pass only receives the relevant subset of pages.

    All relevant pages are sent in a single batched call (qwen2.5vl supports
    multi-image input).  Schema-constrained output ensures the model returns
    exactly the right JSON structure.

    Returns (medical_form, dental_form, errors).
    If both forms are None, caller should fall back to regex parsing.
    """
    if not OLLAMA_AVAILABLE:
        return None, None, ["ollama package not installed — run: pip install ollama"]

    if not any(p.get("image") for p in pages):
        return None, None, ["No page images found — cannot use Ollama Vision"]

    from .field_extractor import medical_page_indices, dental_page_indices

    errors: list[str] = []
    medical_form: Optional[MedicalForm] = None
    dental_form: Optional[DentalForm] = None

    examples = _load_examples()

    if has_medical:
        med_idx = medical_page_indices(pages)
        med_pages = [pages[i] for i in med_idx]
        images_b64 = _pages_to_images_b64(med_pages, max_dimension=800)  # keep batched call manageable
        logger.info(
            f"Ollama Vision (medical): {len(images_b64)} pages "
            f"(classified from {len(pages)} total), model={model}"
        )
        try:
            logger.info("Ollama: extracting medical form …")
            data = _run_extraction(images_b64, MEDICAL_CHECKBOX_PROMPT, model, MEDICAL_CHECKBOX_SCHEMA,
                                   form_type="medical", examples=examples)
            medical_form = _build_medical_form(data)
            logger.info("Ollama medical extraction succeeded")
        except Exception as e:
            logger.error(f"Ollama medical extraction failed: {e}")
            errors.append(f"Ollama medical error: {e}")

    if has_dental:
        dnt_idx = dental_page_indices(pages)
        dnt_pages = [pages[i] for i in dnt_idx]
        images_b64 = _pages_to_images_b64(dnt_pages, max_dimension=1200)  # high res for checkbox clarity
        logger.info(
            f"Ollama Vision (dental): {len(images_b64)} pages "
            f"(classified from {len(pages)} total), model={model}"
        )
        try:
            logger.info("Ollama: extracting dental form …")
            data = _run_extraction(images_b64, DENTAL_PROMPT, model, DENTAL_SCHEMA,
                                   form_type="dental", examples=examples)
            dental_form = _build_dental_form(data)
            logger.info("Ollama dental extraction succeeded")
        except Exception as e:
            logger.error(f"Ollama dental extraction failed: {e}")
            errors.append(f"Ollama dental error: {e}")

    return medical_form, dental_form, errors
