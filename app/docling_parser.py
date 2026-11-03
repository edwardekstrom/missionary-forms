"""
Docling VLM parser for LDS missionary health evaluation forms.

Uses IBM's docling library with the SmolDocling VLM (256M params) to convert
scanned PDF pages to structured text/markdown, then extracts form fields via:
  1. Regex parsing (fast, no extra LLM needed)
  2. Optional Ollama text-only pass for improved accuracy

SmolDocling is ~15-30x faster than llama3.2-vision (3-8s/page vs 45-120s/page)
and runs on Apple Silicon MPS without requiring a GPU.

Install: pip install "docling[vlm]"
"""

import io
import json
import os
import tempfile
from pathlib import Path
from typing import Optional

from loguru import logger

from .models import DentalForm, MedicalForm

try:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.base_models import InputFormat
    DOCLING_AVAILABLE = True
    # Try to import VLM-specific options (requires docling[vlm] extra)
    try:
        from docling.datamodel.pipeline_options import VlmPipelineOptions
        from docling.pipeline.vlm_pipeline import VlmPipeline
        DOCLING_VLM_AVAILABLE = True
    except ImportError:
        DOCLING_VLM_AVAILABLE = False
except ImportError:
    DOCLING_AVAILABLE = False
    DOCLING_VLM_AVAILABLE = False

try:
    import ollama as ollama_client
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# Reuse prompts and JSON helpers from vision_shared
from .vision_shared import (
    DENTAL_PROMPT,
    MEDICAL_PROMPT,
    _build_dental_form,
    _build_medical_form,
    _extract_json,
)
from .form_parser import parse_dental_form, parse_medical_form

# Prompt wrappers for text-only Ollama calls after docling extraction
_MEDICAL_TEXT_PROMPT = """\
Below is text extracted from an LDS Missionary Physician's Health Evaluation form.
Using ONLY the information in this text, extract the form fields and return them as JSON.

EXTRACTED FORM TEXT:
{text}

""" + MEDICAL_PROMPT

_DENTAL_TEXT_PROMPT = """\
Below is text extracted from an LDS Missionary Dental Evaluation form.
Using ONLY the information in this text, extract the form fields and return them as JSON.

EXTRACTED FORM TEXT:
{text}

""" + DENTAL_PROMPT


# ---------------------------------------------------------------------------
# Docling conversion
# ---------------------------------------------------------------------------

_converter: Optional["DocumentConverter"] = None  # lazy singleton


def _get_converter() -> "DocumentConverter":
    """Return a cached DocumentConverter (VLM pipeline if available)."""
    global _converter
    if _converter is not None:
        return _converter

    if DOCLING_VLM_AVAILABLE:
        logger.info("Docling: using VLM pipeline (SmolDocling)")
        try:
            pipeline_options = VlmPipelineOptions()
            _converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_cls=VlmPipeline,
                        pipeline_options=pipeline_options,
                    )
                }
            )
            return _converter
        except Exception as e:
            logger.warning(f"Docling VLM pipeline unavailable: {e} — falling back to standard pipeline")

    logger.info("Docling: using standard pipeline (no VLM)")
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    _converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    return _converter


def pdf_to_markdown(pdf_path: str) -> str:
    """
    Convert a PDF to markdown text using docling (VLM pipeline when available).
    Returns the full document as a markdown string.
    """
    converter = _get_converter()
    result = converter.convert(str(pdf_path))
    return result.document.export_to_markdown()


def pdf_to_text_pages(pdf_path: str) -> list[dict]:
    """
    Convert each PDF page to text using docling.
    Returns list of {page, text} dicts.
    """
    converter = _get_converter()
    result = converter.convert(str(pdf_path))
    doc = result.document

    # Try to get per-page text; fall back to full doc text if unavailable
    pages = []
    try:
        page_texts: dict[int, list[str]] = {}
        for item, _ in doc.iterate_items():
            text = getattr(item, "text", None)
            prov = getattr(item, "prov", None)
            if text and prov:
                for p in (prov if isinstance(prov, list) else [prov]):
                    pno = getattr(p, "page_no", 1)
                    page_texts.setdefault(pno, []).append(text)
        if page_texts:
            for pno, texts in sorted(page_texts.items()):
                pages.append({"page": pno, "text": "\n".join(texts)})
    except Exception:
        pass

    if not pages:
        full_text = doc.export_to_markdown()
        pages = [{"page": 1, "text": full_text}]

    return pages


# ---------------------------------------------------------------------------
# Ollama text-only pass (uses docling text rather than raw images)
# ---------------------------------------------------------------------------

def _extract_with_ollama_text(text: str, prompt_template: str, model: str) -> dict:
    """Send docling-extracted text to Ollama as a text-only prompt, get JSON back."""
    if not OLLAMA_AVAILABLE:
        return {}
    prompt = prompt_template.format(text=text[:12000])  # cap to avoid context limit
    try:
        response = ollama_client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.message.content
        return _extract_json(raw)
    except Exception as e:
        logger.debug(f"Ollama text pass failed: {e}")
        return {}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def parse_with_docling(
    pdf_path: str,
    has_medical: bool,
    has_dental: bool,
    ollama_model: str = "qwen2.5vl",
    use_ollama_text_pass: bool = True,
) -> tuple[Optional[MedicalForm], Optional[DentalForm], list[str]]:
    """
    Use docling VLM (SmolDocling) to convert the PDF to structured text,
    then extract form fields via:
      1. Regex parsing on docling's clean text output
      2. Optional Ollama text-only pass if regex extraction is thin

    Returns (medical_form, dental_form, errors).
    """
    if not DOCLING_AVAILABLE:
        return None, None, ["docling not installed — run: pip install 'docling[vlm]'"]

    errors: list[str] = []
    medical_form: Optional[MedicalForm] = None
    dental_form: Optional[DentalForm] = None

    try:
        logger.info(f"Docling: converting {Path(pdf_path).name} …")
        full_text = pdf_to_markdown(pdf_path)
        logger.info(f"Docling: extracted {len(full_text.split())} words")
        logger.debug(f"Docling text preview: {full_text[:500]}")
    except Exception as e:
        return None, None, [f"Docling conversion failed: {e}"]

    # ── Medical form ──────────────────────────────────────────────────────
    if has_medical:
        try:
            med = parse_medical_form(full_text)
            # Check if regex found enough fields
            fields_found = sum(1 for v in vars(med).values() if v is not None) if med else 0

            if fields_found >= 5:
                medical_form = med
                logger.info(f"Docling+regex: medical extraction succeeded ({fields_found} fields)")
            elif use_ollama_text_pass and OLLAMA_AVAILABLE:
                logger.info("Docling+Ollama text: regex sparse, trying Ollama text pass …")
                data = _extract_with_ollama_text(full_text, _MEDICAL_TEXT_PROMPT, ollama_model)
                if data:
                    medical_form = _build_medical_form(data)
                    logger.info("Docling+Ollama text: medical extraction succeeded")
                else:
                    logger.warning("Docling+Ollama text: no JSON returned")
                    errors.append("Docling+Ollama text: medical returned no JSON")
            else:
                medical_form = med  # use whatever regex found
                logger.info(f"Docling+regex: medical extraction (only {fields_found} fields found)")
        except Exception as e:
            logger.error(f"Docling medical extraction failed: {e}")
            errors.append(f"Docling medical error: {e}")

    # ── Dental form ───────────────────────────────────────────────────────
    if has_dental:
        try:
            dnt = parse_dental_form(full_text)
            fields_found = sum(1 for v in vars(dnt).values() if v is not None) if dnt else 0

            if fields_found >= 3:
                dental_form = dnt
                logger.info(f"Docling+regex: dental extraction succeeded ({fields_found} fields)")
            elif use_ollama_text_pass and OLLAMA_AVAILABLE:
                logger.info("Docling+Ollama text: regex sparse, trying Ollama text pass …")
                data = _extract_with_ollama_text(full_text, _DENTAL_TEXT_PROMPT, ollama_model)
                if data:
                    dental_form = _build_dental_form(data)
                    logger.info("Docling+Ollama text: dental extraction succeeded")
                else:
                    logger.warning("Docling+Ollama text: no JSON returned")
                    errors.append("Docling+Ollama text: dental returned no JSON")
            else:
                dental_form = dnt
                logger.info(f"Docling+regex: dental extraction (only {fields_found} fields found)")
        except Exception as e:
            logger.error(f"Docling dental extraction failed: {e}")
            errors.append(f"Docling dental error: {e}")

    return medical_form, dental_form, errors
