"""\
Shared utilities for vision-based parsers (Ollama, formerly Claude).

Contains prompts, JSON extraction helpers, image encoding, PDF rendering,
and model builders used by vision-based form parsers.
"""

from __future__ import annotations

import base64
import io
import json
import re
from pathlib import Path
from typing import Optional

from loguru import logger
from PIL import Image

from .models import (
    DentalForm,
    Immunizations,
    LabResults,
    MedicalForm,
    PersonalInfo,
    PhysicianInfo,
    SystemExam,
    VitalSigns,
)

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
MEDICAL_PROMPT = """\
You are reading scanned pages of an LDS Missionary Physician's Health Evaluation form.
Extract all visible data and return it as a single JSON object.
Use null for fields you cannot find or read. Only extract filled-in values — ignore instructions text.

Return ONLY this JSON structure, no markdown fences, no extra keys:
{
  "physician_info": {
    "name": null, "address": null, "city": null, "state": null,
    "postal_code": null, "phone": null, "email": null
  },
  "functional_level": null,
  "chronic_condition": null,
  "chronic_comments": null,
  "driving_medication": null,
  "driving_medication_explain": null,
  "exam_date": null,
  "vital_signs": {
    "height": null, "weight": null, "blood_pressure": null,
    "pulse": null, "vision_right": null, "vision_left": null
  },
  "system_exam": {
    "general": null, "skin": null, "eyes": null, "ears": null,
    "nose_throat": null, "chest": null, "heart": null, "abdomen": null,
    "genitalia": null, "back": null, "extremities": null, "neurological": null
  },
  "labs": {
    "urinalysis_blood": null, "urinalysis_protein": null, "urinalysis_sugar": null,
    "hemoglobin": null, "tb_exposure_risk": null, "ppd_result": null,
    "igra_result": null, "chest_xray": null
  },
  "immunizations": {
    "tdap": null, "tdap2": null, "mmr1": null, "mmr2": null, "polio": null,
    "hep_a1": null, "hep_a2": null,
    "hep_b1": null, "hep_b2": null, "hep_b3": null,
    "influenza": null, "covid_doses": null,
    "covid_brand": null
  }
}

Formatting rules:
- Yes/No questions: exactly "Yes" or "No"
- Dates: "MM/DD/YYYY"
- functional_level: single letter "A" through "E" (find the checked level in the Assessment section)
- system_exam values: exactly "Normal", "Abnormal", or "Not Indicated"
- blood_pressure: "systolic/diastolic" e.g. "120/80"
- vision: "20/XX" e.g. "20/20" or "20/25"
- height: numeric + unit e.g. "68 in" or "172 cm"
- weight: numeric + unit e.g. "155 lbs" or "70 kg"
- hemoglobin: numeric only e.g. "16.1"
- igra_result: exactly "Negative", "Positive", or "Not Done"
- chest_xray: exactly "Normal", "Abnormal", or "Not Done"
- ppd_result: e.g. "5 mm", "Negative", or "Not Done"
- immunization fields (tdap, tdap2, mmr1, mmr2, polio, hep_a1, hep_a2, hep_b1, hep_b2, hep_b3, influenza): the actual vaccination DATE e.g. "04/21/2008". Do NOT use "Yes" — use null if you see a checkmark but no date.
- CRITICAL for multi-dose vaccines: assign doses in CHRONOLOGICAL ORDER, earliest date first. tdap = the OLDEST/FIRST Tetanus-diphtheria-pertussis date on the form, tdap2 = the SECOND date (if present). mmr1 = oldest MMR dose, mmr2 = second MMR dose. hep_b1 = oldest Hep B, hep_b2 = middle, hep_b3 = most recent. hep_a1 = oldest Hep A, hep_a2 = more recent. If there are 3+ Tetanus dates on the form, put the two EARLIEST dates in tdap and tdap2. Never skip earlier doses to fill a later slot.  Never put the same date for different doses (tdap and tdap2 never same date, mmr1 and mmr2 never same date, hep_a1 and hep_a2 never same date, hep_b1 and hep_b2 and hep_b3 never same date).
- covid_doses: array of date strings e.g. ["05/21/2021", "06/18/2021"]
- covid_brand: the vaccine brand name if visible (e.g. "Pfizer", "Moderna", "J&J")
"""



MEDICAL_CHECKBOX_PROMPT = """\
You are reading a scanned page of an LDS Missionary Physician's Health Evaluation form.
Focus ONLY on identifying which checkboxes and radio buttons are SELECTED. Ignore instructions text.

HOW TO IDENTIFY A SELECTED CHECKBOX IN THIS FORM:
- A filled dark circle, checkmark, slash mark (/), or X next to an option = SELECTED
- An empty circle outline or empty square = NOT selected
- The word "Normal"/"Abnormal" by itself does NOT mean it is selected — look for the visual mark next to it

Return ONLY this JSON, no markdown fences:
{
  "functional_level": null,
  "chronic_condition": null,
  "driving_medication": null,
  "system_exam": {
    "general": null, "skin": null, "eyes": null, "ears": null,
    "nose_throat": null, "chest": null, "heart": null, "abdomen": null,
    "genitalia": null, "back": null, "extremities": null, "neurological": null
  },
  "labs": {
    "tb_exposure_risk": null,
    "igra_result": null,
    "chest_xray": null
  }
}

Rules (only for VISUALLY SELECTED fields — use null if not on this page):
- functional_level: the SELECTED level letter "A","B","C","D", or "E"
- chronic_condition, driving_medication, tb_exposure_risk: "Yes" or "No"
- system_exam values: "Normal" or "Abnormal" — only fill in if you can see a selection mark
- igra_result: "Negative", "Positive", or "Not Done"
- chest_xray: "Normal", "Abnormal", or "Not Done"
"""



DENTAL_PROMPT = """\
You are reading scanned pages of an LDS Missionary Dental Evaluation form.
Extract all visible data and return it as a single JSON object.
Use null for fields you cannot find or read. Only extract filled-in values — ignore instructions text.

Return ONLY this JSON structure, no markdown fences, no extra keys:
{
  "orthodontic_treatment": null,
  "wisdom_teeth_removed": null,
  "brushing_frequency": null,
  "flossing_frequency": null,
  "pain_or_bleeding": null,
  "bleeding_comments": null,
  "dentist_name": null,
  "dentist_address": null,
  "dentist_city": null,
  "dentist_state": null,
  "dentist_postal": null,
  "dentist_phone": null,
  "dentist_email": null,
  "exam_date": null,
  "oral_exam_with_bitewing": null,
  "third_molars_assessed": null,
  "panoramic_done": null,
  "active_orthodontic": null,
  "dental_decay": null,
  "bonded_retainers": null,
  "oral_hygiene_acceptable": null,
  "dental_comments": null
}

Formatting rules:
- Yes/No questions: exactly "Yes" or "No"
- Yes/No/Not Applicable questions (third_molars_assessed, panoramic_done, active_orthodontic, bonded_retainers): "Yes", "No", or "Not Applicable"
- Dates: "MM/DD/YYYY". For exam_date, use only dates from 2024 or later; return null if no recent date is found.
- brushing_frequency and flossing_frequency: copy the written text as-is (e.g. "Morning and Night", "Daily", "Twice daily")

FIELD GUIDE — map these JSON keys to the exact questions on the form:
- oral_exam_with_bitewing: "Has the prospective missionary had a complete oral examination WITH BITEWING RADIOGRAPHS within the last six months?"
- panoramic_done: "Has a PANORAMIC OR EQUIVALENT IMAGE been taken in the last 6-12 months?" (for evaluating third molars)
- third_molars_assessed: "Have all third molars that were likely to become problematic been EXTRACTED?" (this is about EXTRACTION, not the panoramic image)
- dental_decay: "Has all dental decay and gum infection been RESOLVED?" — "Yes" means decay has been resolved/treated (good outcome). Do not confuse with "does patient have decay?"
- active_orthodontic: "Has all active orthodontic treatment been COMPLETED?" — "Yes" means treatment is complete
- dental_decay, wisdom_teeth_removed, orthodontic_treatment: these are very commonly "Yes" on missionary forms. Look carefully.

CRITICAL — reading checkboxes and circled options:
- Dental forms mark answers by filling in a circle (●), adding a checkmark (✓), or circling the word.
- An empty circle (○) or empty box means NOT selected.
- Do NOT default to "No" — look carefully at EACH checkbox to see if it has a mark inside.
- If a circle next to "Yes" is filled or has a mark, return "Yes". Same for "No" and "Not Applicable".
- If you genuinely cannot tell which option is marked, return null (do not guess "No").
"""




# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _image_to_base64(img: Image.Image, max_dimension: int = 1600) -> str:
    """Resize to keep the longest edge ≤ max_dimension, then encode as base64 PNG."""
    w, h = img.size
    if max(w, h) > max_dimension:
        scale = max_dimension / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def _pages_to_image_blocks(pages: list[dict]) -> list[dict]:
    """Convert page dicts (with 'image' keys) to Anthropic API content blocks."""
    blocks = []
    for p in pages:
        img = p.get("image")
        if img:
            blocks.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": _image_to_base64(img),
                },
            })
    return blocks


def _redact_name_strip(img: Image.Image, top_fraction: float = 0.10) -> Image.Image:
    """
    Black out the top strip of a form page to remove the candidate's name/PII
    before sending to a cloud API.

    The candidate name appears in:
      - Medical page 1: top ~10% (name, DOB, ward)
      - Dental page:    top ~18% (name + authorization section above dental history)
    Other pages get a 10% redaction to cover any running header.
    """
    from PIL import ImageDraw
    img = img.copy().convert("RGB")
    ImageDraw.Draw(img).rectangle(
        [0, 0, img.width, int(img.height * top_fraction)],
        fill=(0, 0, 0),
    )
    return img


def _extract_json(text: str) -> dict:
    """Strip markdown fences and parse JSON from Claude's response."""
    # Remove ```json ... ``` or ``` ... ``` wrappers
    text = re.sub(r"```(?:json)?\s*\n?", "", text)
    text = text.rstrip("`").strip()
    # Find the outermost { ... } in case there's stray text
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        text = text[start:end]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # LLMs occasionally produce slightly malformed JSON; try to repair it
        try:
            from json_repair import repair_json
            return json.loads(repair_json(text))
        except Exception:
            raise




# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def _norm_yn(v) -> Optional[str]:
    """Normalize Yes/No variants from LLM output."""
    if v is None:
        return None
    s = str(v).strip().lower().rstrip(".")
    if s in ("yes", "true", "1", "y", "checked", "positive", "marked"):
        return "Yes"
    if s in ("no", "false", "0", "n", "unchecked", "negative", "unmarked"):
        return "No"
    return str(v).strip()


def _norm_exam(v) -> Optional[str]:
    """Normalize Normal/Abnormal/Not Indicated from LLM output."""
    if v is None:
        return None
    s = str(v).strip().lower().rstrip(".")
    if s in ("normal", "norm", "n"):
        return "Normal"
    if s in ("abnormal", "abnorm", "ab", "a"):
        return "Abnormal"
    if s in ("not indicated", "not indicated", "ni", "n/a", "na"):
        return None  # treat as missing rather than wrong
    return str(v).strip()


def _norm_functional(v) -> Optional[str]:
    """Normalize functional level to single letter A-E."""
    if v is None:
        return None
    s = str(v).strip()
    # Handle "Level A", "A - No limitations", "a", etc.
    m = re.search(r"\b([A-Ea-e])\b", s)
    if m:
        return m.group(1).upper()
    return s


def _norm_igra(v) -> Optional[str]:
    """Normalize IGRA result."""
    if v is None:
        return None
    s = str(v).strip().lower()
    if "neg" in s:
        return "Negative"
    if "pos" in s:
        return "Positive"
    if "not" in s or "done" in s or "n/a" in s:
        return "Not Done"
    return str(v).strip()


def _norm_xray(v) -> Optional[str]:
    """Normalize chest x-ray result."""
    if v is None:
        return None
    s = str(v).strip().lower()
    if "normal" in s and "abnormal" not in s:
        return "Normal"
    if "abnormal" in s:
        return "Abnormal"
    if "not" in s or "done" in s:
        return "Not Done"
    return str(v).strip()


def _norm_date(v) -> Optional[str]:
    """Normalize date to MM/DD/YYYY."""
    if v is None:
        return None
    s = str(v).strip()
    if s.lower() in ("null", "none", "n/a", "not done", ""):
        return None
    return s  # validator comparison handles date normalization


def _build_personal_info(d: Optional[dict]) -> PersonalInfo:
    if not d:
        return PersonalInfo()
    return PersonalInfo(
        first_name=d.get("first_name"),
        middle_name=d.get("middle_name"),
        last_name=d.get("last_name"),
        record_number=str(d["record_number"]) if d.get("record_number") else None,
        date_of_birth=d.get("date_of_birth"),
        age=str(d["age"]) if d.get("age") is not None else None,
        gender=d.get("gender"),
    )


def _build_medical_form(data: dict) -> MedicalForm:
    form = MedicalForm()
    form.personal_info = _build_personal_info(data.get("personal_info"))

    pi = data.get("physician_info") or {}
    form.physician_info = PhysicianInfo(
        name=pi.get("name"),
        credentials=pi.get("credentials"),
        address=pi.get("address"),
        city=pi.get("city"),
        state=pi.get("state"),
        postal_code=str(pi["postal_code"]) if pi.get("postal_code") else None,
        phone=str(pi["phone"]) if pi.get("phone") else None,
        email=pi.get("email"),
    )

    form.functional_level      = _norm_functional(data.get("functional_level"))
    form.chronic_condition     = _norm_yn(data.get("chronic_condition"))
    form.chronic_comments      = data.get("chronic_comments")
    form.driving_medication    = _norm_yn(data.get("driving_medication"))
    form.driving_medication_explain = data.get("driving_medication_explain")
    form.exam_date             = _norm_date(data.get("exam_date"))

    vs = data.get("vital_signs") or {}
    form.vital_signs = VitalSigns(
        height=vs.get("height"),
        weight=vs.get("weight"),
        blood_pressure=vs.get("blood_pressure"),
        pulse=str(vs["pulse"]) if vs.get("pulse") is not None else None,
        vision_right=vs.get("vision_right"),
        vision_left=vs.get("vision_left"),
    )

    se = data.get("system_exam") or {}
    form.system_exam = SystemExam(
        general    =_norm_exam(se.get("general")),
        skin       =_norm_exam(se.get("skin")),
        eyes       =_norm_exam(se.get("eyes")),
        ears       =_norm_exam(se.get("ears")),
        nose_throat=_norm_exam(se.get("nose_throat")),
        chest      =_norm_exam(se.get("chest")),
        heart      =_norm_exam(se.get("heart")),
        abdomen    =_norm_exam(se.get("abdomen")),
        genitalia  =_norm_exam(se.get("genitalia")),
        back       =_norm_exam(se.get("back")),
        extremities=_norm_exam(se.get("extremities")),
        neurological=_norm_exam(se.get("neurological")),
    )

    lab = data.get("labs") or {}
    form.labs = LabResults(
        urinalysis_blood=lab.get("urinalysis_blood"),
        urinalysis_protein=lab.get("urinalysis_protein"),
        urinalysis_sugar=lab.get("urinalysis_sugar"),
        hemoglobin=str(lab["hemoglobin"]) if lab.get("hemoglobin") is not None else None,
        tb_exposure_risk=_norm_yn(lab.get("tb_exposure_risk")),
        ppd_result=lab.get("ppd_result"),
        igra_result=_norm_igra(lab.get("igra_result")),
        chest_xray=_norm_xray(lab.get("chest_xray")),
    )

    imm = data.get("immunizations") or {}
    covid = imm.get("covid_doses")
    if isinstance(covid, str):
        covid = [covid] if covid else None

    form.immunizations = Immunizations(
        tdap     =_norm_date(imm.get("tdap")),
        tdap2    =_norm_date(imm.get("tdap2")),
        mmr1     =_norm_date(imm.get("mmr1")),
        mmr2     =_norm_date(imm.get("mmr2")),
        polio    =_norm_date(imm.get("polio")),
        hep_a1   =_norm_date(imm.get("hep_a1")),
        hep_a2   =_norm_date(imm.get("hep_a2")),
        hep_b1   =_norm_date(imm.get("hep_b1")),
        hep_b2   =_norm_date(imm.get("hep_b2")),
        hep_b3   =_norm_date(imm.get("hep_b3")),
        influenza=_norm_date(imm.get("influenza")),
        covid_doses=covid,
        other=imm.get("covid_brand"),
    )

    return form


def _norm_yn_na(v) -> Optional[str]:
    """Normalize Yes / No / Not Applicable variants."""
    if v is None:
        return None
    s = str(v).strip().lower().rstrip(".")
    if s in ("yes", "true", "1", "y"):
        return "Yes"
    if s in ("no", "false", "0", "n"):
        return "No"
    if "not applicable" in s or s in ("na", "n/a"):
        return "Not Applicable"
    return str(v).strip()


def _build_dental_form(data: dict) -> DentalForm:
    form = DentalForm()
    form.personal_info = _build_personal_info(data.get("personal_info"))
    form.orthodontic_treatment = _norm_yn_na(data.get("orthodontic_treatment"))
    form.orthodontic_complete  = form.orthodontic_treatment
    form.wisdom_teeth_removed  = _norm_yn_na(data.get("wisdom_teeth_removed"))
    form.brushing_frequency    = data.get("brushing_frequency")
    form.flossing_frequency    = data.get("flossing_frequency")
    form.pain_or_bleeding      = _norm_yn(data.get("pain_or_bleeding"))
    form.bleeding_comments     = data.get("bleeding_comments")
    form.dentist_name          = data.get("dentist_name")
    form.dentist_address       = data.get("dentist_address")
    form.dentist_city          = data.get("dentist_city")
    form.dentist_state         = data.get("dentist_state")
    form.dentist_postal        = str(data["dentist_postal"]) if data.get("dentist_postal") else None
    form.dentist_phone         = str(data["dentist_phone"])  if data.get("dentist_phone")  else None
    form.dentist_email         = data.get("dentist_email")
    form.exam_date             = _norm_date(data.get("exam_date"))
    form.oral_exam_with_bitewing = _norm_yn_na(data.get("oral_exam_with_bitewing"))
    form.third_molars_assessed   = _norm_yn_na(data.get("third_molars_assessed"))
    form.panoramic_done          = _norm_yn_na(data.get("panoramic_done"))
    form.active_orthodontic      = _norm_yn_na(data.get("active_orthodontic"))
    form.dental_decay            = _norm_yn(data.get("dental_decay"))
    form.bonded_retainers        = _norm_yn_na(data.get("bonded_retainers"))
    form.oral_hygiene_acceptable = _norm_yn(data.get("oral_hygiene_acceptable"))
    form.dental_comments         = data.get("dental_comments")
    return form


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def render_pages(pdf_path: str, dpi: int = 150) -> list[dict]:
    """
    Render all PDF pages to PIL Images at the given DPI.
    Returns a list of {page, image} dicts suitable for vision parsers.
    Uses low DPI (150) since vision models do not need the full 300 DPI
    required by traditional OCR.
    """
    import fitz  # PyMuPDF
    doc = fitz.open(pdf_path)
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pages = []
    for i, page in enumerate(doc, start=1):
        pix = page.get_pixmap(matrix=mat)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        pages.append({"page": i, "image": img})
    return pages


