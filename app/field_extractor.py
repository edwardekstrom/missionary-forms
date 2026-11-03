"""
Field-level extraction helpers for PDF parsing.

Two complementary strategies:

1. Page classification (classify_pages)
   Use OCR text keywords to identify which pages belong to which form section
   (physician info, vitals, system exam, labs, immunizations, dental).
   Lets Ollama receive only the 1-3 relevant pages instead of all 12, which
   roughly triples throughput and reduces hallucination from irrelevant content.

2. Label-region cropping (find_label_region / crop_for_field)
   Use OCR bounding boxes to find a field label on the page, then crop a small
   rectangle around it.  The tiny crop (often <10% of page area) is sent to
   Ollama with a targeted single-field question rather than a generic "extract
   everything" prompt, dramatically cutting inference time and improving accuracy.

Both strategies degrade gracefully: if bounding-box data is absent or the label
is not found, the caller falls back to sending the full page.
"""

from __future__ import annotations

import re
from typing import Optional

from PIL import Image


# ---------------------------------------------------------------------------
# Section → keyword mapping  (for page-content classification)
# ---------------------------------------------------------------------------

PAGE_SECTION_KEYWORDS: dict[str, list[str]] = {
    "physician_info": [
        "evaluating physician", "name of physician", "physician name",
        "office address", "office phone", "medical officer",
        "assessment of functional", "functional ability",
    ],
    "vitals": [
        "blood pressure", "vital signs", "vision test",
        "height and weight", "pulse rate", "heart rate",
    ],
    "system_exam": [
        "general appearance", "physical examination", "system review",
        "body system", "genitalia", "extremities", "neurological",
        "physical exam findings",
    ],
    "labs": [
        "hemoglobin", "hematocrit", "urinalysis", "dipstick",
        "igra", "interferon gamma", "ppd", "tuberculin",
        "chest x-ray", "chest xray", "chest x ray",
    ],
    "immunizations": [
        "tdap", "mmr", "immunization record", "vaccination date",
        "hepatitis a", "hepatitis b", "varicella", "meningococcal",
        "tetanus/diphtheria", "immunization",
    ],
    "dental": [
        "dental evaluation", "oral examination", "dental examination",
        "dentist", "orthodontic", "wisdom teeth", "oral exam",
        "bitewing", "panoramic", "brushing", "flossing",
        "oral hygiene", "dental decay", "caries", "retainer",
    ],
}

# ---------------------------------------------------------------------------
# Field → label strings  (for bounding-box-based region finding)
# Each field has an ordered list of label strings to search for in OCR obs.
# ---------------------------------------------------------------------------

FIELD_LABELS: dict[str, list[str]] = {
    # ── Physician / dentist contact info ──────────────────────────────────
    "physician_name":        ["Physician's Name", "Physician Name", "Name of Physician", "Doctor"],
    "physician_city":        ["City"],
    "physician_state":       ["State", "Province"],
    "physician_phone":       ["Telephone", "Phone"],
    "exam_date":             ["Date of Exam", "Examination Date", "Date Examined", "Exam Date"],
    "dentist_name":          ["Dentist's Name", "Dentist Name", "Name of Dentist"],
    "dentist_city":          ["City"],
    "dentist_phone":         ["Telephone", "Phone"],

    # ── Vital signs ───────────────────────────────────────────────────────
    "height":                ["Height"],
    "weight":                ["Weight"],
    "blood_pressure":        ["Blood Pressure", "B.P.", "BP"],
    "pulse":                 ["Pulse", "Heart Rate"],
    "vision_right":          ["Right", "Vision R", "OD"],
    "vision_left":           ["Left", "Vision L", "OS"],

    # ── System exam (Normal / Abnormal checkboxes) ─────────────────────
    "general":               ["General Appearance", "General App"],
    "skin":                  ["Skin"],
    "eyes":                  ["Eyes"],
    "ears":                  ["Ears"],
    "nose_throat":           ["Nose", "Throat", "ENT", "Nose/Throat"],
    "chest":                 ["Chest", "Lungs", "Chest/Lungs"],
    "heart":                 ["Heart", "Cardiovascular"],
    "abdomen":               ["Abdomen"],
    "genitalia":             ["Genitalia", "Rectal", "Rectal/Genitalia"],
    "back":                  ["Back", "Spine", "Back/Spine"],
    "extremities":           ["Extremities"],
    "neurological":          ["Neurological", "Neuro"],

    # ── Labs ──────────────────────────────────────────────────────────────
    "hemoglobin":            ["Hemoglobin", "Hgb", "Hematocrit"],
    "igra_result":           ["IGRA", "Interferon"],
    "chest_xray":            ["Chest X-Ray", "Chest Xray", "X-Ray"],
    "urinalysis_blood":      ["Blood", "Dipstick Blood"],
    "urinalysis_protein":    ["Protein"],
    "urinalysis_sugar":      ["Sugar", "Glucose"],
    "tb_exposure_risk":      ["TB Exposure", "Tuberculosis Exposure"],

    # ── Yes / No ──────────────────────────────────────────────────────────
    "chronic_condition":     ["Chronic Condition"],
    "driving_medication":    ["Driving", "Medication"],
    "functional_level":      ["Functional Level", "Fitness Level"],

    # ── Immunizations ─────────────────────────────────────────────────────
    "tdap":                  ["TDAP", "Tetanus"],
    "mmr1":                  ["MMR 1", "MMR-1", "MMR1"],
    "mmr2":                  ["MMR 2", "MMR-2", "MMR2"],
    "polio":                 ["Polio"],
    "hep_a1":                ["Hep A 1", "Hepatitis A 1"],
    "hep_b1":                ["Hep B 1", "Hepatitis B 1"],
    "influenza":             ["Influenza", "Flu"],

    # ── Dental evaluation ─────────────────────────────────────────────────
    "orthodontic_treatment": ["Orthodontic", "Braces"],
    "wisdom_teeth_removed":  ["Wisdom Teeth"],
    "brushing_frequency":    ["Brushing", "Brush"],
    "flossing_frequency":    ["Flossing", "Floss"],
    "pain_or_bleeding":      ["Pain", "Bleeding"],
    "oral_exam_with_bitewing": ["Oral Exam", "Bitewing"],
    "panoramic_done":        ["Panoramic", "Full X-rays"],
    "third_molars_assessed": ["Third Molars", "Molars"],
    "active_orthodontic":    ["Active Orthodontic", "Current Treatment"],
    "dental_decay":          ["Decay", "Caries"],
    "bonded_retainers":      ["Retainer", "Restoration"],
    "oral_hygiene_acceptable": ["Oral Hygiene", "Hygiene"],
}

# Single-field targeted prompts for Ollama
FIELD_PROMPTS: dict[str, str] = {
    "physician_name":  "What is the physician's or doctor's full name shown on this form? Return only the name.",
    "physician_city":  "What city is listed for the physician or doctor? Return only the city name.",
    "physician_state": "What state or province is listed for the physician? Return only the state/province abbreviation.",
    "physician_phone": "What is the physician's phone or telephone number? Return only the number.",
    "exam_date":       "What is the date of the medical examination shown on this form? Return MM/DD/YYYY.",
    "height":          "What is the patient's height? Return value and unit (e.g. '65 in' or '165 cm').",
    "weight":          "What is the patient's weight? Return value and unit (e.g. '130 lbs' or '59 kg').",
    "blood_pressure":  "What is the blood pressure reading? Return as systolic/diastolic (e.g. '120/80').",
    "pulse":           "What is the pulse or heart rate? Return just the number.",
    "vision_right":    "What is the right eye vision? Return as 20/XX format.",
    "vision_left":     "What is the left eye vision? Return as 20/XX format.",
    "functional_level":"What functional level is indicated (A, B, C, D, or E)? Return just the letter.",
    "chronic_condition":"Is a chronic condition checked YES or NO? Return 'Yes' or 'No'.",
    "driving_medication":"Is driving or medication restriction checked YES or NO? Return 'Yes' or 'No'.",
    "tb_exposure_risk": "Is TB exposure risk checked YES or NO? Return 'Yes' or 'No'.",
    "igra_result":     "What is the IGRA result? Return 'Negative', 'Positive', or 'Not Done'.",
    "chest_xray":      "What is the chest x-ray result? Return 'Normal', 'Abnormal', or 'Not Done'.",
    "hemoglobin":      "What is the hemoglobin or hematocrit value? Return just the number.",
    "urinalysis_blood":  "What is the urinalysis dipstick blood result? Return the value (e.g. 'Neg', 'Trace', number).",
    "urinalysis_protein":"What is the urinalysis dipstick protein result? Return the value.",
    "urinalysis_sugar":  "What is the urinalysis dipstick sugar or glucose result? Return the value.",
    "tdap":            "What date was the TDAP or tetanus vaccination? Return MM/DD/YYYY or 'Not Done'.",
    "mmr1":            "What date was the first MMR vaccination? Return MM/DD/YYYY or 'Not Done'.",
    "mmr2":            "What date was the second MMR vaccination? Return MM/DD/YYYY or 'Not Done'.",
    "polio":           "What date was the polio vaccination? Return MM/DD/YYYY or 'Not Done'.",
    "hep_a1":          "What date was the first Hepatitis A vaccination? Return MM/DD/YYYY or 'Not Done'.",
    "hep_b1":          "What date was the first Hepatitis B vaccination? Return MM/DD/YYYY or 'Not Done'.",
    "influenza":       "What date was the influenza/flu vaccination? Return MM/DD/YYYY or 'Not Done'.",
    "dentist_name":    "What is the dentist's full name? Return only the name.",
    "dentist_city":    "What city is listed for the dentist? Return only the city name.",
    "dentist_phone":   "What is the dentist's phone number? Return only the number.",
    "orthodontic_treatment": "Is orthodontic treatment indicated YES or NO? Return 'Yes', 'No', or 'Not Applicable'.",
    "wisdom_teeth_removed":  "Are wisdom teeth removed YES or NO? Return 'Yes', 'No', or 'Not Applicable'.",
    "brushing_frequency":    "How often does the patient brush teeth? Return frequency as written (e.g. 'Twice a day').",
    "flossing_frequency":    "How often does the patient floss? Return frequency as written.",
    "pain_or_bleeding":      "Is pain or bleeding checked YES or NO? Return 'Yes' or 'No'.",
    "oral_exam_with_bitewing":"Is oral exam with bitewing x-ray YES or NO? Return 'Yes', 'No', or 'Not Applicable'.",
    "panoramic_done":         "Is panoramic x-ray done YES or NO? Return 'Yes', 'No', or 'Not Applicable'.",
    "third_molars_assessed":  "Are third molars assessed YES or NO? Return 'Yes', 'No', or 'Not Applicable'.",
    "active_orthodontic":     "Is active orthodontic treatment YES or NO? Return 'Yes', 'No', or 'Not Applicable'.",
    "dental_decay":           "Is dental decay present YES or NO? Return 'Yes' or 'No'.",
    "bonded_retainers":       "Are bonded retainers present YES or NO? Return 'Yes', 'No', or 'Not Applicable'.",
    "oral_hygiene_acceptable":"Is oral hygiene acceptable YES or NO? Return 'Yes' or 'No'.",
}

# System-exam fields share the same structure (Normal/Abnormal radio buttons)
_EXAM_FIELDS = {"general", "skin", "eyes", "ears", "nose_throat", "chest",
                "heart", "abdomen", "genitalia", "back", "extremities", "neurological"}
for _f in _EXAM_FIELDS:
    FIELD_PROMPTS[_f] = f"For the '{_f.replace('_', ' ')}' physical exam finding, is it Normal or Abnormal? Return 'Normal' or 'Abnormal'."


# ---------------------------------------------------------------------------
# Page classification
# ---------------------------------------------------------------------------

def classify_pages(pages: list[dict]) -> dict[str, list[int]]:
    """
    Classify each page by the form sections it likely contains.
    Returns {section_name: [0-based page indices]} sorted by keyword match count
    (most relevant pages first).
    """
    section_pages: dict[str, list[int]] = {sec: [] for sec in PAGE_SECTION_KEYWORDS}
    for i, page in enumerate(pages):
        text = page.get("text", "").lower()
        for section, keywords in PAGE_SECTION_KEYWORDS.items():
            hits = sum(1 for kw in keywords if kw in text)
            if hits > 0:
                section_pages[section].append(i)
    return section_pages


def _score_pages(pages: list[dict], keywords: list[str]) -> list[tuple[int, int]]:
    """Return [(page_idx, hit_count)] sorted by hit_count descending."""
    scored = []
    for i, page in enumerate(pages):
        text = page.get("text", "").lower()
        hits = sum(1 for kw in keywords if kw in text)
        if hits > 0:
            scored.append((i, hits))
    return sorted(scored, key=lambda x: -x[1])


def medical_page_indices(pages: list[dict], max_pages: int = 6) -> list[int]:
    """Return 0-based indices of up to max_pages most relevant for medical content.

    Selects the highest-scoring pages across all medical sections, capped at
    max_pages to limit Ollama inference time.
    """
    medical_sections = ["physician_info", "vitals", "system_exam", "labs", "immunizations"]
    all_keywords = [kw for sec in medical_sections for kw in PAGE_SECTION_KEYWORDS[sec]]

    scored = _score_pages(pages, all_keywords)
    if not scored:
        return list(range(len(pages)))  # fallback: send all

    # Take up to max_pages, sorted by page number for readability
    top_idx = [idx for idx, _ in scored[:max_pages]]
    return sorted(top_idx)


def dental_page_indices(pages: list[dict], max_pages: int = 3) -> list[int]:
    """Return 0-based indices of up to max_pages most relevant for dental content."""
    dental_keywords = PAGE_SECTION_KEYWORDS["dental"]
    scored = _score_pages(pages, dental_keywords)
    if not scored:
        return list(range(len(pages)))  # fallback

    top_idx = [idx for idx, _ in scored[:max_pages]]
    return sorted(top_idx)


# ---------------------------------------------------------------------------
# Bounding-box region finding & cropping
# ---------------------------------------------------------------------------

def find_label_region(
    observations: list[dict],
    label_candidates: list[str],
    img_width: int,
    img_height: int,
    right_extend: int = 250,
    below_extend: int = 60,
) -> Optional[tuple[int, int, int, int]]:
    """
    Search OCR observations for a label matching any of label_candidates.
    Returns a crop box (left, top, right, bottom) in pixel coords that
    covers the label plus surrounding area where the value likely appears.
    Returns None if no matching label is found.

    right_extend: extra pixels to include to the right (for inline values)
    below_extend: extra pixels to include below (for values on next line)
    """
    for label in label_candidates:
        label_lower = label.lower()
        for obs in observations:
            if label_lower in obs["text"].lower():
                x = int(obs["x"])
                y = int(obs["y"])
                w = int(obs["w"])
                h = int(obs["h"])
                left   = max(0, x - 10)
                top    = max(0, y - 5)
                right  = min(img_width,  x + w + right_extend)
                bottom = min(img_height, y + h + below_extend)
                return (left, top, right, bottom)
    return None


def crop_for_field(
    page_img: Image.Image,
    observations: list[dict],
    field_name: str,
) -> Optional[Image.Image]:
    """
    Crop page_img to the region containing the given field's label + value area.
    Returns None if the label cannot be located in the OCR observations.
    """
    labels = FIELD_LABELS.get(field_name)
    if not labels:
        return None
    img_w, img_h = page_img.size
    box = find_label_region(observations, labels, img_w, img_h)
    if box is None:
        return None
    return page_img.crop(box)


def best_page_for_field(
    pages: list[dict],
    field_name: str,
) -> Optional[tuple[int, dict]]:
    """
    Return the (index, page) most likely to contain the given field based on
    OCR observation text matching.  Returns None if no page contains the label.
    """
    labels = FIELD_LABELS.get(field_name, [])
    if not labels:
        return None
    for label in labels:
        label_lower = label.lower()
        for i, page in enumerate(pages):
            for obs in page.get("ocr_observations", []):
                if label_lower in obs["text"].lower():
                    return (i, page)
    return None
