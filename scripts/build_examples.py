#!/usr/bin/env python3
"""
Build few-shot example images for the Ollama VLM parser.

Crops key sections from processed missionary PDFs and saves them with
correct-value labels to logs/examples/. These are loaded by ollama_parser.py
and injected as visual context on every extraction call.

Usage:
  # Build examples from a single PDF (missionary name must match ground_truth.json)
  python scripts/build_examples.py --pdf "processed/Alexander Lloyd.pdf" --missionary "Lloyd, Alexander Thomas"

  # Auto-build from all processed PDFs that have ground truth
  python scripts/build_examples.py --all

  # Preview mode: save crops without needing ground truth (for layout inspection)
  python scripts/build_examples.py --pdf "processed/Alexander Lloyd.pdf" --preview
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image
from app.vision_shared import _image_to_base64, render_pages
from app.config import BASE_DIR, LOGS_DIR
from app.field_extractor import dental_page_indices, medical_page_indices
from app.pdf_processor import combine_pages, extract_pdf_text

EXAMPLES_DIR = LOGS_DIR / "examples"
GROUND_TRUTH_PATH = LOGS_DIR / "ground_truth.json"
PROCESSED_DIR = BASE_DIR / "processed"

# ---------------------------------------------------------------------------
# Crop specifications
# ---------------------------------------------------------------------------
# Each spec: page_idx is the index WITHIN the classified subset (e.g., index 0
# means the first medical page, not necessarily raw page 0 of the PDF).
# crop = (left_frac, top_frac, right_frac, bottom_frac) as fractions of page dims.

MEDICAL_CROPS = [
    {
        "key": "med_functional_chronic",
        "page_idx": 0,
        "crop": (0.0, 0.0, 1.0, 0.45),
        "fields": ["functional_level", "chronic_condition", "driving_medication"],
        "description": (
            "MEDICAL FORM — page 1 top section. Contains the Functional Level A-E "
            "radio buttons (A = No Limitations, E = Severe Limitations) and the "
            "Chronic Condition Yes/No. A FILLED dark circle or checkmark next to an "
            "option means it is SELECTED. An empty circle means NOT selected."
        ),
    },
    {
        "key": "med_vitals",
        "page_idx": 0,
        "crop": (0.40, 0.35, 1.0, 0.82),
        "fields": ["height", "weight", "blood_pressure", "pulse", "vision_right", "vision_left"],
        "description": (
            "MEDICAL FORM — Vital Signs section. Shows height (include unit: in or cm), "
            "weight (include unit: lbs or kg), blood pressure (systolic/diastolic), "
            "pulse rate, and vision acuity right/left. Values are handwritten or typed "
            "next to each label."
        ),
    },
    {
        "key": "med_system_exam",
        "page_idx": 1,
        "crop": (0.0, 0.02, 1.0, 0.65),
        "fields": [
            "general", "skin", "eyes", "ears", "nose_throat", "chest",
            "heart", "abdomen", "genitalia", "back", "extremities", "neurological",
        ],
        "description": (
            "MEDICAL FORM — Physical Examination table. Each row (General, Skin, Eyes, "
            "Ears, Nose/Throat, Chest, Heart, Abdomen, Genitalia, Back, Extremities, "
            "Neurological) has a Normal circle and an Abnormal circle. A FILLED circle "
            "with a dark mark means that option is SELECTED. MOST missionaries have ALL "
            "systems marked Normal. Read EVERY row — do not default to Abnormal."
        ),
    },
    {
        "key": "med_labs",
        "page_idx": 1,
        "crop": (0.0, 0.60, 1.0, 1.0),
        "fields": [
            "tb_exposure_risk", "igra_result", "chest_xray",
            "hemoglobin", "urinalysis_blood", "urinalysis_protein", "urinalysis_sugar",
        ],
        "description": (
            "MEDICAL FORM — Laboratory Results. Shows TB Exposure Risk (Yes/No), "
            "IGRA/QuantiFERON result (Negative/Positive/Not Done), Chest X-Ray result "
            "(Normal/Abnormal/Not Done), Hemoglobin value, and Urinalysis results. "
            "The majority of missionaries have TB Exposure=No, IGRA=Negative, "
            "ChestXray=Not Done."
        ),
    },
    {
        "key": "med_immunizations",
        "page_idx": 2,
        "crop": (0.0, 0.0, 1.0, 0.75),
        "fields": ["tdap", "mmr1", "mmr2", "polio", "influenza"],
        "description": (
            "MEDICAL FORM — Immunization record table. Each row shows a vaccine name "
            "(TDAP, MMR-1, MMR-2, Polio, Hepatitis A/B, Influenza) and the date when "
            "given (MM/DD/YYYY). An empty or blank row means no dose was recorded — "
            "return null, not a guess."
        ),
    },
]

DENTAL_CROPS = [
    {
        "key": "dnt_history",
        "page_idx": 0,
        "crop": (0.0, 0.18, 1.0, 0.47),
        "fields": [
            "orthodontic_treatment", "wisdom_teeth_removed",
            "brushing_frequency", "flossing_frequency", "pain_or_bleeding",
        ],
        "description": (
            "DENTAL FORM — Patient History section (filled out by the candidate). "
            "Yes/No checkboxes for orthodontic treatment, wisdom teeth removed, and "
            "pain/bleeding. Brushing and flossing frequency are written as free text. "
            "'Yes' is very common for orthodontic_treatment and wisdom_teeth_removed. "
            "A circle with a dark mark or checkmark = selected. Empty circle = NOT selected."
        ),
    },
    {
        "key": "dnt_exam_findings",
        "page_idx": 0,
        "crop": (0.0, 0.52, 1.0, 0.87),
        "fields": [
            "oral_exam_with_bitewing", "third_molars_assessed", "panoramic_done",
            "active_orthodontic", "dental_decay", "bonded_retainers", "oral_hygiene_acceptable",
        ],
        "description": (
            "DENTAL FORM — Dental Examination Findings (filled out by the dentist). "
            "Each question has Yes, No, and sometimes Not Applicable circles. "
            "A FILLED circle = selected. CRITICAL: "
            "dental_decay='Yes' means decay has been RESOLVED (treated), a good outcome. "
            "active_orthodontic='Yes' means orthodontic treatment is COMPLETE. "
            "third_molars_assessed, panoramic_done, and active_orthodontic may be 'Not Applicable'."
        ),
    },
    {
        "key": "dnt_dentist_info",
        "page_idx": 0,
        "crop": (0.0, 0.82, 1.0, 1.0),
        "fields": ["dentist_name", "dentist_city", "dentist_state", "dentist_phone", "exam_date"],
        "description": (
            "DENTAL FORM — Dentist Information section (bottom of form). Shows the "
            "individual dentist's personal name (not the practice name), office address, "
            "city, state, postal code, phone number, email, and the date of the exam."
        ),
    },
]

# Map crop key → relevant ground truth fields
_SECTION_FIELDS = {c["key"]: c["fields"] for c in MEDICAL_CROPS + DENTAL_CROPS}


# ---------------------------------------------------------------------------
# Core crop function
# ---------------------------------------------------------------------------

def _crop_page(img: Image.Image, crop_fracs: tuple) -> Image.Image:
    """Crop an image using fractional coordinates (l, t, r, b)."""
    w, h = img.size
    lf, tf, rf, bf = crop_fracs
    box = (int(lf * w), int(tf * h), int(rf * w), int(bf * h))
    return img.crop(box)


def _add_border(img: Image.Image, color: tuple, width: int = 3) -> Image.Image:
    """Add a thin colored border to an image."""
    bordered = Image.new("RGB", (img.width + 2 * width, img.height + 2 * width), color)
    bordered.paste(img.convert("RGB"), (width, width))
    return bordered


def _resize(img: Image.Image, max_dim: int) -> Image.Image:
    w, h = img.size
    if max(w, h) <= max_dim:
        return img
    scale = max_dim / max(w, h)
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_examples_for_pdf(
    pdf_path: Path,
    missionary_name: str,
    ground_truth: dict,
    output_dir: Path,
    max_dim: int = 600,
    preview: bool = False,
) -> dict:
    """
    Render a PDF, apply crop specs, and save example images.

    Returns a manifest dict: {"medical": [...entries], "dental": [...entries]}
    """
    print(f"\n  Rendering {pdf_path.name} …")

    # 1. OCR to get page text for classification
    pages_ocr = extract_pdf_text(pdf_path)
    full_text = combine_pages(pages_ocr)

    # 2. Render pages as PIL images at 200 DPI for good crop quality
    pages_rendered = render_pages(str(pdf_path), dpi=200)
    images = {p["page"]: p["image"] for p in pages_rendered}

    # 3. Classify pages into medical / dental
    med_indices = medical_page_indices(pages_ocr)
    dnt_indices = dental_page_indices(pages_ocr)
    print(f"  Medical page indices: {med_indices}")
    print(f"  Dental page indices:  {dnt_indices}")

    # 4. Get ground truth for this missionary (None if preview mode)
    gt_record = ground_truth.get(missionary_name, {}) if not preview else {}

    manifest: dict = {"medical": [], "dental": []}

    def _process_crops(crop_specs, page_indices, form_type, border_color):
        for spec in crop_specs:
            pi = spec["page_idx"]
            if pi >= len(page_indices):
                print(f"  [{form_type}] skip '{spec['key']}' — page index {pi} out of range (have {len(page_indices)} pages)")
                continue

            raw_page = page_indices[pi]   # 0-based raw PDF page number
            img = images.get(raw_page + 1)  # render_pages uses 1-based page numbers
            if img is None:
                print(f"  [{form_type}] skip '{spec['key']}' — raw page {raw_page + 1} not rendered")
                continue

            # Crop, resize, add border
            cropped = _crop_page(img, spec["crop"])
            cropped = _resize(cropped, max_dim)
            cropped = _add_border(cropped, border_color, width=3)

            # Save
            safe_name = missionary_name.replace(",", "").replace(" ", "")
            fname = f"{spec['key']}_{safe_name}.png"
            subdir = output_dir / form_type
            subdir.mkdir(parents=True, exist_ok=True)
            save_path = subdir / fname
            cropped.save(save_path, format="PNG", optimize=True)
            print(f"  [{form_type}] saved '{spec['key']}' → {save_path.name} ({cropped.width}x{cropped.height})")

            # Build correct_values from ground truth
            gt_section = gt_record.get(form_type, {}) or {}
            correct_values = {
                f: gt_section[f]
                for f in spec["fields"]
                if f in gt_section and gt_section[f] is not None
            }

            manifest[form_type].append({
                "key": spec["key"],
                "image": f"{form_type}/{fname}",
                "missionary": missionary_name,
                "description": spec["description"],
                "correct_values": correct_values,
            })

    _process_crops(MEDICAL_CROPS, med_indices, "medical", (160, 30, 30))   # red border
    _process_crops(DENTAL_CROPS,  dnt_indices, "dental",  (30, 30, 160))   # blue border

    return manifest


# ---------------------------------------------------------------------------
# Name matching: PDF filename → ground truth key
# ---------------------------------------------------------------------------

def _match_gt_name(pdf_name: str, gt_keys: list[str]) -> str | None:
    """
    Try to match a PDF filename to a ground_truth.json key.
    Ground truth keys are "Last, First Middle" format.
    PDF filenames may be "First Last.pdf" or contain extra words.
    """
    from rapidfuzz import fuzz, process

    # Strip extension and noise words
    base = pdf_path_to_search_name(pdf_name)

    result = process.extractOne(
        base,
        gt_keys,
        scorer=fuzz.token_sort_ratio,
        score_cutoff=50,
    )
    return result[0] if result else None


def pdf_path_to_search_name(pdf_name: str) -> str:
    """Strip extension and common noise words from a PDF filename."""
    import re
    base = Path(pdf_name).stem
    # Remove common noise words from combined-form filenames
    noise = r"\b(missionary|medical|dental|forms?|evaluation|all|form|records?)\b"
    base = re.sub(noise, "", base, flags=re.IGNORECASE)
    # Remove " - Jake Merkley", " - Joshua G C" etc (the ward clerk's name)
    base = re.sub(r"\s*-\s*\w+\s+\w+(\s+\w+)?$", "", base)
    return base.strip()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build few-shot example crops for Ollama parser")
    parser.add_argument("--pdf", type=Path, help="Path to a single PDF")
    parser.add_argument("--missionary", help="Ground truth key e.g. 'Lloyd, Alexander Thomas'")
    parser.add_argument("--all", action="store_true", help="Build from all processed/ PDFs with ground truth")
    parser.add_argument("--preview", action="store_true", help="Save crops without ground truth labels (layout inspection)")
    parser.add_argument("--max-dim", type=int, default=600, help="Max image dimension in pixels (default 600)")
    parser.add_argument("--output", type=Path, default=EXAMPLES_DIR, help="Output directory")
    args = parser.parse_args()

    if not args.pdf and not args.all:
        parser.error("Provide --pdf or --all")

    # Load ground truth
    ground_truth: dict = {}
    if GROUND_TRUTH_PATH.exists():
        with open(GROUND_TRUTH_PATH) as f:
            ground_truth = json.load(f)
        print(f"Loaded ground truth: {len(ground_truth)} candidates")
    elif not args.preview:
        print(f"WARNING: No ground truth at {GROUND_TRUTH_PATH}. Crops will have no correct_values.")

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict = {"medical": [], "dental": []}
    processed_keys: set[str] = set()  # track which GT sections already have a crop

    def _build(pdf_path: Path, missionary_name: str):
        m = build_examples_for_pdf(
            pdf_path=pdf_path,
            missionary_name=missionary_name,
            ground_truth=ground_truth,
            output_dir=output_dir,
            max_dim=args.max_dim,
            preview=args.preview,
        )
        # Merge: only add crops for keys not yet present (first PDF wins)
        for form_type in ("medical", "dental"):
            for entry in m[form_type]:
                key = entry["key"]
                if key not in processed_keys:
                    manifest[form_type].append(entry)
                    processed_keys.add(key)
                    # If this crop still has no correct_values and we have
                    # ground truth data for this missionary, note it
                    if not entry["correct_values"] and missionary_name in ground_truth:
                        gt_sec = (ground_truth[missionary_name].get(form_type) or {})
                        filled = sum(1 for v in gt_sec.values() if v)
                        print(f"  [{form_type}] '{key}': {filled} GT fields available but 0 matched — check field names")

    if args.all:
        # Auto-match all processed/ PDFs to ground truth
        gt_keys = list(ground_truth.keys())
        pdf_files = sorted(PROCESSED_DIR.glob("*.pdf"))
        if not pdf_files:
            print(f"No PDFs found in {PROCESSED_DIR}")
            return

        for pdf_path in pdf_files:
            name = _match_gt_name(pdf_path.name, gt_keys)
            if not name:
                print(f"  SKIP {pdf_path.name} — no ground truth match found")
                continue
            print(f"  {pdf_path.name} → {name}")
            _build(pdf_path, name)

            # Once all 8 crop keys are present, we're done
            if len(processed_keys) >= len(MEDICAL_CROPS) + len(DENTAL_CROPS):
                print("\n  All crop keys collected — stopping early.")
                break
    else:
        missionary_name = args.missionary or ""
        if not missionary_name and not args.preview:
            # Try to auto-match
            gt_keys = list(ground_truth.keys())
            missionary_name = _match_gt_name(args.pdf.name, gt_keys) or ""
            if missionary_name:
                print(f"  Auto-matched: {args.pdf.name} → {missionary_name}")
            else:
                print("  WARNING: could not match PDF to ground truth. Use --missionary to specify.")
        _build(args.pdf, missionary_name)

    # Write manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    total = sum(len(v) for v in manifest.values())
    print(f"\nDone. {total} example crops saved.")
    print(f"Manifest: {manifest_path}")
    print(f"\nMedical examples ({len(manifest['medical'])}):")
    for e in manifest["medical"]:
        vals = list(e["correct_values"].items())[:3]
        print(f"  {e['key']}: {e['image']} | {vals}")
    print(f"\nDental examples ({len(manifest['dental'])}):")
    for e in manifest["dental"]:
        vals = list(e["correct_values"].items())[:3]
        print(f"  {e['key']}: {e['image']} | {vals}")


if __name__ == "__main__":
    main()
