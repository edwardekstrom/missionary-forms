#!/usr/bin/env python3
"""
Fast dry-run test: build a ProcessingResult directly from ground_truth.json
and run the browser automation (dry_run=True) without any Ollama/OCR step.

Usage (from an interactive terminal):
    cd /Users/eddy.ekstrom/missionary-forms
    source venv/bin/activate
    python3 _test_dry_run.py [candidate_key]

Default candidate: "Hanson, McKenna Elizabeth"
Pass a different key (exactly as it appears in ground_truth.json) as an argument.
"""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from app.models import (
    DentalForm,
    Immunizations,
    LabResults,
    MedicalForm,
    PersonalInfo,
    PhysicianInfo,
    ProcessingResult,
    SystemExam,
    VitalSigns,
)
from app.web_automator import run_for_result

GT_PATH = Path("logs/ground_truth.json")


def gt_to_processing_result(key: str, gt: dict) -> ProcessingResult:
    """
    Convert a ground_truth.json entry (Last, First Middle format) into a
    ProcessingResult suitable for the browser automation layer.
    """
    entry = gt[key]

    # Derive a display name "First Last" from "Last, First Middle"
    # Used by find_and_click_candidate to search the website.
    parts = key.split(",", 1)
    last = parts[0].strip()
    first_middle = parts[1].strip() if len(parts) > 1 else ""
    first = first_middle.split()[0] if first_middle else last
    missionary_name = f"{first} {last}"

    med_gt  = entry.get("medical") or {}
    dent_gt = entry.get("dental")  or {}

    # ── Medical ──────────────────────────────────────────────────────────────
    medical = MedicalForm(
        personal_info=PersonalInfo(),
        physician_info=PhysicianInfo(
            name        = med_gt.get("physician_name"),
            city        = med_gt.get("physician_city"),
            state       = med_gt.get("physician_state"),
            country     = med_gt.get("physician_country", "United States"),
            phone       = med_gt.get("physician_phone"),
        ),
        functional_level   = med_gt.get("functional_level"),
        chronic_condition  = med_gt.get("chronic_condition"),
        driving_medication = med_gt.get("driving_medication"),
        exam_date          = med_gt.get("exam_date"),
        vital_signs=VitalSigns(
            height          = med_gt.get("height"),
            weight          = med_gt.get("weight"),
            blood_pressure  = med_gt.get("blood_pressure"),
            pulse           = med_gt.get("pulse"),
            vision_right    = med_gt.get("vision_right"),
            vision_left     = med_gt.get("vision_left"),
        ),
        system_exam=SystemExam(
            general     = med_gt.get("general"),
            skin        = med_gt.get("skin"),
            eyes        = med_gt.get("eyes"),
            ears        = med_gt.get("ears"),
            nose_throat = med_gt.get("nose_throat"),
            chest       = med_gt.get("chest"),
            heart       = med_gt.get("heart"),
            abdomen     = med_gt.get("abdomen"),
            genitalia   = med_gt.get("genitalia"),
            back        = med_gt.get("back"),
            extremities = med_gt.get("extremities"),
            neurological= med_gt.get("neurological"),
            breast_pelvic = med_gt.get("breast_pelvic"),
        ),
        labs=LabResults(
            urinalysis_blood   = med_gt.get("urinalysis_blood"),
            urinalysis_protein = med_gt.get("urinalysis_protein"),
            urinalysis_sugar   = med_gt.get("urinalysis_sugar"),
            hemoglobin         = med_gt.get("hemoglobin"),
            tb_exposure_risk   = med_gt.get("tb_exposure_risk"),
            ppd_result         = med_gt.get("ppd_result"),
            igra_result        = med_gt.get("igra_result"),
            chest_xray         = med_gt.get("chest_xray"),
        ),
        immunizations=Immunizations(
            tdap        = med_gt.get("tdap"),
            mmr1        = med_gt.get("mmr1"),
            mmr2        = med_gt.get("mmr2"),
            polio       = med_gt.get("polio"),
            hep_a1      = med_gt.get("hep_a1"),
            hep_a2      = med_gt.get("hep_a2"),
            hep_b1      = med_gt.get("hep_b1"),
            hep_b2      = med_gt.get("hep_b2"),
            hep_b3      = med_gt.get("hep_b3"),
            influenza   = med_gt.get("influenza"),
            meningococcal = med_gt.get("meningococcal"),
            varicella   = med_gt.get("varicella"),
        ),
    )

    # ── Dental ───────────────────────────────────────────────────────────────
    dental = DentalForm(
        personal_info=PersonalInfo(),
        dentist_name         = dent_gt.get("dentist_name"),
        dentist_city         = dent_gt.get("dentist_city"),
        dentist_state        = dent_gt.get("dentist_state"),
        dentist_country      = dent_gt.get("dentist_country", "United States"),
        dentist_phone        = dent_gt.get("dentist_phone"),
        exam_date            = dent_gt.get("exam_date"),
        orthodontic_treatment= dent_gt.get("orthodontic_treatment"),
        orthodontic_complete = dent_gt.get("orthodontic_complete"),
        wisdom_teeth_removed = dent_gt.get("wisdom_teeth_removed"),
        brushing_frequency   = dent_gt.get("brushing_frequency"),
        flossing_frequency   = dent_gt.get("flossing_frequency"),
        pain_or_bleeding     = dent_gt.get("pain_or_bleeding"),
        oral_exam_with_bitewing = dent_gt.get("oral_exam_with_bitewing"),
        third_molars_assessed   = dent_gt.get("third_molars_assessed"),
        panoramic_done          = dent_gt.get("panoramic_done"),
        active_orthodontic      = dent_gt.get("active_orthodontic"),
        dental_decay            = dent_gt.get("dental_decay"),
        bonded_retainers        = dent_gt.get("bonded_retainers"),
        oral_hygiene_acceptable = dent_gt.get("oral_hygiene_acceptable"),
        dental_comments         = dent_gt.get("dental_comments"),
    )

    return ProcessingResult(
        missionary_name = missionary_name,
        pdf_path        = "(ground_truth dry-run)",
        has_medical     = bool(med_gt),
        has_dental      = bool(dent_gt),
        medical         = medical,
        dental          = dental,
    )


def main():
    gt = json.loads(GT_PATH.read_text())

    # Allow overriding the candidate from the command line
    if len(sys.argv) > 1:
        key = " ".join(sys.argv[1:])
    else:
        key = "Hanson, McKenna Elizabeth"

    if key not in gt:
        print(f"ERROR: '{key}' not found in ground_truth.json")
        print("Available candidates:")
        for k in gt:
            print(f"  {k!r}")
        sys.exit(1)

    result = gt_to_processing_result(key, gt)
    print(f"\nDry-run for: {result.missionary_name}  ({key})")
    print("Opening browser — forms will be FILLED but NOT submitted.\n")

    run_for_result(result, dry_run=True)


if __name__ == "__main__":
    main()
