"""
Interactive verification and correction system for extracted form data.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table

from .config import load_config, VERIFIED_DIR
from .models import (
    MedicalForm, DentalForm, ProcessingResult,
    PersonalInfo, PhysicianInfo, VitalSigns,
    SystemExam, LabResults, Immunizations
)

console = Console()

MEDICAL_FIELD_ORDER = [
    ("personal_info", "Personal Information"),
    ("physician_info", "Physician Information"),
    ("functional_level", "Functional Level"),
    ("chronic_condition", "Chronic Condition"),
    ("chronic_comments", "Chronic Comments"),
    ("driving_medication", "Driving Medication"),
    ("driving_medication_explain", "Driving Medication Explanation"),
    ("vital_signs", "Vital Signs"),
    ("system_exam", "System Exam"),
    ("labs", "Lab Results"),
    ("immunizations", "Immunizations"),
    ("exam_date", "Exam Date"),
]

DENTAL_FIELD_ORDER = [
    ("personal_info", "Personal Information"),
    ("orthodontic_treatment", "Orthodontic Treatment"),
    ("orthodontic_complete", "Orthodontic Complete"),
    ("wisdom_teeth_removed", "Wisdom Teeth Removed"),
    ("brushing_frequency", "Brushing Frequency"),
    ("flossing_frequency", "Flossing Frequency"),
    ("pain_or_bleeding", "Pain or Bleeding"),
    ("bleeding_comments", "Bleeding Comments"),
    ("dentist_name", "Dentist Name"),
    ("dentist_address", "Dentist Address"),
    ("dentist_city", "Dentist City"),
    ("dentist_state", "Dentist State"),
    ("dentist_phone", "Dentist Phone"),
    ("dentist_email", "Dentist Email"),
    ("exam_date", "Exam Date"),
    ("oral_exam_with_bitewing", "Oral Exam with Bitewing"),
    ("third_molars_assessed", "Third Molars Assessed"),
    ("panoramic_done", "Panoramic Done"),
    ("active_orthodontic", "Active Orthodontic"),
    ("dental_decay", "Dental Decay"),
    ("bonded_retainers", "Bonded Retainers"),
    ("oral_hygiene_acceptable", "Oral Hygiene Acceptable"),
    ("dental_comments", "Dental Comments"),
]

NESTED_FIELDS = {
    "personal_info": ["first_name", "middle_name", "last_name", "suffix", "record_number", "date_of_birth", "age", "gender"],
    "physician_info": ["name", "credentials", "address", "city", "state", "postal_code", "phone", "email"],
    "vital_signs": ["height", "weight", "blood_pressure", "pulse", "vision_right", "vision_left", "vision_corrected"],
    "system_exam": ["general", "skin", "eyes", "ears", "nose_throat", "chest", "heart", "abdomen", "genitalia", "back", "extremities", "neurological", "breast_pelvic"],
    "labs": ["urinalysis_blood", "urinalysis_protein", "urinalysis_sugar", "hemoglobin", "tb_exposure_risk", "ppd_result", "igra_result", "chest_xray"],
    "immunizations": ["tdap", "tdap2", "mmr1", "mmr2", "polio", "hep_a1", "hep_a2", "hep_b1", "hep_b2", "hep_b3", "influenza", "covid_doses", "covid_brand", "meningococcal", "varicella", "other"],
}


def get_field_value(obj: Any, path: str) -> Any:
    parts = path.split(".")
    for part in parts:
        if obj is None:
            return None
        obj = getattr(obj, part, None)
    return obj


def set_field_value(obj: Any, path: str, value: Any) -> None:
    parts = path.split(".")
    for part in parts[:-1]:
        if obj is None:
            return
        obj = getattr(obj, part, None)
    if obj is not None:
        setattr(obj, parts[-1], value)


def format_value(value: Any) -> str:
    if value is None:
        return "[dim](empty)[/dim]"
    if isinstance(value, list):
        return ", ".join(str(v) for v in value) if value else "[dim](empty list)[/dim]"
    return str(value)
def verify_form(form: Any, field_order: list, form_name: str) -> dict:
    if form is None:
        console.print(f"[yellow]No {form_name} form data to verify.[/yellow]")
        return {}
    
    corrections = {}
    
    console.rule(f"[bold cyan]{form_name} Form Verification[/bold cyan]")
    console.print("[dim]For each field: Enter = keep, type new value, 'skip' = leave empty, 'done' = finish early[/dim]\n")
    
    for field_path, display_name in field_order:
        value = get_field_value(form, field_path)
        display_val = format_value(value)
        
        panel = Panel(
            f"[bold]{display_name}[/bold]\nCurrent: {display_val}",
            title=f"Field: {field_path}",
            border_style="cyan"
        )
        console.print(panel)
        
        prompt_text = f"  [bold]New value[/bold] (Enter=keep, 'skip'=empty, 'done'=finish): "
        user_input = Prompt.ask(prompt_text, default="", show_default=False)
        
        if user_input.lower() == "done":
            break
        elif user_input.lower() == "skip":
            corrections[field_path] = None
            console.print("  [yellow]Set to empty[/yellow]")
        elif user_input.strip():
            corrections[field_path] = user_input.strip()
            console.print(f"  [green]Updated to: {user_input.strip()}[/green]")
        else:
            pass
        
        console.print()
    
    return corrections


def apply_corrections(form: Any, corrections: dict) -> Any:
    for field_path, value in corrections.items():
        set_field_value(form, field_path, value)
    return form


def save_verified_result(
    result, corrections: dict, verified_dir: Path
) -> Path:
    verified_dir.mkdir(parents=True, exist_ok=True)
    
    safe_name = result.missionary_name.replace(", ", "_").replace(" ", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{safe_name}_{timestamp}.json"
    filepath = verified_dir / filename
    
    data = {
        "missionary_name": result.missionary_name,
        "pdf_path": result.pdf_path,
        "has_medical": result.has_medical,
        "has_dental": result.has_dental,
        "verified_at": datetime.now().isoformat(),
        "original_errors": result.errors,
        "original_warnings": result.warnings,
        "corrections": corrections,
    }
    
    if result.medical:
        data["medical"] = result.medical.model_dump()
    if result.dental:
        data["dental"] = result.dental.model_dump()
    
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    
    return filepath
def run_verification_cli(result, verified_dir: Path = None):
    from .config import load_config
    if verified_dir is None:
        cfg = load_config()
        verified_dir = Path(cfg.get("verified_dir", "verified"))
    
    all_corrections = {}
    
    if result.has_medical and result.medical:
        med_corrections = verify_form(result.medical, MEDICAL_FIELD_ORDER, "Medical")
        all_corrections.update({f"medical.{k}": v for k, v in med_corrections.items()})
        
        if med_corrections:
            apply_corrections(result.medical, med_corrections)
    
    if result.has_dental and result.dental:
        dent_corrections = verify_form(result.dental, DENTAL_FIELD_ORDER, "Dental")
        all_corrections.update({f"dental.{k}": v for k, v in dent_corrections.items()})
        
        if dent_corrections:
            apply_corrections(result.dental, dent_corrections)
    
    if all_corrections:
        filepath = save_verified_result(result, all_corrections, verified_dir)
        console.print(f"\n[green]Verified result saved to: {filepath}[/green]")
    else:
        console.print("\n[yellow]No corrections made.[/yellow]")
    
    return result
@click.command()
@click.argument("pdf_path", type=click.Path(exists=True, path_type=Path))
@click.option("--verified-dir", type=click.Path(path_type=Path), help="Directory for verified results")
@click.option("--auto-submit/--no-auto-submit", default=False, help="Auto-submit after verification")
def verify_cmd(pdf_path, verified_dir, auto_submit):
    from app.config import load_config
    from app.form_parser import parse_forms
    from app.pdf_processor import (
        combine_pages,
        detect_form_types,
        extract_pdf_text,
        parse_missionary_name_from_filename,
    )
    from app.web_automator import run_for_result
    
    cfg = load_config()
    if verified_dir is None:
        verified_dir = Path(cfg.get("verified_dir", "verified"))
    
    console.rule(f"[bold cyan]Verification: {pdf_path.name}[/bold cyan]")
    
    from app.pdf_processor import (
        combine_pages,
        detect_form_types,
        extract_pdf_text,
        parse_missionary_name_from_filename,
    )
    from app.form_parser import parse_forms
    from app.web_automator import run_for_result
    
    pages = extract_pdf_text(pdf_path)
    full_text = combine_pages(pages)
    missionary_name = parse_missionary_name_from_filename(pdf_path.name)
    has_medical, has_dental = detect_form_types(pdf_path.name, full_text)
    
    console.print(f"[bold]Missionary:[/bold] {missionary_name}")
    console.print(f"[bold]Forms:[/bold] {'Medical' if has_medical else ''} {'Dental' if has_dental else ''}")
    
    result = parse_forms(
        text=full_text,
        missionary_name=missionary_name,
        pdf_path=str(pdf_path),
        has_medical=has_medical,
        has_dental=has_dental,
    )
    
    result = run_verification_cli(result, verified_dir)
    
    if auto_submit and not Confirm.ask("\nSubmit to website now?"):
        auto_submit = False
    
    if auto_submit:
        from app.web_automator import run_for_result
        run_for_result(result, dry_run=False)
    else:
        console.print("\n[dim]Run 'python main.py process' with the PDF to submit after verification.[/dim]")


if __name__ == "__main__":
    verify_cmd()
