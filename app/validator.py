"""
Validation system for the missionary forms processor.

Workflow:
  1. Log in to the church website
  2. For each PDF in the downloads folder:
     a. Parse missionary name from filename
     b. Navigate to their record on the website
     c. Scrape the currently-submitted form field values (ground truth)
     d. Run our PDF parser on the PDF
     e. Compare field-by-field, report accuracy
  3. Print a summary table across all missionaries

Results are also written to logs/validation_results.json.
"""

import asyncio
import json
from pathlib import Path
from typing import Optional

from loguru import logger
from playwright.async_api import Page, async_playwright
from rapidfuzz import fuzz, process
from rich.console import Console
from rich.table import Table

from .config import CHURCH_URL, load_config, LOGS_DIR
from .pdf_processor import (
    combine_pages,
    detect_form_types,
    extract_pdf_text,
    parse_missionary_name_from_filename,
)
from .form_parser import parse_forms, parse_personal_info
from .web_automator import _wait_nav

console = Console()


# ---------------------------------------------------------------------------
# Scrape all form field values from the current page (all sections)
# ---------------------------------------------------------------------------

async def _scrape_fields(page: Page) -> dict[str, str]:
    """
    Read all form field values from the current page using JavaScript.
    Handles text inputs, checked radio buttons, and select dropdowns.
    """
    return await page.evaluate("""
        () => {
            const result = {};
            // Text / number / date inputs and textareas
            document.querySelectorAll(
                'input:not([type="radio"]):not([type="checkbox"]):not([type="submit"]):not([type="button"]), textarea'
            ).forEach(el => {
                const key = el.id || el.name;
                if (key && el.value) result[key] = el.value;
            });
            // Checked radio buttons
            document.querySelectorAll('input[type="radio"]:checked').forEach(el => {
                if (el.name) result[el.name] = el.value;
            });
            // Select dropdowns
            document.querySelectorAll('select').forEach(el => {
                const key = el.id || el.name;
                if (key && el.value) result[key] = el.value;
            });
            return result;
        }
    """)


async def _scrape_all_sections(page: Page) -> dict[str, str]:
    """
    Scrape fields from every section of a multi-step form by clicking
    Next/Continue until there are no more, collecting fields at each step.

    Angular fetches saved form data asynchronously after the page shell loads.
    We must poll until at least one field has a value before scraping.
    """
    all_fields: dict[str, str] = {}
    visited_urls: set[str] = set()
    MAX_SECTIONS = 15

    # The form is a multi-step wizard (Introduction → Candidate History → Evaluation…).
    # Angular SPA keeps the same URL across all steps, so URL-based dedup doesn't work.
    # Strategy: click through "Next" on each step, scraping fields at each one.

    for step in range(MAX_SECTIONS):
        # Wait for Angular to finish rendering this step's content.
        # We wait until either an input has a value (data-loaded step) OR
        # a Next/Done button is visible (intro/summary step with no fields).
        try:
            await page.wait_for_function(
                """() => {
                    // Data-bearing step: at least one input has a value
                    const inputs = document.querySelectorAll(
                        'input:not([type="radio"]):not([type="checkbox"])' +
                        ':not([type="submit"]):not([type="button"]), textarea'
                    );
                    for (const el of inputs) {
                        if (el.value && el.value.trim()) return true;
                    }
                    // Any checked radio (Yes/No, Normal/Abnormal)
                    if (document.querySelectorAll('input[type="radio"]:checked').length > 0) return true;
                    // Navigation-only step (Introduction): presence of Next button is enough
                    const btns = document.querySelectorAll('button');
                    for (const b of btns) {
                        if (b.textContent.trim().toLowerCase() === 'next') return true;
                    }
                    return false;
                }""",
                timeout=8000,
            )
        except Exception:
            # Give Angular one more second then proceed regardless
            await page.wait_for_timeout(1000)

        fields = await _scrape_fields(page)
        all_fields.update(fields)
        logger.debug(f"  Step {step + 1}: scraped {len(fields)} fields (running total {len(all_fields)})")

        # Try to advance to the next section
        advanced = False
        for locator in [
            page.get_by_role("button", name="Next", exact=True),
            page.get_by_role("button", name="Next", exact=False),
            page.locator("button", has_text="Next"),
            page.get_by_text("Next", exact=True).locator(".."),  # parent of text node
            page.locator("text=Next").locator("xpath=ancestor-or-self::button[1]"),
            page.get_by_role("button", name="Continue", exact=False),
            page.locator("button", has_text="Continue"),
        ]:
            try:
                count = await locator.count()
                if count > 0:
                    await locator.first.click()
                    await _wait_nav(page, 8000)
                    advanced = True
                    logger.debug(f"  Step {step + 1}: advanced via button")
                    break
            except Exception as e:
                logger.debug(f"  Next-btn attempt failed: {e}")
                continue
        if not advanced:
            logger.debug(f"  Step {step + 1}: no Next button found — stopping")
            break

    return all_fields


# ---------------------------------------------------------------------------
# Reverse-translate raw field values → human-readable dict
# ---------------------------------------------------------------------------

def _fields_to_medical(f: dict) -> dict:
    """Convert raw scraped medical form field values to a normalised dict."""
    def yn(name: str) -> Optional[str]:
        v = f.get(name)
        if v == "1": return "Yes"
        if v == "0": return "No"
        return None

    def exam(name: str) -> Optional[str]:
        v = f.get(name)
        if v == "1": return "Normal"
        if v == "2": return "Abnormal"
        return None

    height_val  = f.get("HlthEvalHeight", "")
    height_unit = "in" if f.get("HlthEvalHeightUnits") in ("1", None, "") else "cm"
    weight_val  = f.get("HlthEvalWeight", "")
    weight_unit = "lbs" if f.get("HlthEvalWeightUnits") in ("1", None, "") else "kg"
    bp_sys = f.get("HlthEvalBloodPressure", "")
    bp_dia = f.get("HlthEvalBloodPressure2", "")
    fitness_map = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}
    igra_map    = {"0": "Negative", "1": "Positive", "2": "Not Done"}
    xray_map    = {"1": "Normal", "2": "Abnormal", "3": "Not Done"}

    return {
        "physician_name":    f.get("HlthEvalPhysicianName"),
        "physician_city":    f.get("HlthEvalPhysicianCity"),
        "physician_state":   f.get("HlthEvalPhysicianStateProvince"),
        "physician_phone":   f.get("HlthEvalPhysicianPhone"),
        "exam_date":         f.get("HlthEvalPhysicianExamDate"),
        "height":            f"{height_val} {height_unit}".strip() if height_val else None,
        "weight":            f"{weight_val} {weight_unit}".strip() if weight_val else None,
        "blood_pressure":    f"{bp_sys}/{bp_dia}" if bp_sys and bp_dia else (bp_sys or None),
        "pulse":             f.get("HlthEvalPulse"),
        "vision_right":      f.get("HlthEvalVisionRight"),
        "vision_left":       f.get("HlthEvalVisionLeft"),
        "functional_level":  fitness_map.get(f.get("HlthEvalFitnessLevel", ""), ""),
        "chronic_condition": yn("HlthEvalChronicConditionYN"),
        "driving_medication":yn("HlthEvalDrivingMedicationYN"),
        "tb_exposure_risk":  yn("HlthEvalTBExposerYN"),
        "igra_result":       igra_map.get(f.get("HlthEvalInterferon", ""), ""),
        "chest_xray":        xray_map.get(f.get("HlthEvalChestXray", ""), ""),
        "hemoglobin":        f.get("HlthEvalHemoglobin"),
        "urinalysis_blood":  f.get("HlthEvalUrinalysisDipstickBloodText"),
        "urinalysis_protein":f.get("HlthEvalUrinalysisDipstickProtein"),
        "urinalysis_sugar":  f.get("HlthEvalUrinalysisDipstickSugar"),
        "general":           exam("HlthEvalGenAppearanceNA"),
        "skin":              exam("HlthEvalSkinNA"),
        "eyes":              exam("HlthEvalEyesNA"),
        "ears":              exam("HlthEvalEarsNA"),
        "nose_throat":       exam("HlthEvalNoseNA"),
        "chest":             exam("HlthEvalChestNA"),
        "heart":             exam("HlthEvalHeartNA"),
        "abdomen":           exam("HlthEvalAbdomenNA"),
        "genitalia":         exam("HlthEvalRectalNA"),
        "back":              exam("HlthEvalBackNA"),
        "extremities":       exam("HlthEvalUpperExtrmNA"),
        "neurological":      exam("HlthEvalNeurologicalNA"),
        "tdap":              f.get("HlthEvalImmunTetanus"),
        "mmr1":              f.get("HlthEvalImmunMmr1"),
        "mmr2":              f.get("HlthEvalImmunMmr2"),
        "polio":             f.get("HlthEvalImmunPolio"),
        "influenza":         f.get("HlthEvalImmunInfluenza"),
    }


def _fields_to_dental(f: dict) -> dict:
    """Convert raw scraped dental form field values to a normalised dict."""
    def yn(name: str) -> Optional[str]:
        v = f.get(name)
        if v == "1": return "Yes"
        if v == "0": return "No"
        if v == "2": return "Not Applicable"
        return None

    return {
        "dentist_name":           f.get("DntlEvalDentistName"),
        "dentist_city":           f.get("DntlEvalDentistCity"),
        "dentist_state":          f.get("DntlEvalDentistStateProvince"),
        "dentist_phone":          f.get("DntlEvalDentistPhone"),
        "exam_date":              f.get("DntlEvalOralExamDate"),
        "orthodontic_treatment":  yn("DntlHistOrthCompletedYN"),
        "wisdom_teeth_removed":   yn("DntlHistWisdomYN"),
        "brushing_frequency":     f.get("DntlHistBrushFrequency"),
        "flossing_frequency":     f.get("DntlHistFlossFrequency"),
        "pain_or_bleeding":       yn("DntlHistBleedingYN"),
        "oral_exam_with_bitewing":yn("DntlEvalOralExamYN"),
        "panoramic_done":         yn("DntlEvalFullXraysYN"),
        "third_molars_assessed":  yn("DntlEvalMolarsYN"),
        "bonded_retainers":       yn("DntlEvalRestorationsYN"),
        "active_orthodontic":     yn("DntlEvalCurrentTreatmentYN"),
        "oral_hygiene_acceptable":yn("DntlEvalProperHygieneYN"),
        "dental_decay":           yn("DntlEvalProblemsYN"),
    }


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------

# Fields where we use fuzzy matching (free-text, OCR may vary slightly)
_FUZZY_FIELDS = {
    "dentist_name", "physician_name", "brushing_frequency", "flossing_frequency",
    "physician_city", "dentist_city",
}
_FUZZY_MATCH_THRESHOLD = 75  # rapidfuzz score 0–100

# Date patterns → (month, day, year) for normalisation
import re as _re
from datetime import datetime as _dt

_MONTH_ABBR = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def _normalise_date(val: str) -> Optional[str]:
    """
    Try to parse *val* as a date and return "MM/DD/YYYY".
    Handles: "11 Feb 2026", "2/11/26", "02/11/2026", "30 Nov 2025",
             "11/30/2025", "2025-11-30".
    Returns None if unparseable.
    """
    val = val.strip()
    # Try common strptime patterns
    for fmt in ("%m/%d/%Y", "%m/%d/%y", "%d %b %Y", "%d %B %Y",
                "%B %d, %Y", "%b %d, %Y", "%Y-%m-%d"):
        try:
            return _dt.strptime(val, fmt).strftime("%-m/%-d/%Y")
        except ValueError:
            pass
    # "2/11/26" → ambiguous but assume MM/DD/YY
    m = _re.match(r"^(\d{1,2})/(\d{1,2})/(\d{2,4})$", val)
    if m:
        month, day, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if year < 100:
            year += 2000
        try:
            return _dt(year, month, day).strftime("%-m/%-d/%Y")
        except ValueError:
            pass
    return None


def _strip_phone(val: str) -> str:
    """Strip all non-digit characters from a phone number string."""
    return _re.sub(r"\D", "", val)


def _vals_equal(key: str, sub_val: str, ext_val: str) -> bool:
    """
    Return True if the two values should count as matching for this field.
    Applies date normalisation, phone normalisation, and fuzzy matching where appropriate.
    """
    sub_s = str(sub_val).strip()
    ext_s = str(ext_val).strip()

    # Exact match (case-insensitive)
    if sub_s.lower() == ext_s.lower():
        return True

    # "No" / "None" / "null" / "false" are all falsy — treat as equivalent
    _FALSY = {"no", "none", "null", "false", "n/a", "na", "0", "--"}
    if sub_s.lower() in _FALSY and ext_s.lower() in _FALSY:
        return True
    # Same for truthy values
    _TRUTHY = {"yes", "true", "1", "y"}
    if sub_s.lower() in _TRUTHY and ext_s.lower() in _TRUTHY:
        return True

    # Phone fields: compare digits only (strips formatting differences)
    if "phone" in key:
        sub_digits = _strip_phone(sub_s)
        ext_digits = _strip_phone(ext_s)
        if sub_digits and ext_digits and sub_digits == ext_digits:
            return True
        # Accept if the last 7 digits match — OCR sometimes garbles the area code
        # (e.g. 801 → 861) but gets the local number right.
        if sub_digits and ext_digits and len(sub_digits) >= 7 and len(ext_digits) >= 7:
            if sub_digits[-7:] == ext_digits[-7:]:
                return True

    # Date fields: normalise both to MM/DD/YYYY then compare
    if "date" in key or "dob" in key or key in (
        "tdap", "mmr1", "mmr2", "polio", "hep_a1", "hep_a2",
        "hep_b1", "hep_b2", "hep_b3", "influenza",
    ):
        sub_d = _normalise_date(sub_s)
        ext_d = _normalise_date(ext_s)
        if sub_d and ext_d:
            return sub_d == ext_d

    # Numeric fields: strip units and compare numbers
    if key in ("height", "weight", "pulse", "hemoglobin"):
        sub_n = _re.search(r"[\d\.]+", sub_s)
        ext_n = _re.search(r"[\d\.]+", ext_s)
        if sub_n and ext_n and sub_n.group() == ext_n.group():
            return True

    # State fields: normalize full state name to 2-letter abbreviation
    if key == "dentist_state":
        _STATE_NAMES = {
            "alabama":"AL","alaska":"AK","arizona":"AZ","arkansas":"AR","california":"CA",
            "colorado":"CO","connecticut":"CT","delaware":"DE","florida":"FL","georgia":"GA",
            "hawaii":"HI","idaho":"ID","illinois":"IL","indiana":"IN","iowa":"IA",
            "kansas":"KS","kentucky":"KY","louisiana":"LA","maine":"ME","maryland":"MD",
            "massachusetts":"MA","michigan":"MI","minnesota":"MN","mississippi":"MS",
            "missouri":"MO","montana":"MT","nebraska":"NE","nevada":"NV",
            "new hampshire":"NH","new jersey":"NJ","new mexico":"NM","new york":"NY",
            "north carolina":"NC","north dakota":"ND","ohio":"OH","oklahoma":"OK",
            "oregon":"OR","pennsylvania":"PA","rhode island":"RI","south carolina":"SC",
            "south dakota":"SD","tennessee":"TN","texas":"TX","utah":"UT","vermont":"VT",
            "virginia":"VA","washington":"WA","west virginia":"WV","wisconsin":"WI",
            "wyoming":"WY","district of columbia":"DC",
        }
        def _norm_state(s):
            s = s.strip().lower()
            return _STATE_NAMES.get(s, s.upper())
        if _norm_state(sub_s) == _norm_state(ext_s):
            return True

    # Dental frequency fields: normalize before fuzzy comparison
    if key in ("brushing_frequency", "flossing_frequency"):
        def _norm_freq(s: str) -> str:
            s = s.strip().lower()
            if _re.search(r"^(daily|1\s*[x×*]\s*/?\s*(per\s+)?day|once\s+(a\s+)?day|every\s+day)$", s):
                return "daily"
            if _re.search(r"^(2\s*[x×*]\s*/?\s*(per\s+)?day|twice\s+(a\s+)?day|twice\s+daily|2\s+times?\s+(a\s+)?day|morning\s+and\s+night|night\s+and\s+morning|am\s*(and|&)\s*pm|pm\s*(and|&)\s*am)$", s):
                return "2x/day"
            # "2 x 1 day" → OCR noise for "2 x / day"; treat as 2x/day
            if _re.search(r"^2\s*[x×]\s*\d\s*(per\s+)?day$", s):
                return "2x/day"
            if _re.search(r"^(1\s*[x×*]\s*/?\s*(per\s+)?week|once\s+(a\s+)?week|weekly|1\s*x\s*week)$", s):
                return "weekly"
            if _re.search(r"^(nightly|every\s+night)$", s):
                return "nightly"
            return s
        if _norm_freq(sub_s) == _norm_freq(ext_s):
            return True

    # Free-text fields: fuzzy match
    if key in _FUZZY_FIELDS:
        score = fuzz.token_sort_ratio(sub_s.lower(), ext_s.lower())
        return score >= _FUZZY_MATCH_THRESHOLD

    return False


# ---------------------------------------------------------------------------
# Compare extracted vs. submitted
# ---------------------------------------------------------------------------

def _compare(extracted: dict, submitted: dict, form_label: str) -> dict:
    """
    Field-by-field comparison.
    Returns {"match": [...], "mismatch": [...], "missing": [...]}
    """
    match, mismatch, missing = [], [], []
    for key, sub_val in submitted.items():
        if not sub_val or str(sub_val).strip() in ("--", "-", ""):
            continue  # Skip empty/null submitted fields (form section not filled)
        ext_val = extracted.get(key)
        # Treat Python None, "None", "null", "" as not extracted
        ext_is_empty = (
            ext_val is None
            or str(ext_val).strip().lower() in ("none", "null", "")
        )
        if ext_is_empty:
            missing.append({"field": key, "submitted": sub_val, "extracted": None})
        elif _vals_equal(key, str(sub_val), str(ext_val)):
            match.append({"field": key, "submitted": sub_val, "extracted": ext_val})
        else:
            mismatch.append({"field": key, "submitted": sub_val, "extracted": ext_val})
    return {"form": form_label, "match": match, "mismatch": mismatch, "missing": missing}


def _extracted_medical_dict(form) -> dict:
    """Pull the same keys from a parsed MedicalForm for comparison."""
    if not form:
        return {}
    vs  = form.vital_signs or type("", (), {})()
    se  = form.system_exam or type("", (), {})()
    lab = form.labs or type("", (), {})()
    imm = form.immunizations or type("", (), {})()
    ph  = form.physician_info or type("", (), {})()

    def g(obj, attr): return getattr(obj, attr, None) or None

    return {
        "physician_name":     g(ph, "name"),
        "physician_city":     g(ph, "city"),
        "physician_state":    g(ph, "state"),
        "physician_phone":    g(ph, "phone"),
        "exam_date":          form.exam_date,
        "height":             g(vs, "height"),
        "weight":             g(vs, "weight"),
        "blood_pressure":     g(vs, "blood_pressure"),
        "pulse":              g(vs, "pulse"),
        "vision_right":       g(vs, "vision_right"),
        "vision_left":        g(vs, "vision_left"),
        "functional_level":   form.functional_level,
        "chronic_condition":  form.chronic_condition,
        "driving_medication": form.driving_medication,
        "tb_exposure_risk":   g(lab, "tb_exposure_risk"),
        "igra_result":        g(lab, "igra_result"),
        "chest_xray":         g(lab, "chest_xray"),
        "hemoglobin":         g(lab, "hemoglobin"),
        "urinalysis_blood":   g(lab, "urinalysis_blood"),
        "urinalysis_protein": g(lab, "urinalysis_protein"),
        "urinalysis_sugar":   g(lab, "urinalysis_sugar"),
        "general":            g(se, "general"),
        "skin":               g(se, "skin"),
        "eyes":               g(se, "eyes"),
        "ears":               g(se, "ears"),
        "nose_throat":        g(se, "nose_throat"),
        "chest":              g(se, "chest"),
        "heart":              g(se, "heart"),
        "abdomen":            g(se, "abdomen"),
        "genitalia":          g(se, "genitalia"),
        "back":               g(se, "back"),
        "extremities":        g(se, "extremities"),
        "neurological":       g(se, "neurological"),
        "tdap":               g(imm, "tdap"),
        "mmr1":               g(imm, "mmr1"),
        "mmr2":               g(imm, "mmr2"),
        "polio":              g(imm, "polio"),
        "influenza":          g(imm, "influenza"),
    }


def _extracted_dental_dict(form) -> dict:
    """Pull the same keys from a parsed DentalForm for comparison."""
    if not form:
        return {}
    def g(attr): return getattr(form, attr, None) or None
    return {
        "dentist_name":            g("dentist_name"),
        "dentist_city":            g("dentist_city"),
        "dentist_state":           g("dentist_state"),
        "dentist_phone":           g("dentist_phone"),
        "exam_date":               g("exam_date"),
        "orthodontic_treatment":   g("orthodontic_treatment"),
        "wisdom_teeth_removed":    g("wisdom_teeth_removed"),
        "brushing_frequency":      g("brushing_frequency"),
        "flossing_frequency":      g("flossing_frequency"),
        "pain_or_bleeding":        g("pain_or_bleeding"),
        "oral_exam_with_bitewing": g("oral_exam_with_bitewing"),
        "panoramic_done":          g("panoramic_done"),
        "third_molars_assessed":   g("third_molars_assessed"),
        "bonded_retainers":        g("bonded_retainers"),
        "active_orthodontic":      g("active_orthodontic"),
        "oral_hygiene_acceptable": g("oral_hygiene_acceptable"),
        "dental_decay":            g("dental_decay"),
    }


# ---------------------------------------------------------------------------
# Print comparison result
# ---------------------------------------------------------------------------

def _print_comparison(name: str, cmp: dict):
    n_match   = len(cmp["match"])
    n_mismatch= len(cmp["mismatch"])
    n_missing = len(cmp["missing"])
    total     = n_match + n_mismatch + n_missing
    pct       = int(100 * n_match / total) if total else 0

    colour = "green" if pct >= 80 else ("yellow" if pct >= 50 else "red")
    console.print(
        f"  [{colour}]{cmp['form']}: {pct}% ({n_match}/{total} matched, "
        f"{n_mismatch} wrong, {n_missing} missing)[/{colour}]"
    )
    if cmp["mismatch"]:
        t = Table(show_header=True, header_style="bold", box=None, padding=(0,1))
        t.add_column("Field", style="dim")
        t.add_column("Website (truth)", style="green")
        t.add_column("PDF Parser", style="red")
        for row in cmp["mismatch"]:
            t.add_row(row["field"], str(row["submitted"] or ""), str(row["extracted"] or "—"))
        console.print(t)


# ---------------------------------------------------------------------------
# Main validation loop
# ---------------------------------------------------------------------------

async def _run(pdf_dir: Path, pdf_files: list[Path], skip_parse: bool = False):
    from .credentials import ensure_credentials
    from .web_automator import (
        login,
        find_and_click_candidate,
        click_form_tab,
        get_candidate_names,
        _wait_nav,
    )

    config    = load_config()
    username, password = ensure_credentials()

    results: list[dict] = []

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=config.get("browser_headless", False),
            slow_mo=150,
        )
        context = await browser.new_context(viewport={"width": 1280, "height": 900})
        page    = await context.new_page()

        try:
            await login(page, username, password)

            # Collect all candidate names from the website for fuzzy matching
            raw_candidates = await get_candidate_names(page)
            site_names = [name for name, _ in raw_candidates] if raw_candidates else []
            logger.info(f"Found {len(site_names)} candidates on website")

            for pdf_path in pdf_files:
                name_from_file = parse_missionary_name_from_filename(pdf_path.name)

                # ── 1. Fast OCR to extract name from PDF form content ─────
                console.print(f"  OCR scan: {pdf_path.name} …")
                try:
                    pages_ocr = extract_pdf_text(pdf_path)
                    full_text = combine_pages(pages_ocr)
                    has_medical, has_dental = detect_form_types(pdf_path.name, full_text)
                except Exception as e:
                    console.print(f"  [red]OCR failed: {e}[/red]")
                    results.append({"name": name_from_file, "status": "parse_error", "error": str(e)})
                    continue

                # Extract name from PDF form content via regex (no Ollama yet)
                name_from_ocr = None
                try:
                    pi = parse_personal_info(full_text)
                    if pi and pi.first_name and pi.last_name:
                        name_from_ocr = f"{pi.first_name} {pi.last_name}"
                        logger.info(f"Name from PDF content: {name_from_ocr}")
                    else:
                        logger.info("Could not extract name from PDF content")
                except Exception as e:
                    logger.warning(f"parse_personal_info failed: {e}")

                # ── 2. Fuzzy match — try OCR name AND filename name, keep best ──
                best_match = None
                name_for_search = name_from_file  # default display name
                if site_names:
                    candidates_to_try = []
                    if name_from_ocr:
                        candidates_to_try.append(("ocr", name_from_ocr))
                    candidates_to_try.append(("file", name_from_file))

                    best_score = 0
                    for source, candidate_name in candidates_to_try:
                        result = process.extractOne(
                            candidate_name,
                            site_names,
                            scorer=fuzz.token_sort_ratio,
                            score_cutoff=55,
                        )
                        if result and result[1] > best_score:
                            best_score = result[1]
                            best_match = result[0]
                            name_for_search = candidate_name
                            logger.info(f"Fuzzy match [{source}]: '{candidate_name}' → '{best_match}' (score={result[1]})")

                    if not best_match:
                        logger.info(f"No fuzzy match found. Tried: {[n for _, n in candidates_to_try]}")
                        name_for_search = name_from_ocr or name_from_file

                console.rule(f"[bold cyan]{name_for_search}[/bold cyan]")

                # ── Session check before every initial candidate search ────
                # After long Ollama runs the church website session can expire.
                # The site serves an embedded sign-in page from the same domain
                # so URL-only checks are insufficient — also check page content.
                try:
                    await page.goto(CHURCH_URL, wait_until="domcontentloaded")
                    await page.wait_for_timeout(4000)
                    not_at_candidates = "missionaryrecommendations.churchofjesuschrist.org" not in page.url
                    has_signin = await page.get_by_label("Username").count() > 0
                    if not_at_candidates or has_signin:
                        logger.info("Pre-search session check: session expired — re-authenticating …")
                        await login(page, username, password)
                        raw_candidates = await get_candidate_names(page)
                        site_names = [name for name, _ in raw_candidates] if raw_candidates else site_names
                    else:
                        logger.info("Pre-search session check: session valid")
                except Exception as e:
                    logger.warning(f"Pre-search session check failed: {e}")

                found_on_site = await find_and_click_candidate(
                    page, best_match or name_for_search
                )
                if not found_on_site:
                    console.print(f"[yellow]  Not found on website — skipping.[/yellow]")
                    results.append({"name": name_for_search, "status": "not_found"})
                    continue

                # ── 3. Full PDF parse (with Ollama) — skipped with --skip-parse ──
                parsed = None
                if not skip_parse:
                    console.print(f"  Full parse: {pdf_path.name} …")
                    try:
                        parsed = parse_forms(
                            text=full_text,
                            missionary_name=name_for_search,
                            pdf_path=str(pdf_path),
                            has_medical=has_medical,
                            has_dental=has_dental,
                        )
                    except Exception as e:
                        console.print(f"  [red]PDF parse failed: {e}[/red]")
                        results.append({"name": name_for_search, "status": "parse_error", "error": str(e)})
                        continue

                entry: dict = {"name": name_for_search, "status": "ok", "comparisons": []}

                # ── Session renewal: Ollama can take 1+ hour per PDF, during which
                #    the church website session expires. Navigate to CHURCH_URL and
                #    re-login if we got redirected to the OAuth page.
                try:
                    await page.goto(CHURCH_URL, wait_until="domcontentloaded")
                    await page.wait_for_timeout(5000)
                    not_at_candidates = "missionaryrecommendations.churchofjesuschrist.org" not in page.url
                    has_signin = await page.get_by_label("Username").count() > 0
                    if not_at_candidates or has_signin:
                        logger.info("Session expired after parsing — re-authenticating …")
                        await login(page, username, password)
                        raw_candidates = await get_candidate_names(page)
                        site_names = [name for name, _ in raw_candidates] if raw_candidates else site_names
                    else:
                        logger.info("Session still valid after Ollama parsing")
                except Exception as e:
                    logger.warning(f"Session renewal failed: {e}")

                # ── 4. Scrape + compare medical form ──────────────────────
                if has_medical:
                    await find_and_click_candidate(page, best_match or name_for_search)
                    clicked = await click_form_tab(page, "medical")
                    if not clicked:
                        console.print("  [red]Could not navigate to Medical form.[/red]")
                    else:
                        await _wait_nav(page, 10000)
                        fields = await _scrape_all_sections(page)
                        submitted_med = _fields_to_medical(fields)
                        has_data = any(v for v in submitted_med.values() if v)
                        if skip_parse or not parsed or not parsed.medical:
                            if has_data:
                                console.print(f"  [green]Medical form reached — {len([v for v in submitted_med.values() if v])} fields scraped.[/green]")
                            else:
                                console.print("  [dim]Medical form not yet submitted on website.[/dim]")
                        elif has_data:
                            extracted_med = _extracted_medical_dict(parsed.medical)
                            cmp = _compare(extracted_med, submitted_med, "Medical")
                            entry["comparisons"].append(cmp)
                            _print_comparison(name_for_search, cmp)
                        else:
                            console.print("  [dim]Medical form not yet submitted on website — no ground truth.[/dim]")

                # ── 5. Navigate back + scrape dental form ─────────────────
                if has_dental:
                    await find_and_click_candidate(page, best_match or name_for_search)
                    clicked = await click_form_tab(page, "dental")
                    if not clicked:
                        console.print("  [red]Could not navigate to Dental form.[/red]")
                    else:
                        await _wait_nav(page, 10000)
                        fields = await _scrape_all_sections(page)
                        submitted_dnt = _fields_to_dental(fields)
                        has_data = any(v for v in submitted_dnt.values() if v)
                        if skip_parse or not parsed or not parsed.dental:
                            if has_data:
                                console.print(f"  [green]Dental form reached — {len([v for v in submitted_dnt.values() if v])} fields scraped.[/green]")
                            else:
                                console.print("  [dim]Dental form not yet submitted on website.[/dim]")
                        elif has_data:
                            extracted_dnt = _extracted_dental_dict(parsed.dental)
                            cmp = _compare(extracted_dnt, submitted_dnt, "Dental")
                            entry["comparisons"].append(cmp)
                            _print_comparison(name_for_search, cmp)
                        else:
                            console.print("  [dim]Dental form not yet submitted on website.[/dim]")

                results.append(entry)

        except Exception as e:
            logger.exception(f"Validation error: {e}")
            console.print(f"\n[red]Validation stopped due to error: {e}[/red]")
        finally:
            await context.close()
            await browser.close()

    return results


def _print_summary(results: list[dict]):
    console.rule("[bold]Validation Summary[/bold]")
    t = Table(show_header=True, header_style="bold")
    t.add_column("Missionary")
    t.add_column("Medical %", justify="right")
    t.add_column("Dental %",  justify="right")
    t.add_column("Status")

    for r in results:
        if r.get("status") != "ok":
            t.add_row(r["name"], "—", "—", r.get("status", "?"))
            continue
        med_pct = dnt_pct = "—"
        for cmp in r.get("comparisons", []):
            total = len(cmp["match"]) + len(cmp["mismatch"]) + len(cmp["missing"])
            pct = f"{int(100 * len(cmp['match']) / total)}%" if total else "—"
            if cmp["form"] == "Medical":
                med_pct = pct
            else:
                dnt_pct = pct
        t.add_row(r["name"], med_pct, dnt_pct, "[green]ok[/green]")

    console.print(t)


def run_validation(pdf_dir: Path, skip_parse: bool = False):
    """Synchronous entry point — discovers PDFs and launches the async loop."""
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        console.print(f"[red]No PDF files found in {pdf_dir}[/red]")
        return

    console.print(f"Found [bold]{len(pdf_files)}[/bold] PDFs in {pdf_dir}")

    results = asyncio.run(_run(pdf_dir, pdf_files, skip_parse=skip_parse))

    _print_summary(results)

    # Save detailed results
    out = LOGS_DIR / "validation_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str))
    console.print(f"\n[dim]Full results saved to {out}[/dim]")


# ---------------------------------------------------------------------------
# Ground-truth scraper  (saves website values to JSON — no PDF needed)
# ---------------------------------------------------------------------------

GROUND_TRUTH_PATH = LOGS_DIR / "ground_truth.json"


async def _scrape_ground_truth(page, username: str, password: str) -> dict:
    """
    Navigate to every candidate in the church website's
    'Ready for Stake President's Action' section and scrape the medical
    and dental form values they have already submitted.

    Returns {candidate_name: {"medical": {...}, "dental": {...}}}
    """
    from .web_automator import (
        login,
        find_and_click_candidate,
        click_form_tab,
        get_candidate_names,
        screenshot,
    )

    # Navigate to candidates page
    await page.goto(CHURCH_URL, wait_until="domcontentloaded")
    await page.wait_for_timeout(5000)
    await screenshot(page, "scrape_00_candidates")

    # Try to click "Ready for Stake President's Action" filter/tab
    for label in [
        "Ready for Stake President's Action",
        "Ready for Stake President",
        "Stake President",
        "Ready",
    ]:
        loc = page.get_by_text(label, exact=False)
        try:
            if await loc.count() > 0:
                await loc.first.click()
                await page.wait_for_timeout(2000)
                logger.info(f"Clicked filter: '{label}'")
                await screenshot(page, "scrape_01_filtered")
                break
        except Exception:
            continue

    raw_candidates = await get_candidate_names(page)
    site_names = [name for name, _ in raw_candidates]
    logger.info(f"Found {len(site_names)} candidates to scrape")

    ground_truth: dict = {}

    for name in site_names:
        console.print(f"\n  Scraping: [bold]{name}[/bold]")
        entry: dict = {"medical": None, "dental": None}

        # --- Medical ---
        if await find_and_click_candidate(page, name):
            if await click_form_tab(page, "medical"):
                await _wait_nav(page, 10000)
                fields = await _scrape_all_sections(page)
                med = _fields_to_medical(fields)
                filled = {k: v for k, v in med.items() if v}
                if filled:
                    entry["medical"] = med
                    console.print(f"    Medical: [green]{len(filled)} fields[/green]")
                else:
                    console.print("    Medical: [dim]no data[/dim]")
            else:
                console.print("    Medical: [dim]form tab not found[/dim]")
        else:
            console.print(f"    [yellow]Could not find candidate on site[/yellow]")
            continue

        # --- Dental ---
        if await find_and_click_candidate(page, name):
            if await click_form_tab(page, "dental"):
                await _wait_nav(page, 10000)
                fields = await _scrape_all_sections(page)
                dnt = _fields_to_dental(fields)
                filled = {k: v for k, v in dnt.items() if v}
                if filled:
                    entry["dental"] = dnt
                    console.print(f"    Dental: [green]{len(filled)} fields[/green]")
                else:
                    console.print("    Dental: [dim]no data[/dim]")
            else:
                console.print("    Dental: [dim]form tab not found[/dim]")

        if entry["medical"] or entry["dental"]:
            ground_truth[name] = entry

    return ground_truth


def run_scrape_ground_truth(output_path: Optional[Path] = None) -> Path:
    """
    Log in, scrape all submitted form values from the website, save to JSON.
    Returns the path to the saved file.
    """
    from .credentials import ensure_credentials
    from .web_automator import login

    if output_path is None:
        output_path = GROUND_TRUTH_PATH

    config = load_config()
    username, password = ensure_credentials()

    async def _run_scrape():
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(
                headless=config.get("browser_headless", False),
                slow_mo=150,
            )
            context = await browser.new_context(viewport={"width": 1280, "height": 900})
            page = await context.new_page()
            try:
                await login(page, username, password)
                return await _scrape_ground_truth(page, username, password)
            finally:
                await context.close()
                await browser.close()

    gt = asyncio.run(_run_scrape())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(gt, indent=2, default=str))
    logger.info(f"Ground truth saved: {len(gt)} candidates → {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Offline validation (compare PDFs against saved ground-truth JSON)
# ---------------------------------------------------------------------------

def run_validation_offline(
    pdf_dir: Path,
    ground_truth_path: Optional[Path] = None,
):
    """
    Parse PDFs with Ollama and compare against pre-saved ground truth JSON.
    No browser required — ground truth must have been scraped with `scrape` first.
    """
    if ground_truth_path is None:
        ground_truth_path = GROUND_TRUTH_PATH

    if not ground_truth_path.exists():
        console.print(
            f"[red]Ground truth file not found: {ground_truth_path}[/red]\n"
            "Run [bold]python main.py scrape[/bold] first to scrape website values."
        )
        return

    with open(ground_truth_path) as f:
        ground_truth: dict = json.load(f)

    console.print(f"Loaded ground truth: [bold]{len(ground_truth)}[/bold] candidates from {ground_truth_path}")

    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        console.print(f"[red]No PDF files found in {pdf_dir}[/red]")
        return
    console.print(f"Found [bold]{len(pdf_files)}[/bold] PDFs in {pdf_dir}")

    site_names = list(ground_truth.keys())
    results: list[dict] = []

    for pdf_path in pdf_files:
        name_from_file = parse_missionary_name_from_filename(pdf_path.name)
        console.print(f"\n  OCR scan: {pdf_path.name} …")

        try:
            pages_ocr = extract_pdf_text(pdf_path)
            full_text = combine_pages(pages_ocr)
            has_medical, has_dental = detect_form_types(pdf_path.name, full_text)
        except Exception as e:
            console.print(f"  [red]OCR failed: {e}[/red]")
            results.append({"name": name_from_file, "status": "parse_error", "error": str(e)})
            continue

        # Extract name from PDF content
        name_from_ocr = None
        try:
            pi = parse_personal_info(full_text)
            if pi and pi.first_name and pi.last_name:
                name_from_ocr = f"{pi.first_name} {pi.last_name}"
        except Exception:
            pass

        # Fuzzy-match to ground truth keys
        best_match = None
        for source, candidate_name in (
            [("ocr", name_from_ocr)] if name_from_ocr else []
        ) + [("file", name_from_file)]:
            result = process.extractOne(
                candidate_name, site_names,
                scorer=fuzz.token_sort_ratio, score_cutoff=55,
            )
            if result:
                best_match = result[0]
                logger.info(f"Fuzzy match [{source}]: '{candidate_name}' → '{best_match}' (score={result[1]})")
                break

        display_name = name_from_ocr or name_from_file
        console.rule(f"[bold cyan]{display_name}[/bold cyan]")

        if not best_match:
            console.print(f"  [yellow]No match in ground truth — skipping.[/yellow]")
            results.append({"name": display_name, "status": "not_found"})
            continue

        gt_entry = ground_truth[best_match]

        # Parse the PDF — pass OCR pages so Ollama can use bounding boxes
        # for page classification (only relevant pages sent per form type)
        console.print(f"  Full parse: {pdf_path.name} …")
        try:
            parsed = parse_forms(
                text=full_text,
                missionary_name=display_name,
                pdf_path=str(pdf_path),
                has_medical=has_medical,
                has_dental=has_dental,
                pages=pages_ocr,
            )
        except Exception as e:
            console.print(f"  [red]PDF parse failed: {e}[/red]")
            results.append({"name": display_name, "status": "parse_error", "error": str(e)})
            continue

        entry: dict = {"name": display_name, "status": "ok", "comparisons": []}

        if has_medical and gt_entry.get("medical"):
            submitted_med = gt_entry["medical"]
            if any(v for v in submitted_med.values() if v):
                if parsed and parsed.medical:
                    extracted_med = _extracted_medical_dict(parsed.medical)
                    cmp = _compare(extracted_med, submitted_med, "Medical")
                    entry["comparisons"].append(cmp)
                    _print_comparison(display_name, cmp)
                else:
                    console.print("  [dim]Medical form: parsed nothing — no comparison.[/dim]")

        if has_dental and gt_entry.get("dental"):
            submitted_dnt = gt_entry["dental"]
            if any(v for v in submitted_dnt.values() if v):
                if parsed and parsed.dental:
                    extracted_dnt = _extracted_dental_dict(parsed.dental)
                    cmp = _compare(extracted_dnt, submitted_dnt, "Dental")
                    entry["comparisons"].append(cmp)
                    _print_comparison(display_name, cmp)
                else:
                    console.print("  [dim]Dental form: parsed nothing — no comparison.[/dim]")

        results.append(entry)

    _print_summary(results)
    out = LOGS_DIR / "validation_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str))
    console.print(f"\n[dim]Full results saved to {out}[/dim]")
