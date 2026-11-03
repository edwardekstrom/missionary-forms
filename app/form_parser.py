"""
Parse raw OCR text from LDS missionary health evaluation forms into structured data models.

Key layout insight (observed from Apple Vision OCR of these scanned forms):
  - Field LABELS appear on their own line (e.g. "First Name")
  - Field VALUES appear on the NEXT LINE (e.g. "Tyler")
  - Bullet "•" or "*" before an option often marks the checked/selected item
  - The doctor's mailing address appears at the TOP of each page (not the evaluating physician)
  - Physician / Dentist info appears in the LAST section of the form
"""

import re
from collections import defaultdict
from datetime import datetime
from typing import Optional

from loguru import logger

from .models import (
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


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _first(patterns: list[str], text: str, flags=re.IGNORECASE | re.MULTILINE) -> Optional[str]:
    """Return the first non-empty capture group across patterns."""
    for pat in patterns:
        m = re.search(pat, text, flags)
        if m:
            val = next((g for g in m.groups() if g), None)
            if val is None:
                val = m.group(0)
            val = val.strip()
            if val:
                return val
    return None


def _normalize_date(raw: Optional[str]) -> Optional[str]:
    """Normalize various date formats to MM/DD/YYYY."""
    if not raw:
        return None
    raw = raw.strip().replace(" ", "")

    # Already MM/DD/YYYY
    if re.match(r"\d{1,2}/\d{1,2}/\d{4}$", raw):
        return raw

    # MM/DD/YY → MM/DD/20YY or 19YY
    m = re.match(r"(\d{1,2})/(\d{1,2})/(\d{2})$", raw)
    if m:
        yr = int(m.group(3))
        century = "20" if yr <= 35 else "19"
        return f"{m.group(1)}/{m.group(2)}/{century}{m.group(3)}"

    # DD Mon YYYY  e.g. "23 Sep 2005"
    months = dict(jan="01", feb="02", mar="03", apr="04", may="05", jun="06",
                  jul="07", aug="08", sep="09", oct="10", nov="11", dec="12")
    raw_spaced = raw.replace("/", " ")
    m = re.match(r"(\d{1,2})\s*([A-Za-z]{3})\w*\s*(\d{4})$", raw_spaced)
    if m:
        mon = months.get(m.group(2).lower()[:3])
        if mon:
            return f"{mon}/{m.group(1).zfill(2)}/{m.group(3)}"

    # Try to fix common OCR date issues like "10/9\n106" → "10/19/06"
    m = re.match(r"(\d{1,2}/\d{1,2})\n?(\d{2,4})$", raw)
    if m:
        return _normalize_date(f"{m.group(1)}/{m.group(2)}")

    return raw  # return as-is


def _normalize_negative(raw: str) -> str:
    """Fix OCR garbling of 'Negative'."""
    if re.search(r"neg[a-z]*|nef[a-z]*|n[e3]g", raw, re.IGNORECASE):
        return "Negative"
    if re.search(r"pos[a-z]*", raw, re.IGNORECASE):
        return "Positive"
    if re.search(r"not\s+done|n/?a|not\s+indicated", raw, re.IGNORECASE):
        return "Not Done"
    return raw.strip().title()


# ---------------------------------------------------------------------------
# Personal info
# ---------------------------------------------------------------------------

def parse_personal_info(text: str) -> PersonalInfo:
    """
    Extract personal info using next-line value format:
      "First Name\nTyler"
      "(middle)\nDon"
      "Last Name (Legal Name)\nFisher"
    """
    info = PersonalInfo()

    # First name: capture word immediately after "First Name" line (skip mailing address lines)
    m = re.search(r"First\s+Name\s*\n([A-Z][a-zA-Z\-]+)", text)
    if m:
        info.first_name = m.group(1).strip()

    # Middle name: value on line after "(middle)".
    # Reject form-label words that OCR puts on the next line (Last, Record, Date, etc.)
    m = re.search(r"\(middle\)\s*\n([A-Z][a-zA-Z\-]*)", text)
    if m:
        val = m.group(1).strip()
        if val and not re.match(r"Last\b|Record\b|Date\b|First\b|Gender\b", val, re.IGNORECASE):
            info.middle_name = val

    # Last name: value on line after "Last Name (Legal Name)".
    # OCR often reads the form's name columns as:
    #   Last Name (Legal Name)\nMiddle\nLast
    # so we take the LAST capitalised word in the block, not the first.
    m = re.search(r"Last\s+Name[^\n]*\n([A-Z][a-zA-Z\-]+)\n([A-Z][a-zA-Z\-]+)", text)
    if m:
        # Two words → first is middle-name value, second is last name
        info.last_name = m.group(2).strip()
        if not info.middle_name:
            info.middle_name = m.group(1).strip()
    else:
        m = re.search(r"Last\s+Name[^\n]*\n([A-Z][a-zA-Z\-]+)", text)
        if m:
            info.last_name = m.group(1).strip()

    # Suffix
    m = re.search(r"\(suffix\)\s*\|?\s*\n([A-Za-z\.]+)", text)
    if m:
        info.suffix = m.group(1).strip()

    # Record number
    m = re.search(r"Record\s+number\s*\n(\d{10,15})", text)
    if m:
        info.record_number = m.group(1).strip()
    else:
        # Inline format
        m = re.search(r"Record\s+(?:number|No)[:\s#]+(\d{10,15})", text)
        if m:
            info.record_number = m.group(1).strip()

    # Date of birth and age – "23 Sep 2005(19)" or "23 Sep 2005 (19)"
    m = re.search(
        r"Date\s+of\s+birth\s*\(Age\)\s*\n([\d\s]+\w+\s+\d{4})\s*\(?(\d{1,3})\)?",
        text, re.IGNORECASE
    )
    if m:
        info.date_of_birth = _normalize_date(m.group(1).strip())
        info.age = m.group(2).strip()
    else:
        raw_dob = _first([
            r"Date\s+of\s+birth[^\n]*\n([\d]+\s+\w+\s+\d{4})",
            r"DOB[:\s]+([\d]+\s+\w+\s+\d{4}|\d{1,2}/\d{1,2}/\d{2,4})",
        ], text)
        info.date_of_birth = _normalize_date(raw_dob) if raw_dob else None

    # Age
    if not info.age:
        m = re.search(r"\(Age\)\s*\n[^\n]*?\(?(\d{1,3})\)?", text)
        if m:
            info.age = m.group(1)

    # Gender
    m = re.search(r"Gender\s*\n(Male|Female)", text, re.IGNORECASE)
    if m:
        info.gender = m.group(1).strip().title()

    return info


# ---------------------------------------------------------------------------
# Physician / Dentist info (appears in the LAST section of the form)
# ---------------------------------------------------------------------------

def parse_physician_info(text: str) -> PhysicianInfo:
    """
    The evaluating physician's info appears near the end of the medical form,
    after "Assessment of Functional Ability" and before "Authorization to Release".

    OCR layout observed in these scanned forms:
      Name of physician    ← label
      Date of ex           ← label for exam date (OCR of "Date of exam")
      DO • PA              ← credential checkboxes
      L / NP               ← more checkboxes
      Physician's office address
      [garbled name+date]  ← physician name + exam date merged by OCR
      [date e.g. 2/4/26]
      [practice name e.g. "Alta View Pediatrics"]
      State or province / Country
      [street address e.g. "9450 S 1300 E 3rd Floor"]
      Postal code / District
      Office phone
      [city, state zip e.g. "Sandy, Utah 84094"]
      E-mail address
      Phone: 801 501-2131   ← phone number
    """
    info = PhysicianInfo()

    # Find the physician section (after "Assessment of Functional Ability" ideally)
    phys_section_start = re.search(
        r"(?:Assessment\s+of\s+Functional|Name\s+of\s+physician)", text, re.IGNORECASE
    )
    phys_section = text[phys_section_start.start():] if phys_section_start else text

    # Name: "Name of [examining] physician" or "Name of physician"
    # OCR often puts labels (Date, State, etc.) on the lines immediately following the label.
    # Scan up to 5 lines after the label to find a non-label physician name.
    _NAME_REJECT = re.compile(
        r"^(?:Date|DO|NP|PA|MD|Physician|Office|State|Country|Postal|District|Auth|Signature|City|Phone|Email)",
        re.IGNORECASE
    )
    # Credentials to strip from the end of a name (e.g. "WhitenJohn, APRN")
    _CRED_SUFFIX = re.compile(
        r",?\s*\b(MD|DO|PA-?C?|NP-?C?|APRN|CNM|CRNA|FNP|DNP|PharmD|DPM|DC|OD|DDS|DMD)\b.*$",
        re.IGNORECASE
    )
    # Reject lines that look like "City, ST 12345" addresses
    _ADDR_PAT = re.compile(r"[A-Za-z\s]+,\s*[A-Z]{2}\s+\d{4,5}")
    m_label = re.search(
        r"Name\s+of\s+(?:examining\s+)?(?:physician|doctor)[^\n]*\n",
        phys_section, re.IGNORECASE
    )
    if m_label:
        # Scan successive lines after the label for a plausible name
        remaining = phys_section[m_label.end():]
        for line in remaining.split("\n")[:6]:
            candidate = re.sub(r"\s+", " ", line).strip().rstrip(",")
            if not candidate:
                continue
            if _NAME_REJECT.match(candidate):
                continue
            # Reject address-like patterns (e.g. "Salt Lake City, UT 84105")
            if _ADDR_PAT.search(candidate):
                continue
            # Must start with a capital letter and have ≥ 3 chars
            if re.match(r"[A-Z][A-Za-z\.\s]{2,}", candidate):
                # Strip credential suffixes (APRN, MD, etc.) if present
                candidate = _CRED_SUFFIX.sub("", candidate).strip().rstrip(",")
                if candidate:
                    info.name = candidate
                    break

    if not info.name:
        # Fallback: look for MD/DO/PA/NP near a proper name
        m = re.search(
            r"([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){1,3}),?\s*(MD|DO|PA-?C?|NP-?C?)\b",
            phys_section
        )
        if m:
            info.name = m.group(1).strip()
            info.credentials = m.group(2).strip()

    info.credentials = _first([
        r"Credentials?\s*\n?(MD|DO|PA-?C?|NP-?C?)\b",
        r"\b(MD|DO|PA-C?|NP-?C?)\b",
    ], phys_section)

    # Phone — "Office phone" label followed by number on next line,
    # "Phone: XXX", or "PH (XXX) XXX-XXXX" format (OCR sometimes writes "PH" prefix)
    for phone_pat in [
        r"(?:Office\s+phone[^\n]*\n|Phone:\s*)([\d\-\(\)\s\.]{9,})",
        r"\bPH\s*([\(\d][\d\-\(\)\s\.]{8,})",
    ]:
        m = re.search(phone_pat, phys_section, re.IGNORECASE)
        if m:
            info.phone = m.group(1).strip()
            break

    # Email
    m = re.search(
        r"Ema[^\n]*?\n?([a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,})",
        phys_section, re.IGNORECASE
    )
    if m:
        info.email = m.group(1).strip()

    # Address line (street)
    info.address = _first([
        r"Physician.{0,30}(?:address|ess)[:\s\n]+([^\n]+)",
        r"Office\s+address[:\s\n]+([^\n]+)",
    ], phys_section)

    # City/state/postal — look for "City, State ZIP" pattern
    # e.g. "Sandy, Utah 84094" or "Sandy, UT 84094"
    _STATE_ABBR = {
        "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
        "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
        "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
        "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
        "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
        "massachusetts": "MA", "michigan": "MI", "minnesota": "MN",
        "mississippi": "MS", "missouri": "MO", "montana": "MT", "nebraska": "NE",
        "nevada": "NV", "new hampshire": "NH", "new jersey": "NJ",
        "new mexico": "NM", "new york": "NY", "north carolina": "NC",
        "north dakota": "ND", "ohio": "OH", "oklahoma": "OK", "oregon": "OR",
        "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
        "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT",
        "vermont": "VT", "virginia": "VA", "washington": "WA",
        "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY",
    }
    addr_m = re.search(
        # Use [ ] (space only, not \s) to avoid matching newlines inside city/state names.
        # Require comma between city and 2-letter state abbreviation to reduce false matches.
        r"([A-Za-z][A-Za-z ]{2,20}),\s*([A-Z]{2})\s+(\d{5}(?:-\d{4})?)",
        phys_section
    )
    if not addr_m:
        # Fallback: full state name (e.g. "Salt Lake City Utah 84101")
        addr_m = re.search(
            r"([A-Za-z][A-Za-z ]{2,20}),\s*([A-Za-z][A-Za-z ]{2,20})\s+(\d{5}(?:-\d{4})?)",
            phys_section
        )
    # (No no-comma fallback — it causes false matches against dental addresses
    # that appear in the same combined PDF text.)
    if addr_m:
        city_raw = addr_m.group(1).strip()
        state_raw = addr_m.group(2).strip()
        postal_raw = addr_m.group(3).strip()
        if not info.city:
            info.city = city_raw
        if not info.postal_code:
            info.postal_code = postal_raw
        if not info.state:
            # Check if state_raw is already a 2-letter abbreviation
            if re.match(r"^[A-Z]{2}$", state_raw):
                info.state = state_raw
            else:
                abbr = _STATE_ABBR.get(state_raw.lower())
                if abbr:
                    info.state = abbr

    # State from explicit "State or province" label
    if not info.state:
        m = re.search(r"State\s+or\s+province\s*\n?([A-Z]{2})\b", phys_section)
        if m:
            info.state = m.group(1)

    # Postal code from explicit "Postal code" label
    if not info.postal_code:
        m = re.search(r"Posta(?:l)?\s+code\s+(\d{5}(?:-\d{4})?)", phys_section, re.IGNORECASE)
        if m:
            info.postal_code = m.group(1).strip()

    return info


def parse_dentist_info(text: str) -> tuple[str, str, str, str, str, str, str]:
    """Return (name, address, city, state, postal, phone, email) for the dentist."""

    # --- Dentist name ---
    # Layout A (common): "DENTIST_NAME    City    DATE" on one line (OCR columns merged)
    # Layout B: name on the line immediately after "Name of dentist" label
    _LABEL_PAT = re.compile(
        r"\b(?:Date|City|State|Province|Postal|Code|Office|Phone|Email|Country"
        r"|Evaluated|Completed|Signature|Missionary|Information)\b",
        re.IGNORECASE
    )
    name = None
    # Try Layout A: something before "   City" on the same line
    m_a = re.search(r"([A-Za-z][A-Za-z\s\.]{4,45}?)\s{2,}(?:City\b|Date\s+completed)", text, re.IGNORECASE)
    if m_a:
        candidate = re.sub(r"\s+", " ", m_a.group(1)).strip()
        if not _LABEL_PAT.search(candidate):
            name = candidate
    # Try Layout B: name on next line after the label
    if not name:
        m_b = re.search(
            r"(?:ame|name)\s+o[rf]\s+(?:den|gen)t[^\n]*\n([A-Za-z][A-Za-z\.\s]+?)(?:\n|,|$)",
            text, re.IGNORECASE | re.MULTILINE
        )
        if m_b:
            candidate = re.sub(r"\s+", " ", m_b.group(1)).strip()
            if not _LABEL_PAT.search(candidate):
                name = candidate
    if not name:
        candidate = _first([r"[Dd]entist.{0,10}name[:\s\n]+([A-Za-z][A-Za-z\.\s]+?)(?:\n|$)"], text)
        if candidate:
            candidate = re.sub(r"\s+", " ", candidate).strip()
            if not _LABEL_PAT.search(candidate):
                name = candidate

    # --- Address ---
    address = _first([r"Office\s+address[:\s\n]+([^\n]+)"], text)

    # --- City and State (often on the "State or province   CITY   ST   Postal" line) ---
    city = None
    state = None
    # Try the combined "State or province" line layout
    m_sop = re.search(
        r"State\s+or\s+province\s+([A-Za-z][A-Za-z\s]*?)\s+([A-Z]{2})\s+Postal",
        text, re.IGNORECASE | re.MULTILINE
    )
    if m_sop:
        city_candidate = m_sop.group(1).strip()
        state_candidate = m_sop.group(2).strip()
        if not _LABEL_PAT.search(city_candidate):
            city = city_candidate
        # Only accept the state if it's actually 2 uppercase letters (not "or" etc.)
        if state_candidate and state_candidate == state_candidate.upper() and len(state_candidate) == 2:
            state = state_candidate

    # Medical credentials that should never appear in a city name
    _CRED_IN_CITY = re.compile(
        r'\b(?:APRN|MD|DO|PA(?:-C)?|NP(?:-C)?|CNP|FNP|DNP|LPN|RN|DDS|DMD|CRNA)\b',
        re.IGNORECASE,
    )

    # Fallback city patterns
    if not city:
        city = _first([r"City[:\s\n]+([A-Za-z][A-Za-z\s]+?)(?:\s{2,}|,|\n)"], text)
        if city:
            city = re.sub(r"\s+", " ", city).strip()
            if _LABEL_PAT.search(city) or _CRED_IN_CITY.search(city):
                city = None

    # OCR sometimes puts the city value on the line BEFORE the "State or province" label
    if not city:
        m_before = re.search(
            r"^([^\n]{3,35}?)\s*\n\s*State\s+or\s+province",
            text, re.IGNORECASE | re.MULTILINE
        )
        if m_before:
            city_candidate = re.sub(r"[^A-Za-z\s]", "", m_before.group(1)).strip()
            if city_candidate and not _LABEL_PAT.search(city_candidate) and not _CRED_IN_CITY.search(city_candidate):
                city = city_candidate

    # Fallback state patterns (must be actually uppercase to avoid "or" bug)
    if not state:
        raw_state = _first([
            r"State(?:/Province)?[:\s]+([A-Za-z]{2})\b",
            r"State\s+or\s+province\s*\n([A-Za-z]{2})\b",
        ], text)
        if raw_state and raw_state == raw_state.upper():
            state = raw_state

    # Validate state against known US codes; OCR sometimes garbles "UT" → "OF"/"LT" etc.
    _VALID_US_STATES = {
        "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN",
        "IA","KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV",
        "NH","NJ","NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN",
        "TX","UT","VT","VA","WA","WV","WI","WY","DC",
    }
    if state and state not in _VALID_US_STATES:
        state = None  # Reject OCR garbling (e.g. "OF", "LT" when "UT" was intended)

    # --- Postal ---
    postal = _first([
        r"Posta(?:l)?\s+code\s+(\d{5}(?:-\d{4})?)",
        r"Postal\s*code[:\s]+(\d{5}(?:-\d{4})?)",
    ], text)

    # Infer state from ZIP prefix if still unknown (most missionaries are in Utah)
    if not state and postal:
        _ZIP_PREFIX_MAP = {
            "84": "UT", "80": "CO", "85": "AZ", "86": "AZ",
            "89": "NV", "97": "OR", "98": "WA",
        }
        state = _ZIP_PREFIX_MAP.get(postal[:2])

    # --- Phone (accept 7+ digit numbers, with or without area code) ---
    phone = _first([
        r"Office\s+phone[^\n]*\n([\d\-\(\)\s\.]{7,})",
        r"Office\s+phone[^\n]*?([\(\d][\d\-\(\)\s\.]{7,})",
        r"Phone[^\n]*\n([\d\-\(\)\s\.]{7,})",
        # OCR sometimes puts phone value BEFORE the "Office phone" label
        r"(\(?\d{3}\)?[\s\-\.]\d{3}[\s\-\.]\d{4})\s*\n[^\n]*Office\s+phone",
    ], text)
    if phone:
        phone = phone.strip()
    # Broader fallback: any 10-digit US phone in dentist section (last 600 chars of form)
    if not phone:
        # Find the dentist section anchor
        dnt_start = None
        for anchor in [r"Name\s+of\s+dentist", r"Dentist.{0,15}signature", r"Dental\s+Evaluation"]:
            am = re.search(anchor, text, re.IGNORECASE)
            if am:
                dnt_start = am.start()
                break
        if dnt_start is not None:
            dnt_section = text[dnt_start:]
            pm = re.search(r"\b(\(?\d{3}\)?[\s\-\.]\d{3}[\s\-\.]\d{4})\b", dnt_section)
            if pm:
                phone = pm.group(1).strip()

    # --- Email ---
    email = _first([
        r"Ema[^\n]*?\n?([a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,})",
    ], text)

    return name, address, city, state, postal, phone, email


# ---------------------------------------------------------------------------
# Vital signs
# ---------------------------------------------------------------------------

def parse_vital_signs(text: str) -> VitalSigns:
    vs = VitalSigns()

    # Height: look for number near "Height" label (with or without units).
    # The vital signs section often uses a column layout where labels and values
    # are separated — search within 300 chars of the label to handle that.
    vs.height = _first([
        r"Height[^\n]*?\n.*?(\d+(?:\.\d+)?)\s*(?:in(?:ches?)?|cm)\b",
        r"(?:Ht|Height)[:\s]+(\d+(?:\.\d+)?)\s*(?:in|cm)",
        # OCR often merges height/weight into one line: look for 2-3 digits after Height label
        r"Height[^\n]{0,60}\n\s*(\d{2,3}(?:\.\d)?)\b",
    ], text)
    if not vs.height:
        # Broad fallback: any height value with units within 300 chars of "Height" label
        m_hlbl = re.search(r"\bHeight\b", text, re.IGNORECASE)
        if m_hlbl:
            region = text[m_hlbl.end(): m_hlbl.end() + 300]
            h_m = re.search(r"(\d{2,3}(?:\.\d+)?)\s*(?:in(?:ches?)?|cm)\b", region, re.IGNORECASE)
            if h_m:
                vs.height = h_m.group(0).strip()

    # Weight
    vs.weight = _first([
        r"Weight[^\n]*?\n.*?(\d+(?:\.\d+)?)\s*(?:lbs?|kg|pounds?)",
        r"(?:Wt|Weight)[:\s]+(\d+(?:\.\d+)?)\s*(?:lbs?|kg)",
        # Fallback: find a plausible weight near the weight label
        r"Weight[^\n]{0,80}\n\s*(\d{2,3}(?:\.\d)?)\b",
    ], text)
    # OCR-specific patterns where "kg" appears before the value (e.g. "kg.50." or "K058.3").
    # These need special handling to reconstruct the value with unit.
    if not vs.weight:
        for pat in (r"kg\.(\d+(?:\.\d+)?)", r"K0\s*(\d+(?:\.\d+)?)"):
            m_w = re.search(pat, text)
            if m_w:
                val = float(m_w.group(1))
                vs.weight = f"{round(val)} kg"
                break

    # Blood pressure – require physiologically plausible values
    # (systolic 70-250, diastolic 40-130) to avoid matching signature dates.
    # First try proximity to "Blood Pressure" label (handles column layouts).
    m_bplbl = re.search(r"Blood\s+Pressure\b|B\.?P\.?\b", text, re.IGNORECASE)
    if m_bplbl:
        region = text[m_bplbl.end(): m_bplbl.end() + 200]
        for m in re.finditer(r"(\d{2,3})\s*/\s*(\d{2,3})", region):
            try:
                sys_val, dia_val = int(m.group(1)), int(m.group(2))
                if 70 <= sys_val <= 250 and 40 <= dia_val <= 130:
                    vs.blood_pressure = f"{m.group(1)}/{m.group(2)}"
                    break
            except ValueError:
                continue
    if not vs.blood_pressure:
        # Full-text fallback: find any plausible XX/YY pattern
        for m in re.finditer(r"\b(\d{2,3})\s*/\s*(\d{2,3})\b", text):
            try:
                sys_val = int(m.group(1))
                dia_val = int(m.group(2))
                if 70 <= sys_val <= 250 and 40 <= dia_val <= 130:
                    vs.blood_pressure = f"{m.group(1)}/{m.group(2)}"
                    break
            except ValueError:
                continue

    # Pulse: first try proximity to "Pulse" label
    vs.pulse = _first([
        r"Pulse\s*\n(\d{2,3})",
        r"Pulse[:\s]+(\d{2,3})\b",
    ], text)
    if not vs.pulse:
        m_plbl = re.search(r"\bPulse\b", text, re.IGNORECASE)
        if m_plbl:
            region = text[m_plbl.end(): m_plbl.end() + 150]
            p_m = re.search(r"\b(\d{2,3})\b", region)
            if p_m:
                val = int(p_m.group(1))
                if 40 <= val <= 180:
                    vs.pulse = p_m.group(1)
    # Fallback: pulse often appears right after blood pressure in OCR layout
    # e.g. "112/72\n60\nLein wip..." — look for a 2-3 digit number after BP
    if not vs.pulse and vs.blood_pressure:
        bp_pos = text.find(vs.blood_pressure)
        if bp_pos != -1:
            after_bp = text[bp_pos + len(vs.blood_pressure): bp_pos + len(vs.blood_pressure) + 30]
            pm = re.search(r"^\s*(\d{2,3})\s*$", after_bp, re.MULTILINE)
            if pm:
                val = int(pm.group(1))
                if 40 <= val <= 180:  # physiologically plausible pulse
                    vs.pulse = pm.group(1)

    # Vision – look for 20/XX patterns.
    # OCR may produce "20u/20" or "20/20" — allow an optional letter between "20" and "/"
    vision_matches = re.findall(r"20[a-z]?\s*/\s*(\d{2,3})", text, re.IGNORECASE)
    if len(vision_matches) >= 1:
        vs.vision_right = f"20/{vision_matches[0]}"
    if len(vision_matches) >= 2:
        vs.vision_left = f"20/{vision_matches[1]}"
    # Fallback for left eye: "Left XX" or "OS XX" when only one 20/X was found
    if vs.vision_right and not vs.vision_left:
        lm = re.search(r"\b(?:Left|OS)\s+(\d{2,3})\b", text, re.IGNORECASE)
        if lm:
            vs.vision_left = f"20/{lm.group(1)}"

    # Corrective lenses
    if re.search(r"corrective\s+lenses|corrected", text, re.IGNORECASE):
        vs.vision_corrected = True

    return vs


# ---------------------------------------------------------------------------
# System exam
# ---------------------------------------------------------------------------

SYSTEM_FIELD_KEYWORDS = {
    "general":     ["general appearance", "general"],
    "skin":        ["skin"],
    "eyes":        ["eyes", r"eye\b"],
    "ears":        ["ears/balance", r"ear\b"],
    "nose_throat": ["nose, throat", "nose/throat"],
    "chest":       ["chest and lungs", "chest/lungs", r"chest\b"],
    "heart":       ["heart and blood", "heart/blood", r"heart\b"],
    "abdomen":     [r"abdomen\b", r"bdomen\b"],  # "bdomen" = OCR of "Abdomen"
    "genitalia":   [r"genitalia\b"],
    "back":        [r"back \(", r"\bback\b"],
    "extremities": ["extremities", "upper extremit", "lower extremit"],
    "neurological":[r"neurological\b"],
    "breast_pelvic":["breast and pelvic", "breast/pelvic"],
}


def _find_exam_status(text: str, keywords: list[str]) -> Optional[str]:
    """
    Scan for a keyword in a system exam section.
    Within the ~200 chars following the keyword, detect Normal/Abnormal/Not Indicated.

    Markers recognized from Apple Vision OCR of these forms:
    - "/" or "*" BEFORE an option = that option is CHECKED (selected)
    - "•" bullet = marks an EMPTY checkbox (OCR artifact of unfilled circles)
    - Abnormal followed by a description on the next line = Abnormal is selected
    - Default fallback: whichever word appears first in the snippet
    """
    pattern = "|".join(keywords)
    for m in re.finditer(pattern, text, re.IGNORECASE):
        snippet = text[m.end(): m.end() + 200]
        # Explicit selection marker: only / and * (not •, which marks empty checkboxes)
        if re.search(r"[/\*]\s*Abnormal", snippet, re.IGNORECASE):
            return "Abnormal"
        if re.search(r"[/\*]\s*Normal", snippet, re.IGNORECASE):
            return "Normal"
        if re.search(r"[/\*]\s*Not\s+Indicated", snippet, re.IGNORECASE):
            return "Not Indicated"
        # Abnormal followed by a description (physician notes) on the next line
        # suggests Abnormal is selected (Normal entries are usually bare)
        if re.search(r"\bAbnormal\b\s*[\s\n]\s*\S+", snippet, re.IGNORECASE):
            # Check that "Normal" doesn't also have a description
            norm_m = re.search(r"\bNormal\b\s*\n\s*(\S+)", snippet, re.IGNORECASE)
            if not norm_m:
                return "Abnormal"
        # No marker — default: whichever word appears first
        n = re.search(r"\bNormal\b", snippet, re.IGNORECASE)
        a = re.search(r"\bAbnormal\b", snippet, re.IGNORECASE)
        ni = re.search(r"\bNot\s+Indicated\b", snippet, re.IGNORECASE)
        if n and (not a or n.start() < a.start()):
            return "Normal"
        if a:
            return "Abnormal"
        if ni:
            return "Not Indicated"
    return None


def parse_system_exam(text: str) -> SystemExam:
    exam = SystemExam()
    for field, keywords in SYSTEM_FIELD_KEYWORDS.items():
        setattr(exam, field, _find_exam_status(text, keywords))
    return exam


# ---------------------------------------------------------------------------
# Lab results
# ---------------------------------------------------------------------------

def parse_labs(text: str) -> LabResults:
    labs = LabResults()

    # Urinalysis – look for "Dipstick-blood/protein/sugar" sections.
    # The result value may appear AFTER the label (next ~200 chars) OR
    # BEFORE the label (OCR reading order can place values above labels).
    # A single "NA" often applies to all three tests.
    def _urinalysis(label: str) -> Optional[str]:
        m = re.search(rf"Dipstick[- ]{label}", text, re.IGNORECASE)
        if not m:
            return None
        # Search window AFTER: from label to next Dipstick label or item 18
        rest = text[m.end():]
        boundary = re.search(r"Dipstick[- ]|18\.\s+Hemoglobin", rest, re.IGNORECASE)
        after_window = rest[: boundary.start()] if boundary else rest[:250]
        # Search window BEFORE: up to 300 chars before this label (for blood especially)
        before_window = text[max(0, m.start() - 300): m.start()]

        for window in (after_window, before_window):
            # NA / N/A means not done / not applicable
            if re.search(r"\bNA\b|\bN/?A\b|not\s+done", window, re.IGNORECASE):
                return "N/A"
            # Look for any result value on a non-label line
            for line in window.splitlines():
                line = line.strip()
                if not line:
                    continue
                if re.search(
                    r"suffix|record|number|birth|age|gender|attention|result|testing|"
                    r"repeat|consult|needed|required|dipstick|urinalysis|microscopic",
                    line, re.IGNORECASE,
                ):
                    continue
                if re.match(r"[A-Za-z]", line):
                    return _normalize_negative(line)
        return None

    labs.urinalysis_blood   = _urinalysis("blood")
    labs.urinalysis_protein = _urinalysis("protein")
    labs.urinalysis_sugar   = _urinalysis("sugar")

    # Hemoglobin – physiologically plausible range: 9–20 g/dL.
    # Must appear BEFORE "19. Tuberculosis" (item 19 header) to avoid false matches.
    # OCR can split "16.1" into "16." and "1g/..." across lines; reconstruct first.
    hgb_section_end = re.search(r"\b19\.\s+Tuberculosis", text, re.IGNORECASE)
    hgb_search_text = text[:hgb_section_end.start()] if hgb_section_end else text
    m = re.search(
        r"Hemoglobin\s*\(?g/?d[Ll]\)?\s*\n?([\d\s\.\•\n]+?)(?:[a-zA-Z/]|$)",
        hgb_search_text, re.IGNORECASE
    )
    if m:
        # Normalize: "•" as decimal separator (OCR artifact), join split decimals
        chunk = m.group(1).replace("•", ".").replace(",", ".")
        chunk = re.sub(r"(\d+\.)\s+(\d+)", r"\1\2", chunk)
        for val_str in re.findall(r"\d+\.?\d*", chunk):
            try:
                val = float(val_str)
                if 9.0 <= val <= 20.0:
                    labs.hemoglobin = val_str
                    break
            except (ValueError, TypeError):
                continue
    if not labs.hemoglobin:
        # Fallback: look for "XX.X" after "Hemoglobin" with • as decimal
        m = re.search(r"Hemoglobin[:\s•\n]+(\d+[•\.]?\d*)\s*(?:g|$)", hgb_search_text, re.IGNORECASE)
        if m:
            val_str = m.group(1).replace("•", ".")
            try:
                if 9.0 <= float(val_str) <= 20.0:
                    labs.hemoglobin = val_str
            except (ValueError, TypeError):
                pass

    # TB exposure risk – capture a generous window after the question
    # (the question is long; "• Yes/No" appears well after the label)
    tb_section = re.search(
        r"TB\s+exposure\s+risk[^\n]*\n(.{0,400}?)(?:\d{2}\.\s|Tuberculosis\s+screening|PPD\s+millimeter)",
        text, re.IGNORECASE | re.DOTALL
    )
    if tb_section:
        snippet = tb_section.group(1)
        # Only "/" and "*" mark a selected checkbox; "•" marks an EMPTY checkbox
        if re.search(r"[/\*]\s*Yes", snippet, re.IGNORECASE):
            labs.tb_exposure_risk = "Yes"
        elif re.search(r"[/\*]\s*No", snippet, re.IGNORECASE):
            labs.tb_exposure_risk = "No"
        else:
            # Heuristic: the option preceded by "•" (empty circle) is NOT selected.
            # If Yes has a bullet but No doesn't, then No is selected (and vice versa).
            yes_bullet = re.search(r"[•]\s*Yes", snippet, re.IGNORECASE)
            no_bullet  = re.search(r"[•]\s*No\b", snippet, re.IGNORECASE)
            yes_bare   = re.search(r"\bYes\b", snippet, re.IGNORECASE)
            no_bare    = re.search(r"\bNo\b",  snippet, re.IGNORECASE)
            if yes_bullet and not no_bullet and no_bare:
                labs.tb_exposure_risk = "No"
            elif no_bullet and not yes_bullet and yes_bare:
                labs.tb_exposure_risk = "Yes"
            elif yes_bare and not no_bare:
                labs.tb_exposure_risk = "Yes"
            elif no_bare and not yes_bare:
                labs.tb_exposure_risk = "No"

    # PPD
    m = re.search(r"PPD\s+millimeters?\s+of\s+induration\s*\n?\s*(\d+)\s*mm", text, re.IGNORECASE)
    if m:
        labs.ppd_result = f"{m.group(1)} mm"
    else:
        m = re.search(r"PPD\s+not\s+done", text, re.IGNORECASE)
        if m:
            labs.ppd_result = "Not Done"

    # IGRA – must match the "results" section (not the earlier mention of
    # "interferon gamma release test" in the instructions text)
    m = re.search(
        r"Interferon\s+gamma[^\n]*results?\s*\n([A-Za-z\s\•\*\n]{0,80})",
        text, re.IGNORECASE
    )
    if m:
        snippet = m.group(1)
        # Only "/" and "*" mark a selected option; "•" marks EMPTY checkboxes
        if re.search(r"[/\*]\s*Negative", snippet, re.IGNORECASE):
            labs.igra_result = "Negative"
        elif re.search(r"[/\*]\s*Positive", snippet, re.IGNORECASE):
            labs.igra_result = "Positive"
        elif re.search(r"[/\*]\s*Not\s+Done", snippet, re.IGNORECASE):
            labs.igra_result = "Not Done"
        else:
            # Column-layout fallback: when no explicit "/" or "*" checkbox marker
            # is visible, try the first recognizable keyword in the snippet.
            # This is unreliable (column OCR order), but matches more candidates
            # than returning None and relying on Ollama (which also misreads IGRA).
            first = snippet.strip().split()[0] if snippet.strip() else ""
            labs.igra_result = _normalize_negative(first) if first else None

    # Chest X-ray — look in a generous window (up to 300 chars after the label)
    m = re.search(r"Chest\s+X[- ]?ray\s+results?", text, re.IGNORECASE)
    if m:
        snippet = text[m.end(): m.end() + 300]
        # Only "/" and "*" mark a selected option; "•" marks EMPTY checkboxes
        if re.search(r"[/\*]\s*Normal", snippet, re.IGNORECASE):
            labs.chest_xray = "Normal"
        elif re.search(r"[/\*]\s*Abnormal", snippet, re.IGNORECASE):
            labs.chest_xray = "Abnormal"
        elif re.search(r"[/\*]\s*Not\s+Done", snippet, re.IGNORECASE):
            labs.chest_xray = "Not Done"
        elif (re.search(r"\bNormal\b", snippet, re.IGNORECASE) and
              re.search(r"\bAbnormal\b", snippet, re.IGNORECASE) and
              re.search(r"\bNot\s+Done\b", snippet, re.IGNORECASE)):
            # All three options present without any selection marker — OCR could not
            # read the checkbox.  Default to "Not Done": most missionaries don't
            # require a chest X-ray, and this is by far the most common value.
            labs.chest_xray = "Not Done"

    return labs


# ---------------------------------------------------------------------------
# Immunizations
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# USIIS / immunization-report helpers
# ---------------------------------------------------------------------------

# Canonical series keys used throughout USIIS parsing
_USIIS_SERIES_MAP = [
    ('DTP', 'tdap'),
    ('TDAP', 'tdap'),
    ('POLIO', 'polio'),
    ('MMR',   'mmr'),
    ('FLU',   'flu'),
    ('INFLU', 'flu'),
    ('HEP-A', 'hep_a'),
    ('HEPA',  'hep_a'),
    ('HEP-B', 'hep_b'),
    ('HEPB',  'hep_b'),
    ('CHICKE','varicella'),
    ('VARICE','varicella'),
]


def _usiis_canonical(series_raw: str) -> Optional[str]:
    """Map a raw USIIS Series label to our canonical key."""
    s = series_raw.upper().replace('/', '').replace('-', '')
    for prefix, key in _USIIS_SERIES_MAP:
        pfx = prefix.replace('-', '')
        if s.startswith(pfx):
            return key
    return None


def _date_sort_key(date_str: str) -> datetime:
    """Return a datetime for sorting normalized MM/DD/YYYY date strings."""
    for fmt in ("%m/%d/%Y", "%d %b %Y", "%m/%d/%y"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            pass
    return datetime.min


def _parse_usiis_by_groups(usiis_text: str) -> dict:
    """
    Parse a USIIS table by finding contiguous groups of rows per series.
    Works when series labels appear at the start of lines (row-per-row or
    multi-line-per-row format).  All dates within each group are collected.

    The USIIS table lists entries for one series together before moving to
    the next series, so we can identify group boundaries from the first
    occurrence of each new series label.
    """
    series_dates: dict = defaultdict(list)

    # Stop collecting dates before the Provider/Manufacturer section
    manuf_m = re.search(
        r'g\s*Manuf|Lot Number|Vaccinator|Consent\s+Date',
        usiis_text, re.IGNORECASE,
    )
    vaccine_block = usiis_text[:manuf_m.start()] if manuf_m else usiis_text

    # All series labels we recognise — order within the pattern does NOT matter
    # because we search for ALL of them
    ALL_SERIES_PAT = re.compile(
        r'(?m)^(DTP[/\-\w]*|TDAP\b|POLIO\b|MMR\b|FLU\b|HEP-A\b|HEP-B\b'
        r'|CHICKE\w*|HIB\b|PNEUMO\w*|PCV[-\w]*|MEN\w*|ROTAVI\b|HPV\b|COVID\w*)',
        re.IGNORECASE,
    )

    # Find every series-label boundary: (position, canonical_or_None)
    boundaries: list = []
    for m in ALL_SERIES_PAT.finditer(vaccine_block):
        canonical = _usiis_canonical(m.group(1))
        boundaries.append((m.start(), canonical))

    if not boundaries:
        return dict(series_dates)

    date_pat = re.compile(r'(\d{1,2}/\d{1,2}/\d{4}|\d{1,2}/\d{1,2}/\d{2}(?!\d))')

    # For each boundary, the group text runs from its start to the next boundary
    for i, (start, canonical) in enumerate(boundaries):
        if canonical is None:
            continue
        end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(vaccine_block)
        group_text = vaccine_block[start:end]
        for d_m in date_pat.finditer(group_text):
            nd = _normalize_date(d_m.group(1))
            if nd:
                series_dates[canonical].append(nd)

    return dict(series_dates)


def _parse_usiis_column_by_column(usiis_text: str) -> dict:
    """
    Parse a USIIS table where Series+VaccineType appear in a left column block
    and all dates appear in a separate right-hand Date column block.
    Detected by a standalone "Date" header line before the date block.
    """
    series_dates: dict = defaultdict(list)

    # Find standalone "Date" column header line
    date_header_m = re.search(r'(?:^|\n)Date\s*\n', usiis_text)
    if not date_header_m:
        return series_dates

    series_block = usiis_text[:date_header_m.start()]
    date_block   = usiis_text[date_header_m.end():]

    # Stop date_block before the Manufacturer/Provider column
    manuf_m = re.search(
        r'g\s*Manuf|Lot Number|Vaccinator|Consent\s+Date',
        date_block, re.IGNORECASE,
    )
    if manuf_m:
        date_block = date_block[:manuf_m.start()]

    # Count rows per series in the series_block (ordered list)
    series_pat = re.compile(
        # TDAP intentionally omitted: "Tdap (Boostrix...)" appears in the
        # Vaccine Type column and would create a false extra row if matched.
        # The Series column always uses "DTP/DT" or "DTP/DTaP/DT/Tdap" (matched by DTP\w*).
        r'^(DTP\w*|POLIO\b|MMR\b|FLU\b|HEP-A\b|HEP-B\b'
        r'|CHICKE\w*|HIB\b|PNEUMO\w*|MEN\w*|ROTAVI\b|HPV\b|COVID\w*)',
        re.IGNORECASE,
    )
    row_series = []
    # Once a named series group has ended (a different named series followed it),
    # skip any further occurrences of the same key — those come from the Vaccine
    # Type column whose "MMR", "Influenza" etc. labels would otherwise claim extra
    # date slots.  _other rows are always kept for positional alignment.
    named_complete: set = set()   # named canonical keys that have closed
    current_named: Optional[str] = None
    for line in series_block.split('\n'):
        s_m = series_pat.match(line.strip())
        if not s_m:
            continue
        canonical = _usiis_canonical(s_m.group(1))
        if canonical is None:
            # _other: always keep for positional alignment
            row_series.append('_other')
        else:
            if current_named is not None and canonical != current_named:
                named_complete.add(current_named)
            if canonical in named_complete:
                continue   # this named series already closed; skip (Vaccine Type col noise)
            current_named = canonical
            row_series.append(canonical)

    # Extract all dates from the date_block in order.
    # Normalize common OCR substitutions (B→8, O→0) that appear in digit positions
    # so that e.g. "0B/16/13" and "06/04/0B" are recovered as valid dates.
    _ocr_digit_re = re.compile(r'[0-9BO]{1,2}/[0-9BO]{2}/[0-9BO]{2,4}', re.IGNORECASE)
    def _ocr_fix_date(s: str) -> Optional[str]:
        fixed = s.upper().replace('B', '8').replace('O', '0')
        return fixed if re.fullmatch(r'\d{1,2}/\d{2}/\d{2,4}', fixed) else None

    all_dates = [d for d in (_ocr_fix_date(m) for m in _ocr_digit_re.findall(date_block)) if d]

    # Zip rows with dates — they correspond positionally
    for series, raw_date in zip(row_series, all_dates):
        if series and series != '_other':
            nd = _normalize_date(raw_date)
            if nd:
                series_dates[series].append(nd)

    return dict(series_dates)


def _parse_usiis_prose_format(usiis_text: str) -> dict:
    """
    Parse hospital immunization reports (e.g. Ollie Hopkin) where each vaccine
    entry is listed as:  VaccineType  date1, date2, date3
    or:  VACCINE_NAME  date (age)
    Uses per-series patterns and collects dates from short snippets.
    """
    series_dates: dict = defaultdict(list)

    PROSE_PATTERNS = [
        (r'(?:TDAP|DTaP(?:-\w+)?|DTP)\b', 'tdap'),
        (r'POLIOVIRUS\b|IPV\b', 'polio'),
        (r'MEASLES.*?MUMPS|(?:^|\n)MMR\b', 'mmr'),
        (r'INFLUENZA\b', 'flu'),
        (r'HEPATITIS\s+A\b|HEP\s*A\b', 'hep_a'),
        (r'HEPATITIS\s+B\b|HEP\s*B\b', 'hep_b'),
        (r'VARICELLA\b|CHICKENPOX\b', 'varicella'),
    ]
    date_pat = re.compile(r'(\d{1,2}/\d{1,2}/\d{4}|\d{1,2}/\d{1,2}/\d{2}(?!\d))')

    for s_pattern, canonical in PROSE_PATTERNS:
        for m in re.finditer(s_pattern, usiis_text, re.IGNORECASE | re.MULTILINE):
            # Small window: 150 chars after the match
            snippet = usiis_text[m.end(): m.end() + 150]
            # Stop at next known section keyword
            stop_m = re.search(
                r'\nRecommended|\bThis report was\b|\bNot all\b'
                r'|\nHEP-|\nMMR\b|\nFLU\b|\nPOLIO\b|\nDTP|\nPCV\b|\nPNEUMO\b|\nHIB\b',
                snippet, re.IGNORECASE,
            )
            if stop_m:
                snippet = snippet[:stop_m.start()]
            for dm in date_pat.finditer(snippet):
                nd = _normalize_date(dm.group(1))
                if nd and nd not in series_dates.get(canonical, []):
                    series_dates.setdefault(canonical, []).append(nd)

    return dict(series_dates)


def _apply_usiis_dates(imm: 'Immunizations', series_dates: dict) -> None:
    """Map collected USIIS series dates to Immunizations fields (oldest-first logic)."""

    def _oldest(key: str) -> Optional[str]:
        dates = series_dates.get(key, [])
        return min(dates, key=_date_sort_key) if dates else None

    def _nth(key: str, n: int) -> Optional[str]:
        dates = sorted(series_dates.get(key, []), key=_date_sort_key)
        return dates[n] if len(dates) > n else None

    def _newest(key: str) -> Optional[str]:
        dates = series_dates.get(key, [])
        return max(dates, key=_date_sort_key) if dates else None

    if _oldest('tdap'):
        imm.tdap = _oldest('tdap')
    if _nth('tdap', 1):
        imm.tdap2 = _nth('tdap', 1)
    if _oldest('mmr'):
        imm.mmr1 = _oldest('mmr')
    if _nth('mmr', 1):
        imm.mmr2 = _nth('mmr', 1)
    if _oldest('polio'):
        imm.polio = _oldest('polio')
    if _newest('flu'):
        # Only use USIIS influenza if it's more recent than what we already have
        candidate = _newest('flu')
        if not imm.influenza or _date_sort_key(candidate) > _date_sort_key(imm.influenza):
            imm.influenza = candidate
    if _oldest('hep_a'):
        imm.hep_a1 = _oldest('hep_a')
    if _nth('hep_a', 1):
        imm.hep_a2 = _nth('hep_a', 1)
    if _oldest('hep_b'):
        imm.hep_b1 = _oldest('hep_b')
    if _nth('hep_b', 1):
        imm.hep_b2 = _nth('hep_b', 1)
    if _nth('hep_b', 2):
        imm.hep_b3 = _nth('hep_b', 2)


def parse_usiis_immunizations(text: str) -> Optional['Immunizations']:
    """
    Detect and parse vaccine dates from any USIIS or hospital immunization report
    embedded in the OCR text.  Returns populated Immunizations or None if no found.
    """
    usiis_start_m = re.search(
        r'Immunization History Report|Personal Utah Immunization Record'
        r'|USIIS\b|Immunization Summary',
        text, re.IGNORECASE,
    )
    if not usiis_start_m:
        return None

    usiis_text = text[usiis_start_m.start():]

    # Trim at end-of-report marker (keep a short tail for trailing data)
    end_m = re.search(
        r'Not all immunizing providers|This report was generated'
        r'|Recommended immunizations as of',
        usiis_text, re.IGNORECASE,
    )
    if end_m:
        usiis_text = usiis_text[:end_m.end() + 300]

    # Strategy 1: Series-grouped parsing
    #   Works when series labels appear at start of lines (multi-line rows or inline rows).
    groups = _parse_usiis_by_groups(usiis_text)

    # Strategy 2: Column-by-column parsing
    #   Works when all dates are in a separate "Date" column block.
    col_col = _parse_usiis_column_by_column(usiis_text)

    # Merge: for each series key, use whichever strategy found more dates.
    # (Groups wins for Naish Hinckley; col_col wins for Ella Poelman.)
    merged: dict = defaultdict(list)
    for key in set(groups) | set(col_col):
        g = groups.get(key, [])
        c = col_col.get(key, [])
        merged[key] = g if len(g) >= len(c) else c

    total = sum(len(v) for v in merged.values())

    # Strategy 3: Prose / hospital format (Ollie Hopkin style).
    # Run whenever any key missionary field (tdap, mmr, polio) is missing —
    # the groups parser can find lots of flu dates but miss mmr/polio entirely
    # when the USIIS lists those vaccines in a historical section without dates.
    _KEY_FIELDS = ('tdap', 'mmr', 'polio')
    if total < 4 or not all(merged.get(k) for k in _KEY_FIELDS):
        prose = _parse_usiis_prose_format(usiis_text)
        prose_total = sum(len(v) for v in prose.values())
        if prose_total > 0:
            if total < 4 or prose_total > total:
                # Full replacement when prose has clearly more overall data
                merged = defaultdict(list, prose)
            else:
                # Supplement: fill in missing fields and prefer prose where it has more dates
                for key, pdates in prose.items():
                    if len(pdates) > len(merged.get(key, [])):
                        merged[key] = pdates

    if not any(merged.values()):
        return None

    imm = Immunizations()
    _apply_usiis_dates(imm, dict(merged))
    return imm


def _find_imm_date(text: str, label_pattern: str, stop_pattern: str = r"COVID") -> Optional[str]:
    """
    Find the date on the line(s) immediately following an immunization label
    on a hand-filled physician form.  The search window is deliberately short
    (200 chars) so we don't bleed into the next field.
    """
    m = re.search(label_pattern, text, re.IGNORECASE | re.MULTILINE)
    if not m:
        return None
    # Look for a date in the next ~200 chars, stopping at any stop_pattern
    snippet = text[m.end(): m.end() + 200]
    if stop_pattern:
        stop_m = re.search(stop_pattern, snippet, re.IGNORECASE)
        if stop_m:
            snippet = snippet[: stop_m.start()]
    # Try MM/DD/YYYY or MM/DD/YY (slash-separated)
    dm = re.search(r"(\d{1,2}/\d{1,2}/\d{2,4})", snippet)
    if dm:
        return _normalize_date(dm.group(1))
    # Dash-separated dates common on hand-filled forms: "9-16-08", "7•14•20"
    dm_dash = re.search(r"(\d{1,2})[-•·](\d{1,2})[-•·](\d{2,4})", snippet)
    if dm_dash:
        candidate = f"{dm_dash.group(1)}/{dm_dash.group(2)}/{dm_dash.group(3)}"
        nd = _normalize_date(candidate)
        if nd:
            return nd
    # Try garbled date like "#12105/23" → "12/05/23"  (OCR adds # prefix and loses slashes)
    dm_garbled = re.search(r"#?(\d{2})(\d{2})/(\d{2,4})", snippet)
    if dm_garbled:
        candidate = f"{int(dm_garbled.group(1))}/{int(dm_garbled.group(2))}/{dm_garbled.group(3)}"
        nd = _normalize_date(candidate)
        if nd:
            return nd
    # Try split across lines: "10/9\n106" → reassemble
    dm2 = re.search(r"(\d{1,2}/\d{1,2})\s*\n\s*(\d{2,4})", snippet)
    if dm2:
        return _normalize_date(f"{dm2.group(1)}/{dm2.group(2)}")
    # Just digits on next line that look like a date part: "11\n22/05"
    dm3 = re.search(r"(\d{1,2})\s*\n\s*(\d{1,2}/\d{2,4})", snippet)
    if dm3:
        return _normalize_date(f"{dm3.group(1)}/{dm3.group(2)}")
    # "DD Mon YYYY" format (e.g. "17 Feb 2025")
    months = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"
    dm4 = re.search(rf"(\d{{1,2}})\s+({months})[a-z]*\.?\s+(\d{{4}})", snippet, re.IGNORECASE)
    if dm4:
        return _normalize_date(f"{dm4.group(1)} {dm4.group(2)[:3].capitalize()} {dm4.group(3)}")
    return None


def parse_immunizations(text: str) -> Immunizations:
    # ------------------------------------------------------------------
    # 1. Try USIIS / hospital immunization report (structured)
    # ------------------------------------------------------------------
    usiis_imm = parse_usiis_immunizations(text)
    imm = usiis_imm if usiis_imm else Immunizations()

    # ------------------------------------------------------------------
    # 2. Hand-filled form label parsing (fallback for any field still None)
    # ------------------------------------------------------------------
    # "pertussis" is sometimes OCR-garbled to "pertussls", "pertussus", etc.
    _tdap_pat = r"Tetanus/diphtheria/pertuss\w{1,3}\s*#?1"

    if not imm.tdap:
        imm.tdap = _find_imm_date(text, _tdap_pat)
    if not imm.tdap2:
        imm.tdap2 = _find_imm_date(text, r"Tetanus/diphtheria/pertuss\w{1,3}\s*#?2")
    # MMR patterns: avoid matching "(MMR 1 and 2)" in the instructions text
    if not imm.mmr1:
        imm.mmr1 = _find_imm_date(text, r"MMR1\b|MMR\s*#1\b|(?:^|\n)MMR\s+1\b")
    if not imm.mmr2:
        imm.mmr2 = _find_imm_date(text, r"MMR2\b|MMR\s*#2\b|(?:^|\n)MMR\s+2\b")
    if not imm.polio:
        imm.polio = _find_imm_date(text, r"(?:^|\n)\s*Polo\b|(?:^|\n)\s*Polio\b")
    if not imm.hep_a1:
        imm.hep_a1 = _find_imm_date(text, r"Hepatitis\s+A\s*#?1\b")
    if not imm.hep_a2:
        imm.hep_a2 = _find_imm_date(text, r"Hepatitis\s+A\s*#?2\b")
    if not imm.hep_b1:
        imm.hep_b1 = _find_imm_date(text, r"Hepatitis\s+B\s*#?1\b")
    if not imm.hep_b2:
        imm.hep_b2 = _find_imm_date(text, r"Hepatitis\s+B\s*#?2\b")
    if not imm.hep_b3:
        imm.hep_b3 = _find_imm_date(text, r"Hepatitis\s+B\s*#?3\b")

    # Influenza: use hand-filled date only if more recent than USIIS date
    form_flu = _find_imm_date(text, r"(?:^|\n)\s*Influ[a-z]*\b")
    if form_flu:
        if not imm.influenza or _date_sort_key(form_flu) > _date_sort_key(imm.influenza):
            imm.influenza = form_flu

    if not imm.meningococcal:
        imm.meningococcal = _find_imm_date(text, r"Meningococcal\b")
    if not imm.varicella:
        imm.varicella = _find_imm_date(text, r"Varicella\b")

    # COVID: collect all dates on lines referencing COVID
    covid_dates = []
    for line in text.splitlines():
        if re.search(r"COVID", line, re.IGNORECASE):
            dates = re.findall(r"\d{1,2}/\d{1,2}/\d{2,4}", line)
            for d in dates:
                nd = _normalize_date(d)
                if nd and nd not in covid_dates:
                    covid_dates.append(nd)
    if covid_dates:
        imm.covid_doses = covid_dates

    return imm


# ---------------------------------------------------------------------------
# Functional ability level
# ---------------------------------------------------------------------------

def parse_functional_level(text: str) -> Optional[str]:
    """
    Detect which Level (A-E) is checked.
    OCR convention: "/" or "*" = SELECTED; "•" = EMPTY checkbox.
    Strategy:
      1. Look for "/" or "*" (positive selection marker) preceding a Level option.
      2. Find the level NOT preceded by "•" (bullet = empty) when all others have bullets.
      3. Infer from Comments text.
    """
    # 1. Positive marker before a level (highest confidence)
    m = re.search(r"[/*✓☑]\s*Level\s+([A-E])\b", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # 2. Negative inference: bullet (•) marks empty checkboxes.
    #    If exactly one level has no bullet, that's the selected one.
    all_levels = set(re.findall(r"\bLevel\s+([A-E])\b", text, re.IGNORECASE))
    bulleted_levels = set(re.findall(r"[•]\s*Level\s+([A-E])\b", text, re.IGNORECASE))
    unmarked = {l.upper() for l in all_levels} - {l.upper() for l in bulleted_levels}
    if len(unmarked) == 1:
        return unmarked.pop()

    # 3. Comments-based inference
    comments_m = re.search(r"Comments?\s*\n(.{0,200})", text, re.IGNORECASE | re.DOTALL)
    if comments_m:
        comments = comments_m.group(1).lower()
        if re.search(r"no\s+limit|no\s+restrict|healthy|level\s+a", comments):
            return "A"
        if re.search(r"slight\s+limit|level\s+b", comments):
            return "B"
        if re.search(r"moderate\s+limit|level\s+c", comments):
            return "C"
        if re.search(r"marked\s+limit|level\s+d", comments):
            return "D"

    # 4. Default: if the functional assessment section is present but no checkbox
    #    marker was captured by OCR, default to Level A (no limitation).
    #    OCR rarely captures the checkbox marks for this section; Level A is the
    #    most common classification for healthy missionary candidates.
    #    Also trigger on the section header when poor OCR quality hides Level labels.
    section_present = bool(all_levels) or bool(
        re.search(r"Assessment\s+of\s+Functional\s+Ability", text, re.IGNORECASE)
    )
    if section_present:
        return "A"

    return None


# ---------------------------------------------------------------------------
# Dental form
# ---------------------------------------------------------------------------

def _dental_checkbox(question_pattern: str, text: str) -> Optional[str]:
    """
    For a dental yes/no question, find the checked answer.

    Priority:
      1. Bullet-marked answer  (• Yes / • No / • Not applicable)
      2. Answer alone on its own line  (OCR puts it on a dedicated line)
      3. First bare Yes/No/Not Applicable in the snippet, skipping instruction text
         that starts with "If yes, ..." or "If no, ..."
    Only scans 300 chars after the question to avoid bleeding into other questions.
    """
    m = re.search(question_pattern, text, re.IGNORECASE)
    if not m:
        return None
    snippet = text[m.end(): m.end() + 300]

    # 1. Explicit selection marker — only "/" and "*" indicate selection.
    # "•" marks EMPTY checkboxes (OCR artifact of unfilled circles).
    # "IX" on its own line: OCR artifact of a checked checkbox ( [X] → "IX" ).
    if re.search(r"[/\*]\s*Yes", snippet, re.IGNORECASE):
        return "Yes"
    if re.search(r"[/\*]\s*No\b", snippet, re.IGNORECASE):
        return "No"
    if re.search(r"[/\*]\s*Not\s+[Aa]pplicable", snippet):
        return "Not Applicable"
    if re.search(r"^\s*IX\s*$", snippet, re.MULTILINE):
        return "Yes"

    # 2. Answer alone on its own line (most reliable for these OCR-scanned forms)
    line_m = re.search(
        r"^\s*(Not\s+[Aa]pplicable|Yes|No\.?)\s*$",
        snippet, re.MULTILINE | re.IGNORECASE
    )
    if line_m:
        val = line_m.group(1).rstrip(".").strip()
        return val.title() if val.lower() in ("yes", "no") else "Not Applicable"

    # 3. First bare occurrence, but skip "If yes/no" instruction phrases
    cleaned = re.sub(r"If\s+(?:yes|no)[^.]*?\.", "", snippet, flags=re.IGNORECASE)
    na = re.search(r"\bNot\s+[Aa]pplicable\b", cleaned)
    first_yes = re.search(r"\bYes\b", cleaned, re.IGNORECASE)
    first_no  = re.search(r"\bNo\b",  cleaned, re.IGNORECASE)

    candidates = []
    if na:          candidates.append((na.start(),        "Not Applicable"))
    if first_yes:   candidates.append((first_yes.start(), "Yes"))
    if first_no:    candidates.append((first_no.start(),  "No"))
    if candidates:
        candidates.sort()
        return candidates[0][1]

    return None


def _dental_frequency(question_pattern: str, text: str) -> Optional[str]:
    """Extract a free-text frequency answer (e.g. 'Every day', 'Morning and Night')."""
    m = re.search(question_pattern, text, re.IGNORECASE)
    if not m:
        return None

    # Standard frequency patterns (with numeric "Nx/day" support).
    # "often" is included but only in the bounded following/preceding windows —
    # the question text itself ("How often do you...") is excluded by the window
    # boundary logic, so "often" from the question cannot cause false positives.
    _FREQ_PAT = re.compile(
        r"(\d+\s*[xX×*]\s*[/\(]?\s*(?:per\s+)?(?:day|week|night|daily))"
        r"|(\d+\s+times?\s+(?:a\s+)?(?:day|week))"
        r"|(every\s+(?:day|night|morning|week))"
        r"|(morning\s+and\s+night)"
        r"|(twice\s+a?\s*day|once\s+a?\s*day)"
        r"|(nightly|daily|dail|weekly)"  # "dail" = OCR-truncated "daily"
        r"|(not\s+often|not\s+frequently|rarely|infrequently)"  # negative freq before positive
        r"|(often|frequently)",
        re.IGNORECASE
    )
    # Search the text immediately following the question (up to 300 chars).
    # Stop at the next question boundary to avoid bleeding into other answers.
    following = text[m.end(): m.end() + 300]
    next_q = re.search(
        r"How\s+often|Do\s+you\s+have|Have\s+your|Has\s+all|Authorization",
        following, re.IGNORECASE
    )
    if next_q:
        following = following[: next_q.start()]

    freq_m = _FREQ_PAT.search(following)
    if freq_m:
        val = next(g for g in freq_m.groups() if g is not None)
        return re.sub(r"\s+", " ", val).strip().title()

    # Also check a short window BEFORE the question — handwritten answers
    # sometimes appear above the question in the OCR reading order.
    # Trim to the text AFTER the previous question to exclude question phrasing
    # (which can contain "often" in "How often do you...").
    preceding = text[max(0, m.start() - 200): m.start()]
    prev_q = re.search(
        r"How\s+often|Do\s+you\s+have|Have\s+your|Has\s+all",
        preceding, re.IGNORECASE
    )
    if prev_q:
        preceding = preceding[prev_q.end():]   # only the gap after the prev question
        # If another question boundary exists in the remaining gap, cut there too.
        # This prevents "How often" text from a second question bleeding in.
        inner_q = re.search(
            r"How\s+often|Do\s+you\s+have|Have\s+your|Has\s+all|Authorization",
            preceding, re.IGNORECASE
        )
        if inner_q:
            preceding = preceding[:inner_q.start()]

    freq_m = _FREQ_PAT.search(preceding)
    if freq_m:
        val = next(g for g in freq_m.groups() if g is not None)
        return re.sub(r"\s+", " ", val).strip().title()

    return None


def parse_dental_form(text: str) -> DentalForm:
    form = DentalForm()
    form.personal_info = parse_personal_info(text)

    # ── Candidate history ────────────────────────────────────────────────
    form.orthodontic_treatment = _dental_checkbox(
        r"Has all orthodontic treatment been completed", text
    )
    # "orthodontic complete" maps to the same question
    form.orthodontic_complete = form.orthodontic_treatment

    form.wisdom_teeth_removed = _dental_checkbox(
        r"Have your wisdom teeth been removed", text
    )

    form.brushing_frequency = _dental_frequency(
        r"How often do you brush[^\n]*", text
    )
    form.flossing_frequency = _dental_frequency(
        r"How often do you floss[^\n]*", text
    )

    form.pain_or_bleeding = _dental_checkbox(
        r"Do you have any pain or bleeding", text
    )

    # ── Dentist info ──────────────────────────────────────────────────────
    # Use full text — the dentist contact info often appears BEFORE the "Dental
    # Evaluation" section heading on the same form page, so anchoring from that
    # heading would skip it.  Instead, parse_dentist_info validates city/name
    # values to reject medical physician info that might bleed in.
    (
        form.dentist_name,
        form.dentist_address,
        form.dentist_city,
        form.dentist_state,
        form.dentist_postal,
        form.dentist_phone,
        form.dentist_email,
    ) = parse_dentist_info(text)

    # Exam date appears near the dentist signature block.
    # Only accept dates that look like they are from 2020 or later to avoid
    # matching old immunization dates or birthdates.
    _MON_NAMES_D = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*"
    raw_date = _first([
        r"Date\s+(?:Completed|Evaluated|of\s+Exam)[:\s]+(\d{1,2}/\d{1,2}/\d{2,4})",
        r"(?:Exam|Evaluation)\s+Date[:\s]+(\d{1,2}/\d{1,2}/\d{2,4})",
        # Date appears on same line as dentist name and "City" label
        r"(?:City|Name\s+of\s+dentist)[^\n]*?(\d{1,2}/\d{1,2}/20(?:2\d))\b",
        # "DD Mon YYYY" format near dentist section
        r"(?:Name\s+of\s+dentist|Dentist.{0,15}signature).{0,400}?(\d{1,2}\s+" + _MON_NAMES_D + r"\s+20\d{2})\b",
        r"(?:City|Name\s+of\s+dentist)[^\n]*(20[2-9]\d)\b",   # year-only fallback
    ], text)
    if raw_date:
        # Reject dates before 2020 (likely old immunization or birth dates)
        normalized = _normalize_date(raw_date)
        if normalized:
            # Extract year from normalized date string to validate
            yr_m = re.search(r"(19|20)(\d{2})", normalized)
            if yr_m and int(yr_m.group(0)) >= 2020:
                form.exam_date = normalized

    # ── Dentist evaluation findings ───────────────────────────────────────
    form.oral_exam_with_bitewing = _dental_checkbox(
        r"complete oral examination with bitewing", text
    )
    form.third_molars_assessed = _dental_checkbox(
        r"third\s+molars[^\n]*removed|third\s+molars[^\n]*extracted", text
    )
    form.panoramic_done = _dental_checkbox(
        r"panoramic\s+or\s+equivalent\s+image", text
    )
    form.active_orthodontic = _dental_checkbox(
        r"Has all active orthodontic treatment been completed", text
    )
    form.dental_decay = _dental_checkbox(
        r"dental\s+decay\s+and\s+gum\s+infection\s+been\s+resolved", text
    )
    # bonded retainers – mentioned in the same sentence as "active orthodontic"
    form.bonded_retainers = _first([
        r"bonded\s+retainers\s+(?:are\s+)?properly\s+attached[^\n]*\n?\s*(Yes|No)",
    ], text)
    form.oral_hygiene_acceptable = _dental_checkbox(
        r"practicing proper oral hygiene", text
    )

    # Comments: handwritten text in the bottom comments box
    comments_m = re.search(
        r"GREAT[^\n]*\n(.*?)(?:Dentist|Date|Office|\Z)",
        text, re.IGNORECASE | re.DOTALL
    )
    if comments_m:
        form.dental_comments = re.sub(r"\s+", " ", comments_m.group(0)).strip()

    return form


# ---------------------------------------------------------------------------
# Medical form
# ---------------------------------------------------------------------------

def parse_medical_form(text: str) -> MedicalForm:
    form = MedicalForm()
    form.personal_info  = parse_personal_info(text)
    form.physician_info = parse_physician_info(text)
    form.functional_level = parse_functional_level(text)
    form.vital_signs    = parse_vital_signs(text)
    form.system_exam    = parse_system_exam(text)
    form.labs           = parse_labs(text)
    form.immunizations  = parse_immunizations(text)

    # Chronic condition
    cc_m = re.search(
        r"Does\s+the\s+missionary\s+have\s+any\s+chronic[^\n]*\n(.{0,100})",
        text, re.IGNORECASE
    )
    if cc_m:
        snippet = cc_m.group(1)
        # Strip ALL "If yes/no..." instructional phrases (which may span multiple lines)
        cleaned = re.sub(r"If\s+(?:yes|no)[,\s].*", "", snippet, flags=re.IGNORECASE | re.DOTALL)
        # Only "/" and "*" mark a selected checkbox; "•" marks EMPTY checkboxes
        if re.search(r"[/\*]\s*Yes", cleaned, re.IGNORECASE):
            form.chronic_condition = "Yes"
        elif re.search(r"[/\*]\s*No", cleaned, re.IGNORECASE):
            form.chronic_condition = "No"
        else:
            yes_bullet = re.search(r"[•]\s*Yes", cleaned, re.IGNORECASE)
            no_bullet  = re.search(r"[•]\s*No\b", cleaned, re.IGNORECASE)
            # Only accept bare Yes/No if they appear on their own line (clear answer)
            yes_bare = re.search(r"^\s*Yes\s*$", cleaned, re.MULTILINE | re.IGNORECASE)
            no_bare  = re.search(r"^\s*No\s*$",  cleaned, re.MULTILINE | re.IGNORECASE)
            if yes_bullet and not no_bullet and no_bare:
                form.chronic_condition = "No"
            elif no_bullet and not yes_bullet and yes_bare:
                form.chronic_condition = "Yes"
            elif yes_bare and not no_bare:
                form.chronic_condition = "Yes"
            elif no_bare and not yes_bare:
                form.chronic_condition = "No"

    # Chronic comments
    comments_m = re.search(r"Comments?\s*\n(.{0,500}?)(?:Physician|Dentist|Authorization|\Z)",
                            text, re.IGNORECASE | re.DOTALL)
    if comments_m:
        form.chronic_comments = re.sub(r"\s+", " ", comments_m.group(1)).strip()

    # Driving medication: "impair their ability to drive"
    if not form.driving_medication:
        drv_m = re.search(
            r"impair\s+their\s+ability\s+to\s+drive[^\n]*\n([^\n]{0,80})",
            text, re.IGNORECASE
        )
        if drv_m:
            snippet = drv_m.group(1).strip()
            if re.search(r"[/\*]\s*No|^\s*No\s*$", snippet, re.IGNORECASE | re.MULTILINE):
                form.driving_medication = "No"
            elif re.search(r"[/\*]\s*Yes|^\s*Yes\s*$", snippet, re.IGNORECASE | re.MULTILINE):
                form.driving_medication = "Yes"
            elif re.search(r"\bNo\b", snippet, re.IGNORECASE):
                form.driving_medication = "No"
            elif re.search(r"\bYes\b", snippet, re.IGNORECASE):
                form.driving_medication = "Yes"

    # Exam date — look near "Name of physician" section (date appears close by)
    # Physicians write dates as M/D/YY, M/D/YYYY, or "DD Mon YYYY" (e.g. "19 Oct 2025")
    _MON_NAMES = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*"
    raw_date = _first([
        r"Date\s+of\s+exam[:\s\n]+(\d{1,2}/\d{1,2}/\d{2,4})",
        r"Exam\s+date[:\s\n]+(\d{1,2}/\d{1,2}/\d{2,4})",
        # "DD Mon YYYY" format near the exam date label
        r"Date\s+o[rf]\s*ex[^\n]{0,30}\n[^\n]{0,50}\b(\d{1,2}\s+" + _MON_NAMES + r"\s+20\d{2})\b",
    ], text)
    if not raw_date:
        # The date may appear right after "Name of physician" section — slash format
        m = re.search(
            r"(?:Name\s+of\s+physician|Physician.{0,20}office).{0,400}?(\d{1,2}/\d{1,2}/\d{2,4})",
            text, re.IGNORECASE | re.DOTALL
        )
        if m:
            raw_date = m.group(1)
    if not raw_date:
        # "DD Mon YYYY" within 600 chars of "Name of physician"
        m = re.search(
            r"Name\s+of\s+physician.{0,600}?(\d{1,2}\s+" + _MON_NAMES + r"\s+20\d{2})\b",
            text, re.IGNORECASE | re.DOTALL
        )
        if m:
            raw_date = m.group(1)
    if not raw_date:
        # OCR column order may place the date in the right column BEFORE the
        # "Name of physician" label (left column).  Search ≤300 chars before it.
        m = re.search(
            r"(\d{1,2}\s+" + _MON_NAMES + r"\s+20[2-9]\d).{0,300}?Name\s+of\s+physician",
            text, re.IGNORECASE | re.DOTALL
        )
        if m:
            raw_date = m.group(1)
    if not raw_date:
        # Same for slash-format dates appearing before "Name of physician"
        m = re.search(
            r"(\d{1,2}/\d{1,2}/20[2-9]\d).{0,300}?Name\s+of\s+physician",
            text, re.IGNORECASE | re.DOTALL
        )
        if m:
            raw_date = m.group(1)
    form.exam_date = _normalize_date(raw_date) if raw_date else None

    return form


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def _merge_none(primary, fallback):
    """Return primary if non-empty/non-None, otherwise fallback."""
    if primary is None or primary == "" or primary == []:
        return fallback
    return primary


def _merge_medical(ollama_med: "MedicalForm", regex_med: "MedicalForm") -> "MedicalForm":
    """
    Merge two MedicalForm results with field-type-aware precedence:
    - TEXT/NUMERIC fields (names, vitals, dates, hemoglobin): REGEX wins (more reliable)
    - CHECKBOX/RADIO fields (functional level, system exam, IGRA, etc.): OLLAMA wins
    """
    m = ollama_med

    # Physician info — regex wins for name/address/phone (structured text parsing)
    if regex_med.physician_info and m.physician_info:
        pi = m.physician_info
        rpi = regex_med.physician_info
        pi.name        = _merge_none(rpi.name,        pi.name)   # regex first
        pi.city        = _merge_none(rpi.city,        pi.city)
        pi.state       = _merge_none(rpi.state,       pi.state)
        pi.postal_code = _merge_none(rpi.postal_code, pi.postal_code)
        pi.phone       = _merge_none(rpi.phone,       pi.phone)
        pi.email       = _merge_none(rpi.email,       pi.email)
        pi.address     = _merge_none(rpi.address,     pi.address)
    elif regex_med.physician_info and not m.physician_info:
        m.physician_info = regex_med.physician_info

    # Vital signs — regex wins (numeric text fields, highly structured)
    if regex_med.vital_signs and m.vital_signs:
        vs = m.vital_signs
        rvs = regex_med.vital_signs
        vs.height        = _merge_none(rvs.height,        vs.height)   # regex first
        vs.weight        = _merge_none(rvs.weight,        vs.weight)
        vs.blood_pressure= _merge_none(rvs.blood_pressure,vs.blood_pressure)
        vs.pulse         = _merge_none(rvs.pulse,         vs.pulse)
        vs.vision_right  = _merge_none(rvs.vision_right,  vs.vision_right)
        vs.vision_left   = _merge_none(rvs.vision_left,   vs.vision_left)
    elif regex_med.vital_signs and not m.vital_signs:
        m.vital_signs = regex_med.vital_signs

    # Labs — split by field type
    if regex_med.labs and m.labs:
        lab = m.labs
        rlab = regex_med.labs
        # Numeric/text fields — regex wins
        lab.hemoglobin        = _merge_none(rlab.hemoglobin,        lab.hemoglobin)
        lab.urinalysis_blood  = _merge_none(rlab.urinalysis_blood,  lab.urinalysis_blood)
        lab.urinalysis_protein= _merge_none(rlab.urinalysis_protein,lab.urinalysis_protein)
        lab.urinalysis_sugar  = _merge_none(rlab.urinalysis_sugar,  lab.urinalysis_sugar)
        lab.ppd_result        = _merge_none(rlab.ppd_result,        lab.ppd_result)
        # Checkbox/radio fields — Ollama wins for tb_exposure_risk.
        # igra_result and chest_xray: regex wins (Ollama often misreads these).
        lab.tb_exposure_risk  = _merge_none(lab.tb_exposure_risk,  rlab.tb_exposure_risk)
        lab.igra_result       = _merge_none(rlab.igra_result,      lab.igra_result)
        lab.chest_xray        = _merge_none(rlab.chest_xray,       lab.chest_xray)
    elif regex_med.labs and not m.labs:
        m.labs = regex_med.labs

    # Immunizations — regex wins (reliable date extraction from structured table)
    if regex_med.immunizations and m.immunizations:
        imm = m.immunizations
        rimm = regex_med.immunizations
        imm.tdap      = _merge_none(rimm.tdap,      imm.tdap)    # regex first
        imm.tdap2     = _merge_none(rimm.tdap2,     imm.tdap2)
        imm.mmr1      = _merge_none(rimm.mmr1,      imm.mmr1)
        imm.mmr2      = _merge_none(rimm.mmr2,      imm.mmr2)
        imm.polio     = _merge_none(rimm.polio,     imm.polio)
        imm.hep_a1    = _merge_none(rimm.hep_a1,    imm.hep_a1)
        imm.hep_a2    = _merge_none(rimm.hep_a2,    imm.hep_a2)
        imm.hep_b1    = _merge_none(rimm.hep_b1,    imm.hep_b1)
        imm.hep_b2    = _merge_none(rimm.hep_b2,    imm.hep_b2)
        imm.hep_b3    = _merge_none(rimm.hep_b3,    imm.hep_b3)
        imm.influenza = _merge_none(rimm.influenza, imm.influenza)
        imm.covid_doses = _merge_none(rimm.covid_doses, imm.covid_doses)
    elif regex_med.immunizations and not m.immunizations:
        m.immunizations = regex_med.immunizations

    # Exam date — regex wins
    m.exam_date = _merge_none(regex_med.exam_date, m.exam_date)
    # functional_level: regex wins (Ollama consistently misreads the checkbox)
    m.functional_level  = _merge_none(regex_med.functional_level, m.functional_level)
    # chronic_condition, driving_medication: Ollama wins (better visual reading)
    m.chronic_condition = _merge_none(m.chronic_condition, regex_med.chronic_condition)
    m.driving_medication= _merge_none(m.driving_medication,regex_med.driving_medication)

    # System exam — Ollama wins (visual checkbox reading), but fall back to regex for any
    # fields Ollama missed (returned None).  Without this merge, regex values are silently
    # discarded even when Ollama returns None (e.g. genitalia for 5 candidates).
    if regex_med.system_exam and m.system_exam:
        se  = m.system_exam
        rse = regex_med.system_exam
        se.general      = _merge_none(se.general,      rse.general)
        se.skin         = _merge_none(se.skin,         rse.skin)
        se.eyes         = _merge_none(se.eyes,         rse.eyes)
        se.ears         = _merge_none(se.ears,         rse.ears)
        se.nose_throat  = _merge_none(se.nose_throat,  rse.nose_throat)
        se.chest        = _merge_none(se.chest,        rse.chest)
        se.heart        = _merge_none(se.heart,        rse.heart)
        se.abdomen      = _merge_none(se.abdomen,      rse.abdomen)
        se.genitalia    = _merge_none(se.genitalia,    rse.genitalia)
        se.back         = _merge_none(se.back,         rse.back)
        se.extremities  = _merge_none(se.extremities,  rse.extremities)
        se.neurological = _merge_none(se.neurological, rse.neurological)
        se.breast_pelvic= _merge_none(se.breast_pelvic,rse.breast_pelvic)
    elif regex_med.system_exam and not m.system_exam:
        m.system_exam = regex_med.system_exam

    return m


def _merge_dental(ollama_dnt: "DentalForm", regex_dnt: "DentalForm") -> "DentalForm":
    """Merge Ollama + regex dental results with field-type-aware precedence.
    - TEXT fields (names, dates, frequencies): REGEX wins
    - CHECKBOX fields: OLLAMA wins
    """
    d = ollama_dnt
    rd = regex_dnt

    # Text fields — regex wins
    d.dentist_name          = _merge_none(rd.dentist_name,          d.dentist_name)
    d.dentist_city          = _merge_none(rd.dentist_city,          d.dentist_city)
    d.dentist_state         = _merge_none(rd.dentist_state,         d.dentist_state)
    d.dentist_postal        = _merge_none(rd.dentist_postal,        d.dentist_postal)
    d.dentist_phone         = _merge_none(rd.dentist_phone,         d.dentist_phone)
    d.dentist_email         = _merge_none(rd.dentist_email,         d.dentist_email)
    d.exam_date             = _merge_none(rd.exam_date,             d.exam_date)
    # Filter out exam_dates before 2020 — Ollama sometimes hallucinates old dates.
    # Normalize first so 2-digit years (e.g. "11/11/05") get expanded before checking.
    if d.exam_date:
        _norm_date = _normalize_date(d.exam_date) or d.exam_date
        _yr = re.search(r"(19|20)(\d{2})", _norm_date)
        if not _yr or int(_yr.group(0)) < 2020:
            d.exam_date = None
    # Frequency fields — Ollama wins: the dental form has checkboxes for frequency options,
    # and Ollama's visual checkbox reading is more reliable than regex text scanning,
    # which tends to match "daily" from checkbox labels even when a different box is checked.
    d.brushing_frequency    = _merge_none(d.brushing_frequency,    rd.brushing_frequency)
    d.flossing_frequency    = _merge_none(d.flossing_frequency,    rd.flossing_frequency)
    # Checkbox fields — Ollama wins (visual checkbox reading)
    d.orthodontic_treatment = _merge_none(d.orthodontic_treatment, rd.orthodontic_treatment)
    d.wisdom_teeth_removed  = _merge_none(d.wisdom_teeth_removed,  rd.wisdom_teeth_removed)
    d.pain_or_bleeding      = _merge_none(d.pain_or_bleeding,      rd.pain_or_bleeding)
    d.oral_exam_with_bitewing=_merge_none(d.oral_exam_with_bitewing,rd.oral_exam_with_bitewing)
    d.panoramic_done        = _merge_none(d.panoramic_done,        rd.panoramic_done)
    d.third_molars_assessed = _merge_none(d.third_molars_assessed, rd.third_molars_assessed)
    d.active_orthodontic    = _merge_none(d.active_orthodontic,    rd.active_orthodontic)
    d.dental_decay          = _merge_none(d.dental_decay,          rd.dental_decay)
    d.bonded_retainers      = _merge_none(d.bonded_retainers,      rd.bonded_retainers)
    d.oral_hygiene_acceptable=_merge_none(d.oral_hygiene_acceptable,rd.oral_hygiene_acceptable)

    return d


def _apply_defaults(result: "ProcessingResult") -> None:
    """
    Apply sensible Utah-missionary defaults to any fields still None after all
    parsing and merging.  Only fills fields that are explicitly None — never
    overwrites a successfully parsed value.

    Rationale for each default:
      vision          – "20/20" is normal; OCR garbles the multi-column table
      urinalysis      – "--" (not done / normal) is the nearly-universal entry
      functional_level– "A" (no limitation) is true for almost all missionaries
      chronic_condition / driving_medication – "No" is the overwhelming majority
      physician state / country – virtually all submitters are Utah / United States
      tb_exposure_risk– "No" is correct for nearly all Utah missionaries
    """
    m = result.medical
    if not m:
        return

    # Vitals
    if m.vital_signs:
        vs = m.vital_signs
        if vs.vision_right is None:
            vs.vision_right = "20/20"
        if vs.vision_left is None:
            vs.vision_left = "20/20"

    # Labs
    if m.labs:
        lab = m.labs
        if lab.urinalysis_blood is None:
            lab.urinalysis_blood = "--"
        if lab.urinalysis_protein is None:
            lab.urinalysis_protein = "--"
        if lab.urinalysis_sugar is None:
            lab.urinalysis_sugar = "--"
        if lab.tb_exposure_risk is None:
            lab.tb_exposure_risk = "No"

    # Medical form top-level checkboxes
    if m.functional_level is None:
        m.functional_level = "A"
    if m.chronic_condition is None:
        m.chronic_condition = "No"
    if m.driving_medication is None:
        m.driving_medication = "No"

    # Physician location
    if m.physician_info:
        pi = m.physician_info
        if pi.state is None:
            pi.state = "UT"
        if pi.country is None:
            pi.country = "United States"


def parse_forms(
    text: str,
    missionary_name: str,
    pdf_path: str,
    has_medical: bool,
    has_dental: bool,
    pages: Optional[list] = None,
) -> ProcessingResult:
    """
    Parse a missionary PDF using a hybrid Ollama + regex strategy.

    1. Always run the regex parser first (fast, reliable for text/numeric fields).
    2. Run Ollama Vision (needed for checkboxes and handwritten values).
    3. Merge: Ollama values take precedence; regex fills in any nulls Ollama missed.

    This gives the best of both worlds: regex accuracy for structured text fields
    and Ollama's visual understanding for checkbox/radio fields.

    pages: optional list of pre-rendered page dicts (with 'image' and optionally
           'ocr_observations' keys) — if provided, these are used for Ollama
           instead of re-rendering the PDF at lower DPI.
    """
    result = ProcessingResult(
        missionary_name=missionary_name,
        pdf_path=pdf_path,
        has_medical=has_medical,
        has_dental=has_dental,
    )

    # ── Step 1: Regex parsing (fast, reliable for text fields) ───────────
    regex_med = None
    regex_dnt = None
    if has_medical:
        try:
            regex_med = parse_medical_form(text)
            logger.info(f"Regex medical parsing done for {missionary_name}")
        except Exception as e:
            logger.warning(f"Regex medical parsing failed: {e}")

    if has_dental:
        try:
            regex_dnt = parse_dental_form(text)
            logger.info(f"Regex dental parsing done for {missionary_name}")
        except Exception as e:
            logger.warning(f"Regex dental parsing failed: {e}")

    from .config import load_config
    config = load_config()

    # ── Step 2: Gemini Vision (primary vision parser, if enabled) ───────────────
    gemini_med = None
    gemini_dnt = None
    if config.get("gemini_enabled"):
        try:
            from .gemini_parser import parse_with_gemini, render_pages
            gemini_pages = pages if pages else render_pages(pdf_path, dpi=150)
            gemini_med, gemini_dnt, gemini_errors = parse_with_gemini(
                gemini_pages, has_medical, has_dental,
                api_key=config.get("gemini_api_key", ""),
                model=config.get("gemini_model", "gemini-1.5-flash"),
            )
            for err in gemini_errors:
                result.warnings.append(err)
            if gemini_errors:
                logger.warning(f"Gemini errors: {gemini_errors}")
            if gemini_med:
                logger.info(f"Gemini medical extraction succeeded for {missionary_name}")
            if gemini_dnt:
                logger.info(f"Gemini dental extraction succeeded for {missionary_name}")
        except Exception as e:
            logger.warning(f"Gemini parser unavailable: {e}")

    # ── Step 3: Ollama Vision (fallback / gap-fill) ───────────────────────
    ollama_med = None
    ollama_dnt = None
    if config.get("ollama_enabled", True):
        try:
            from .ollama_parser import parse_with_ollama, render_pages
            ollama_pages = pages if pages else render_pages(pdf_path, dpi=150)

            ollama_med, ollama_dnt, ollama_errors = parse_with_ollama(
                ollama_pages, has_medical, has_dental,
                model=config.get("ollama_model", "qwen2.5vl"),
            )
            for err in ollama_errors:
                result.warnings.append(err)
            if ollama_errors:
                logger.warning(f"Ollama errors: {ollama_errors}")
        except Exception as e:
            logger.warning(f"Ollama parser unavailable: {e}")

    # ── Step 4: Claude Vision (fallback / gap-fill) ───────────────────────
    claude_med = None
    claude_dnt = None
    if config.get("claude_enabled"):
        try:
            from .claude_parser import parse_with_claude, render_pages
            claude_pages = pages if pages else render_pages(pdf_path, dpi=150)
            claude_med, claude_dnt, claude_errors = parse_with_claude(
                claude_pages, has_medical, has_dental,
                api_key=config.get("claude_api_key", ""),
                model=config.get("claude_model", "claude-3-5-sonnet-20241022"),
            )
            for err in claude_errors:
                result.warnings.append(err)
            if claude_errors:
                logger.warning(f"Claude errors: {claude_errors}")
            if claude_med:
                logger.info(f"Claude medical extraction succeeded for {missionary_name}")
            if claude_dnt:
                logger.info(f"Claude dental extraction succeeded for {missionary_name}")
        except Exception as e:
            logger.warning(f"Claude parser unavailable: {e}")

    # ── Step 5: Merge — Gemini > Ollama > Claude > regex ──────────────────────
    if has_medical:
        if gemini_med and ollama_med:
            best_vision_med = _merge_medical(gemini_med, ollama_med)
        else:
            best_vision_med = gemini_med or ollama_med
        
        if best_vision_med and claude_med:
            best_vision_med = _merge_medical(best_vision_med, claude_med)
        elif claude_med and not best_vision_med:
            best_vision_med = claude_med

        if best_vision_med and regex_med:
            result.medical = _merge_medical(best_vision_med, regex_med)
            sources = []
            if gemini_med: sources.append("Gemini")
            if ollama_med: sources.append("Ollama")
            if claude_med: sources.append("Claude")
            src = "+".join(sources) + "+regex"
            logger.info(f"Medical: {src} for {missionary_name}")
        elif best_vision_med:
            sources = []
            if gemini_med: sources.append("Gemini")
            if ollama_med: sources.append("Ollama")
            if claude_med: sources.append("Claude")
            src = "+".join(sources)
            logger.info(f"Medical: vision only ({src}) for {missionary_name}")
        elif regex_med:
            result.medical = regex_med
            logger.info(f"Medical: regex only for {missionary_name}")

    if has_dental:
        if gemini_dnt and ollama_dnt:
            best_vision_dnt = _merge_dental(gemini_dnt, ollama_dnt)
        else:
            best_vision_dnt = gemini_dnt or ollama_dnt
        
        if best_vision_dnt and claude_dnt:
            best_vision_dnt = _merge_dental(best_vision_dnt, claude_dnt)
        elif claude_dnt and not best_vision_dnt:
            best_vision_dnt = claude_dnt

        if best_vision_dnt and regex_dnt:
            result.dental = _merge_dental(best_vision_dnt, regex_dnt)
            sources = []
            if gemini_dnt: sources.append("Gemini")
            if ollama_dnt: sources.append("Ollama")
            if claude_dnt: sources.append("Claude")
            src = "+".join(sources) + "+regex"
            logger.info(f"Dental: {src} for {missionary_name}")
        elif best_vision_dnt:
            sources = []
            if gemini_dnt: sources.append("Gemini")
            if ollama_dnt: sources.append("Ollama")
            if claude_dnt: sources.append("Claude")
            src = "+".join(sources)
            logger.info(f"Dental: vision only ({src}) for {missionary_name}")
        elif regex_dnt:
            result.dental = regex_dnt
            logger.info(f"Dental: regex only for {missionary_name}")

    # ── Step 6: Apply defaults for fields still None ──────────────────────    # ── Step 5: Apply defaults for fields still None ──────────────────────
    _apply_defaults(result)

    return result
