"""
Pydantic data models for missionary medical and dental forms.
"""

from pydantic import BaseModel
from typing import Optional, List


class PersonalInfo(BaseModel):
    first_name: Optional[str] = None
    middle_name: Optional[str] = None
    last_name: Optional[str] = None
    suffix: Optional[str] = None
    record_number: Optional[str] = None
    date_of_birth: Optional[str] = None
    age: Optional[str] = None
    gender: Optional[str] = None


class PhysicianInfo(BaseModel):
    name: Optional[str] = None
    credentials: Optional[str] = None  # MD, DO, PA, NP
    address: Optional[str] = None
    city: Optional[str] = None
    country: Optional[str] = "United States"   # HlthEvalPhysicianCountry (select)
    state: Optional[str] = "Utah"
    district: Optional[str] = None     # HlthEvalPhysicianDistrict
    postal_code: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None


class VitalSigns(BaseModel):
    height: Optional[str] = None        # e.g. "61 in" or "155 cm"
    weight: Optional[str] = None        # e.g. "125 lbs" or "57 kg"
    blood_pressure: Optional[str] = None  # e.g. "120/80"
    pulse: Optional[str] = None         # e.g. "65"
    vision_right: Optional[str] = None  # e.g. "20/20"
    vision_left: Optional[str] = None   # e.g. "20/18"
    vision_corrected: Optional[bool] = None


class SystemExam(BaseModel):
    """Each field is 'Normal', 'Abnormal', or 'Not Indicated'."""
    general: Optional[str] = None
    skin: Optional[str] = None
    eyes: Optional[str] = None
    ears: Optional[str] = None
    nose_throat: Optional[str] = None
    chest: Optional[str] = None
    heart: Optional[str] = None
    abdomen: Optional[str] = None
    genitalia: Optional[str] = None
    back: Optional[str] = None
    extremities: Optional[str] = None
    neurological: Optional[str] = None
    breast_pelvic: Optional[str] = None


class LabResults(BaseModel):
    urinalysis_blood: Optional[str] = None    # "Negative", "Positive", "Not Done"
    urinalysis_protein: Optional[str] = None
    urinalysis_sugar: Optional[str] = None
    hemoglobin: Optional[str] = None          # numeric value e.g. "16.1"
    tb_exposure_risk: Optional[str] = None    # "Yes" or "No"
    ppd_result: Optional[str] = None          # e.g. "<5 mm", "Neg", "Not Done"
    igra_result: Optional[str] = None         # "Negative", "Positive", "Not Done"
    chest_xray: Optional[str] = None          # "Normal", "Abnormal", "Not Done"


class Immunizations(BaseModel):
    tdap: Optional[str] = None
    tdap2: Optional[str] = None
    mmr1: Optional[str] = None
    mmr2: Optional[str] = None
    polio: Optional[str] = None
    hep_a1: Optional[str] = None
    hep_a2: Optional[str] = None
    hep_b1: Optional[str] = None
    hep_b2: Optional[str] = None
    hep_b3: Optional[str] = None
    influenza: Optional[str] = None
    covid_doses: Optional[List[str]] = None
    meningococcal: Optional[str] = None
    varicella: Optional[str] = None
    other: Optional[str] = None


class MedicalForm(BaseModel):
    personal_info: PersonalInfo = PersonalInfo()
    physician_info: PhysicianInfo = PhysicianInfo()
    functional_level: Optional[str] = None   # "A"–"E" → website 1–5
    chronic_condition: Optional[str] = None  # "Yes" or "No" → 1/0
    chronic_comments: Optional[str] = None   # HlthEvalFitnessComment
    driving_medication: Optional[str] = None         # "Yes"/"No" → HlthEvalDrivingMedicationYN
    driving_medication_explain: Optional[str] = None # HlthEvalDrivingMedicationExplain
    vital_signs: VitalSigns = VitalSigns()
    system_exam: SystemExam = SystemExam()
    labs: LabResults = LabResults()
    immunizations: Immunizations = Immunizations()
    exam_date: Optional[str] = None


class DentalForm(BaseModel):
    personal_info: PersonalInfo = PersonalInfo()

    # Candidate self-reported history
    orthodontic_treatment: Optional[str] = None   # "Yes" or "No"
    orthodontic_complete: Optional[str] = None    # "Yes" or "No"
    wisdom_teeth_removed: Optional[str] = None    # "Yes" or "No"
    brushing_frequency: Optional[str] = None      # e.g. "Morning and Night", "Daily"
    flossing_frequency: Optional[str] = None      # e.g. "Daily", "Weekly"
    pain_or_bleeding: Optional[str] = None        # "Yes" or "No"

    # Extra comments from candidate history section
    bleeding_comments: Optional[str] = None  # DntlHistBleedingComments

    # Dentist information
    dentist_name: Optional[str] = None
    dentist_address: Optional[str] = None
    dentist_city: Optional[str] = None
    dentist_country: Optional[str] = "United States"   # DntlEvalDentistCountry (select)
    dentist_state: Optional[str] = "Utah"
    dentist_district: Optional[str] = None   # DntlEvalDentistDistrict
    dentist_postal: Optional[str] = None
    dentist_phone: Optional[str] = None
    dentist_email: Optional[str] = None
    exam_date: Optional[str] = None

    # Dentist evaluation findings
    oral_exam_with_bitewing: Optional[str] = None   # "Yes" or "No"
    third_molars_assessed: Optional[str] = None     # "Yes", "No", or "Not Applicable"
    panoramic_done: Optional[str] = None            # "Yes", "No", or "Not Applicable"
    active_orthodontic: Optional[str] = None        # "Yes", "No", or "Not Applicable"
    dental_decay: Optional[str] = None              # "Yes" or "No"
    bonded_retainers: Optional[str] = None          # "Yes" or "No"
    oral_hygiene_acceptable: Optional[str] = None   # "Yes" or "No"
    dental_comments: Optional[str] = None


class ProcessingResult(BaseModel):
    missionary_name: str
    pdf_path: str
    has_medical: bool = False
    has_dental: bool = False
    medical: Optional[MedicalForm] = None
    dental: Optional[DentalForm] = None
    errors: List[str] = []
    warnings: List[str] = []
