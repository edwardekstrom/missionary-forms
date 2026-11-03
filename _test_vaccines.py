#!/usr/bin/env python3
"""Quick vaccine-field accuracy check using the real OCR pipeline.
Expected values are derived from reading the actual USIIS records, NOT from the
ground_truth.json (which has null for all vaccine fields).
"""
import os
from pathlib import Path
from app.pdf_processor import extract_pdf_text, combine_pages
from app.form_parser import parse_usiis_immunizations

BASE = Path("/Users/eddy.ekstrom/Downloads/Prospective Missionary_s Documents (File responses)/All")

# Expected values = what the USIIS should produce after correct parsing.
# Logic: tdap=oldest DTP, mmr1=oldest MMR, mmr2=2nd-oldest MMR,
#        polio=oldest POLIO, influenza=newest FLU.
CANDIDATES = [
    ("Ella Poelman",
     BASE / "2026.02.02 missionary health forms for Ellie Poelman - Scott Poelman.pdf",
     {'tdap':'04/21/2008','mmr1':'02/26/2009','mmr2':'03/18/2013','polio':'04/21/2008','influenza':'11/18/2025'}),
    ("Naish Hinckley",
     BASE / "Naish Hinckley Medical Forms - Greg Fowles.pdf",
     # Oldest DTP=11/01/2007, oldest MMR=01/21/2010, 2nd MMR=08/15/2012,
     # oldest POLIO=11/01/2007, newest FLU=11/27/2020
     {'tdap':'11/01/2007','mmr1':'01/21/2010','mmr2':'08/15/2012','polio':'11/01/2007','influenza':'11/27/2020'}),
    ("Henry Owen",
     BASE / "Henry Owen Medical Dental - Jake Merkley.pdf",
     # Oldest DTP=06/04/2008, oldest MMR=07/06/2009 (after OCR fix: 0B→08 for 2nd slot),
     # 2nd MMR=08/16/2013, oldest POLIO=06/04/2008, newest FLU=12/18/2023
     {'tdap':'06/04/2008','mmr1':'07/06/2009','mmr2':'08/16/2013','polio':'06/04/2008','influenza':'12/18/2023'}),
    ("Kaleb Mangum",
     BASE / "Kaleb Mangum Medical Dental - Jake Merkley.pdf",
     None),  # Known broken format — no expected values
    ("Ollie Hopkin",
     BASE / "Ollie Hopkin Medical Dental - Jake Merkley.pdf",
     None),
    ("Anna Miner",
     BASE / "Anna Miner Medical - Dave Miner.pdf",
     None),
    ("Jacob Durham",
     BASE / "All Medical Docs - Jacob Durham.pdf",
     None),
]

FIELDS = ['tdap','mmr1','mmr2','polio','influenza']

total_ok = 0
total = 0

for name, pdf_path, gt in CANDIDATES:
    if not pdf_path.exists():
        print(f"\n=== {name}: PDF NOT FOUND ===")
        continue
    print(f"\n=== {name} ===", flush=True)
    pages = extract_pdf_text(pdf_path)
    full_text = combine_pages(pages)
    vaccines = parse_usiis_immunizations(full_text)
    if vaccines is None:
        print("  NO USIIS SECTION FOUND")
        continue
    for field in FIELDS:
        got = getattr(vaccines, field, None)
        exp = gt.get(field) if gt else None
        if gt is None or exp is None:
            mark = 'SKIP(no GT)'
        elif got == exp:
            mark = 'OK'
            total_ok += 1
            total += 1
        else:
            mark = 'FAIL'
            total += 1
        print(f"  {field:12s}: got={str(got):14s}  exp={str(exp) if gt else '?':14s}  [{mark}]")

print(f"\nTotal: {total_ok}/{total} correct")
