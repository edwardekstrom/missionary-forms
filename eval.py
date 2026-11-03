#!/usr/bin/env python3
"""
Simple evaluation script to compare model outputs against ground truth.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent))

from app.config import load_config, PROCESSED_DIR
from app.form_parser import parse_forms
from app.pdf_processor import extract_pdf_text, combine_pages, detect_form_types
from app.models import MedicalForm, DentalForm


@dataclass
class FieldResult:
    field: str
    predicted: Any
    expected: Any
    match: bool
    category: str  # exact, partial, missing, extra


def norm(v):
    if v is None: return None
    if isinstance(v, str):
        v = v.strip()
        if v.lower() in ('yes','no','not applicable','not done','negative','positive','normal','abnormal'):
            return v.lower()
        return v
    return v


def compare(pred, exp):
    p, e = norm(pred), norm(exp)
    if p is None and e is None: return True, 'exact'
    if p is None and e is not None: return False, 'missing'
    if p is not None and e is None: return False, 'extra'
    if p == e: return True, 'exact'
    if isinstance(p,str) and isinstance(e,str):
        if p in e or e in p: return True, 'partial'
        try:
            from datetime import datetime
            for fmt in ['%d %b %Y','%m/%d/%Y','%d %B %Y','%Y-%m-%d']:
                try:
                    if datetime.strptime(p,fmt) == datetime.strptime(e,fmt):
                        return True, 'partial'
                except: pass
        except: pass
    return False, 'partial'


def evaluate(pred_form, gt_dict, form_type, name):
    pred = pred_form.model_dump() if pred_form else {}
    gt = gt_dict or {}
    all_fields = set(pred.keys()) | set(gt.keys())
    results = []
    for f in sorted(all_fields):
        m, c = compare(pred.get(f), gt.get(f))
        results.append(FieldResult(f, pred.get(f), gt.get(f), m, c))
    return results


def run():
    # Disable vision parsers for fast baseline evaluation
    import app.config as config_mod
    original_load = config_mod.load_config
    def fast_config():
        c = original_load()
        c['ollama_enabled'] = False
        c['gemini_enabled'] = False
        c['claude_enabled'] = False
        return c
    config_mod.load_config = fast_config
    
    load_config()
    
    gt = json.load(open('logs/ground_truth.json'))
    pdfs = list(PROCESSED_DIR.glob("*.pdf"))
    
    print(f"Found {len(pdfs)} PDFs")
    print(f"Ground truth entries: {len(gt)}")
    
    manual_mapping = {
        'Lloyd, Alexander Thomas': 'Alexander Lloyd.pdf',
        'Gillan, Sean Ryan': 'Sean Gillan Missionary Medical Forms - Joshua G C.pdf',
        'West, Jordan James': 'Jordan West Missionary Medical Forms - Joshua G C.pdf',
        'Thackeray, Lily Jane': 'Lily Thackeray.pdf',
        'Mangum, Kaleb Lawrence': 'Nathan Medical_Dental - Jake Merkley.pdf',
        'Moehle, Nathan Thomas': 'Nathan Medical_Dental - Jake Merkley.pdf',
        'Martin, Joseph Thatcher': 'Thatcher Medical Dental - Jake Merkley.pdf',
    }
    
    pdf_map = {}
    for gt_name, pdf_name in manual_mapping.items():
        matches = [p for p in pdfs if p.name == pdf_name]
        if matches:
            pdf_map[gt_name] = matches[0]
            print(f"  MATCHED: {gt_name} -> {pdf_name}")
        else:
            print(f"  NOT FOUND: {gt_name} -> {pdf_name}")
    
    print(f"\nMatched {len(pdf_map)}/{len(gt)} PDFs")
    
    all_results = {}
    for name, pdf_path in pdf_map.items():
        print(f"\nEvaluating: {name}")
        pages = extract_pdf_text(pdf_path)
        text = combine_pages(pages)
        has_med, has_den = detect_form_types(pdf_path.name, text)
        
        result = parse_forms(text=text, missionary_name=name, pdf_path=str(pdf_path),
                            has_medical=has_med, has_dental=has_den)
        
        med_res = evaluate(result.medical, gt[name].get('medical',{}), 'medical', name) if has_med and result.medical else []
        den_res = evaluate(result.dental, gt[name].get('dental',{}), 'dental', name) if has_den and result.dental else []
        
        all_results[name] = {'medical': med_res, 'dental': den_res}
        
        for form_type, res in [('medical', med_res), ('dental', den_res)]:
            if res:
                exact = sum(1 for r in res if r.match and r.category=='exact')
                partial = sum(1 for r in res if r.category=='partial')
                total = len(res)
                acc = (exact + partial*0.5)/total if total else 0
                print(f"  {form_type}: {exact}/{total} exact, acc={acc:.1%}")
    
    # Summary
    print("\n" + "="*50)
    print("SUMMARY (Regex-only baseline)")
    for form_type in ['medical', 'dental']:
        all_res = [r for name, r in all_results.items() for r in r.get(form_type, [])]
        if all_res:
            exact = sum(1 for r in all_res if r.match and r.category=='exact')
            partial = sum(1 for r in all_res if r.category=='partial')
            total = len(all_res)
            acc = (exact + partial*0.5)/total
            print(f"\n{form_type.upper()}:")
            print(f"  Total fields: {total}")
            print(f"  Exact: {exact} ({exact/total:.1%})")
            print(f"  Partial: {partial} ({partial/total:.1%})")
            print(f"  Accuracy: {acc:.1%}")

    # Show top errors
    print("\nTOP ERRORS:")
    for name, res in all_results.items():
        for form_type in ['medical', 'dental']:
            errors = [r for r in res.get(form_type, []) if not r.match]
            if errors:
                print(f"\n  {name} - {form_type}:")
                for e in errors[:5]:
                    print(f"    {e.field}: pred='{e.predicted}' exp='{e.expected}' [{e.category}]")


if __name__ == '__main__':
    load_config()
    run()