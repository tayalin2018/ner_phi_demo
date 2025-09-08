
import re
from typing import List, Tuple

ICD_RE = re.compile(r'\b[A-TV-Z][0-9][0-9A-Z](?:\.[0-9A-Z]{1,4})?\b')
CPT_RE = re.compile(r'\b[0-9A-Z]{5}(?:-[0-9A-Z]{2})?\b')
DOCKET_RE = re.compile(r'(?ix)\b(?:No\.\s*)?(?:\d+:)?\d{2,4}-(?:cv|cr|mc|md|mj|po|bk|civ)-\d{1,6}\b')
CASE_RE = re.compile(r'(?i)\b(Case|Claim|File)\s*(No\.|#)?\s*[A-Z0-9-]{4,20}\b')
RX_RE = re.compile(r'(?i)\b(Rx|Prescription|Script)\s*(ID|#|No\.)?\s*[A-Z0-9-]{5,20}\b')

MRN_CTX_RE = re.compile(r'(?i)\b(MRN|Med(?:ical)?\s*Rec(?:ord)?\s*No\.?|Record\s*#)\b')
INS_CTX_RE = re.compile(r'(?i)\b(Member|Policy|Subscriber|Insurance)\s*(ID|No\.|#)?\b')
ID_CORE_RE = re.compile(r'\b[A-Z0-9-]{6,20}\b')

def find_icd(text: str):
    return [(m.start(), m.end(), "ICD_CODE") for m in ICD_RE.finditer(text)]

def find_cpt(text: str):
    return [(m.start(), m.end(), "CPT_CODE") for m in CPT_RE.finditer(text)]

def find_docket(text: str):
    return [(m.start(), m.end(), "DOCKET_NO") for m in DOCKET_RE.finditer(text)]

def find_case(text: str):
    return [(m.start(), m.end(), "CASE_NO") for m in CASE_RE.finditer(text)]

def find_prescription(text: str):
    return [(m.start(), m.end(), "PRESCRIPTION_ID") for m in RX_RE.finditer(text)]

def find_mrn(text: str):
    spans = []
    for ctx in MRN_CTX_RE.finditer(text):
        tail = text[ctx.end(): ctx.end()+24]
        m = ID_CORE_RE.search(tail)
        if m:
            s = ctx.end() + m.start()
            e = ctx.end() + m.end()
            spans.append((s,e,"MRN"))
    return spans

def find_insurance_id(text: str):
    spans = []
    for ctx in INS_CTX_RE.finditer(text):
        tail = text[ctx.end(): ctx.end()+20]
        m = ID_CORE_RE.search(tail)
        if m:
            s = ctx.end() + m.start()
            e = ctx.end() + m.end()
            spans.append((s,e,"INSURANCE_ID"))
    return spans

def find_all(text: str):
    finders = [find_icd, find_cpt, find_docket, find_case, find_prescription, find_mrn, find_insurance_id]
    spans = []
    for fn in finders:
        spans.extend(fn(text))
    if not spans: 
        return []
    spans.sort(key=lambda x: (x[0], -(x[1]-x[0])))
    merged = []
    for s,e,l in spans:
        if not merged: 
            merged.append([s,e,l])
            continue
        ms,me,ml = merged[-1]
        if s < me:  # overlap
            if (e-s) > (me-ms):
                merged[-1] = [s,e,l]
        else:
            merged.append([s,e,l])
    return [(s,e,l) for s,e,l in merged]
