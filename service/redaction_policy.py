import hashlib

def _hash_tag(value: str, salt: str = "demo_salt") -> str:
    h = hashlib.sha1((salt + value).encode("utf-8")).hexdigest()[:4].upper()
    return f"ID_{h}"

def redact_span(label: str, value: str):
    if label in ("MRN","INSURANCE_ID","CASE_NO","DOCKET_NO"):
        return _hash_tag(value)
    elif label in ("PRESCRIPTION_ID",):
        return "[REDACTED]"
    elif label in ("ICD_CODE","CPT_CODE"):
        return f"{label[:3]}:[MASK]"
    else:
        return "[REDACTED]"

def apply_redactions(text: str, spans):
    spans = sorted(spans, key=lambda x: x["start"], reverse=True)
    for sp in spans:
        s,e,l = sp["start"], sp["end"], sp["label"]
        value = text[s:e]
        rep = redact_span(l, value)
        text = text[:s] + rep + text[e:]
    return text
