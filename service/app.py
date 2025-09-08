
import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from rules.validators import find_all
from rules.conflict_resolution import resolve
from service.redaction_policy import apply_redactions

MODEL_DIR = os.environ.get("MODEL_DIR", "outputs/roberta_phi")
tok = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
nlp = pipeline("token-classification", model=model, tokenizer=tok, aggregation_strategy="simple")

class AnonIn(BaseModel):
    text: str

app = FastAPI(title="PHI/PII Anonymizer")

@app.get("/health")
def health():
    return {"status":"ok"}

def ner_spans(text: str) -> List[Dict]:
    preds = nlp(text)
    spans = []
    for p in preds:
        lab = p["entity_group"]
        if lab.startswith("B-") or lab.startswith("I-"):
            lab = lab.split("-",1)[1]
        spans.append({"start": p["start"], "end": p["end"], "label": lab, "score": float(p["score"])})
    return spans

def merge_with_rules(text: str, ner_list: List[Dict]) -> List[Dict]:
    rule_spans = [{"start": s, "end": e, "label": l, "score": 1.0} for (s,e,l) in find_all(text)]
    combined = [(sp["start"], sp["end"], sp["label"]) for sp in (ner_list + rule_spans)]
    final = resolve(combined)
    return [{"start": s, "end": e, "label": l} for (s,e,l) in final]

@app.post("/anonymize")
def anonymize(inp: AnonIn):
    spans = ner_spans(inp.text)
    spans = merge_with_rules(inp.text, spans)
    redacted = apply_redactions(inp.text, spans)
    return {"text": inp.text, "spans": spans, "redacted_text": redacted}
