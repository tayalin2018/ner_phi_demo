import argparse, os, sys, json
from collections import Counter
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from rules.validators import find_all
from rules.conflict_resolution import resolve
from service.redaction_policy import apply_redactions

def anonymize_text(nlp, text):
    preds = nlp(text)
    spans_ner = [(p["start"], p["end"], p["entity_group"].split("-",1)[-1]) for p in preds]
    spans_rule = find_all(text)
    combined = resolve(spans_ner + spans_rule)
    spans = [{"start": s, "end": e, "label": l} for (s,e,l) in combined]
    redacted = apply_redactions(text, spans)
    return redacted, spans

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV/TSV/JSONL/TXT file")
    ap.add_argument("--output", required=True, help="Output CSV path")
    ap.add_argument("--text-col", default="text", help="Text column name (CSV/TSV/JSONL)")
    ap.add_argument("--sep", default=None, help="Separator override for CSV/TSV")
    ap.add_argument("--model_dir", default=os.environ.get("MODEL_DIR","outputs/phi_fast"))
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    nlp = pipeline("token-classification", model=model, tokenizer=tok, aggregation_strategy="simple")

    ext = os.path.splitext(args.input)[1].lower()
    rows = []

    def add_row(base, text):
        red, spans = anonymize_text(nlp, text)
        base["redacted_text"] = red
        base["spans_json"] = json.dumps(spans, ensure_ascii=False)
        cnt = Counter(s["label"] for s in spans)
        for lab, c in cnt.items():
            base[f"count_{lab}"] = c
        return base

    if ext in (".csv", ".tsv"):
        sep = args.sep or ("," if ext == ".csv" else "\t")
        df = pd.read_csv(args.input, sep=sep)
        if args.text_col not in df.columns:
            print(f"Text column '{args.text_col}' not found. Available: {list(df.columns)}", file=sys.stderr)
            sys.exit(2)
        for _, r in df.iterrows():
            text = str(r[args.text_col])
            rows.append(add_row(dict(r), text))

    elif ext in (".jsonl", ".ndjson"):
        with open(args.input, encoding="utf-8") as f:
            for line in f:
                ex = json.loads(line)
                text = str(ex.get(args.text_col) or ex.get("text") or "")
                rows.append(add_row(ex, text))

    elif ext == ".txt":
        with open(args.input, encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.rstrip("\n")
                if not line: continue
                rows.append(add_row({"id": i, args.text_col: line}, line))
    else:
        print("Unsupported input type. Use CSV/TSV/JSONL/TXT.", file=sys.stderr)
        sys.exit(1)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.output, index=False)

if __name__ == "__main__":
    main()
