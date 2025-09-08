import argparse, json, os, numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, Trainer
from seqeval.metrics import classification_report

def read_jsonl(p):
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def char_spans_to_token_labels(text, entities, tokenizer, label2id, max_len=512):
    enc = tokenizer(text, return_offsets_mapping=True, truncation=True, max_length=max_len)
    offs = enc["offset_mapping"]; labels = [-100]*len(offs)
    ents = sorted([(e["start"], e["end"], e["label"]) for e in entities], key=lambda x:x[0])
    for i,(s,e) in enumerate(offs):
        if s==e: continue
        tag = "O"
        for (es,ee,lab) in ents:
            if s>=es and e<=ee:
                ps,pe = offs[i-1] if i>0 else (None,None)
                start_tok = not(i>0 and ps is not None and ps>=es and pe<=ee)
                tag = ("B-" if start_tok else "I-") + lab
                break
        labels[i] = label2id.get(tag, label2id.get("O", 0))
    enc.pop("offset_mapping")
    return enc, labels

def tok_ds(examples, tokenizer, label2id):
    X,A,Y = [],[],[]
    for ex in examples:
        enc, lab = char_spans_to_token_labels(ex["text"], ex["entity_spans"], tokenizer, label2id)
        X.append(enc["input_ids"]); A.append(enc["attention_mask"]); Y.append(lab)
    return Dataset.from_dict({"input_ids":X,"attention_mask":A,"labels":Y})

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--data", default="data/test.jsonl")
    ap.add_argument("--output_csv", default="out/per_label_test.csv")
    ap.add_argument("--output_md",  default="out/per_label_test.md")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)

    # label maps (prefer model config; fallback to labels.txt if needed)
    id2label = {int(k):v for k,v in getattr(model.config, "id2label", {}).items()}
    if not id2label:
        labs = [l.strip() for l in open(os.path.join(args.model_dir,"labels.txt"),encoding="utf-8") if l.strip()]
        id2label = {i:l for i,l in enumerate(labs)}
    label2id = {v:k for k,v in id2label.items()}

    examples = list(read_jsonl(args.data))
    ds = tok_ds(examples, tok, label2id)
    collator = DataCollatorForTokenClassification(tok)
    trainer = Trainer(model=model, tokenizer=tok, data_collator=collator)
    preds = trainer.predict(ds)

    pred_ids = np.argmax(preds.predictions, axis=-1)
    y_pred, y_true = [], []
    for pred, lab in zip(pred_ids, preds.label_ids):
        pp,ll = [],[]
        for p,l in zip(pred,lab):
            if l != -100:
                pp.append(id2label[p]); ll.append(id2label[l])
        y_pred.append(pp); y_true.append(ll)

    rep = classification_report(y_true, y_pred, output_dict=True, digits=4)

    # Build a small table
    import pandas as pd
    rows = []
    for k,v in rep.items():
        if isinstance(v, dict):
            rows.append({
                "label": k,
                "precision": v.get("precision"),
                "recall": v.get("recall"),
                "f1": v.get("f1-score"),
                "support": v.get("support"),
            })
    df = pd.DataFrame(rows)

    # Move summary rows to the bottom if present
    order_last = ["micro avg","macro avg","weighted avg","accuracy"]
    df["__ord__"] = df["label"].apply(lambda x: 1 if x in order_last else 0)
    df = df.sort_values(["__ord__","label"]).drop(columns="__ord__")

    # Save
    df.to_csv(args.output_csv, index=False)
    try:
        md = df.to_markdown(index=False)
        with open(args.output_md, "w", encoding="utf-8") as f:
            f.write(md + "\n")
    except Exception:
        pass

    # Print headline
    headline = {k:v for k,v in rep.get("micro avg", rep.get("weighted avg", {})).items() if k in ("precision","recall","f1-score")}
    print("Saved per-label report to:", args.output_csv, "and", args.output_md)
    if headline:
        print("Headline:", {k: round(float(v),4) for k,v in headline.items()})
