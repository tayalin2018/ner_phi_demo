
import argparse, json, os
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, Trainer
from seqeval.metrics import f1_score, precision_score, recall_score

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def load_label_maps(model_dir):
    with open(os.path.join(model_dir, "labels.txt"), "r", encoding="utf-8") as f:
        label_list = [l.strip() for l in f if l.strip()]
    id2label = {i:l for i,l in enumerate(label_list)}
    label2id = {l:i for i,l in id2label.items()}
    return label_list, id2label, label2id

def char_spans_to_token_labels(text, entities, tokenizer, label2id, max_length=512):
    enc = tokenizer(text, return_offsets_mapping=True, truncation=True, max_length=max_length)
    offsets = enc["offset_mapping"]
    labels = [-100] * len(offsets)
    ents = sorted([(e["start"], e["end"], e["label"]) for e in entities], key=lambda x: x[0])
    for i,(s,e) in enumerate(offsets):
        if s==e: continue
        tag = "O"
        for (es,ee,lab) in ents:
            if s>=es and e<=ee:
                prev_inside = False
                if i>0:
                    ps,pe = offsets[i-1]
                    prev_inside = (ps>=es and pe<=ee)
                tag = ("B-" if not prev_inside else "I-") + lab
                break
        labels[i] = label2id.get(tag, label2id["O"])
    enc.pop("offset_mapping")
    return enc, labels

def tokenize_dataset(examples, tokenizer, label2id):
    all_input_ids, all_attention_mask, all_labels = [], [], []
    for ex in examples:
        enc, lab = char_spans_to_token_labels(ex["text"], ex["entity_spans"], tokenizer, label2id)
        all_input_ids.append(enc["input_ids"])
        all_attention_mask.append(enc["attention_mask"])
        all_labels.append(lab)
    return Dataset.from_dict({"input_ids": all_input_ids, "attention_mask": all_attention_mask, "labels": all_labels})

def compute_metrics(id2label, preds, labels):
    pred_ids = np.argmax(preds, axis=-1)
    y_pred, y_true = [], []
    for pred, lab in zip(pred_ids, labels):
        pred_l, lab_l = [], []
        for p, l in zip(pred, lab):
            if l != -100:
                pred_l.append(id2label[p])
                lab_l.append(id2label[l])
        y_pred.append(pred_l); y_true.append(lab_l)
    return {"precision": precision_score(y_true, y_pred), "recall": recall_score(y_true, y_pred), "f1": f1_score(y_true, y_pred)}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, required=True)
    ap.add_argument("--data", type=str, default="data/test.jsonl")
    args = ap.parse_args()

    label_list, id2label, label2id = load_label_maps(args.model_dir)
    tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)

    examples = list(read_jsonl(args.data))
    ds = tokenize_dataset(examples, tok, label2id)
    collator = DataCollatorForTokenClassification(tok)
    trainer = Trainer(model=model, tokenizer=tok, data_collator=collator)
    preds = trainer.predict(ds)
    metrics = compute_metrics(id2label, preds.predictions, preds.label_ids)
    print("Test metrics:", metrics)
