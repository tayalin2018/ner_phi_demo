
import argparse, json, os
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, Trainer, TrainingArguments
from seqeval.metrics import f1_score, precision_score, recall_score

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def build_label_list(labels_path: str):
    with open(labels_path, "r", encoding="utf-8") as f:
        labs = [l.strip() for l in f if l.strip()]
    id2label = {i:lab for i,lab in enumerate(labs)}
    label2id = {lab:i for i,lab in id2label.items()}
    return labs, id2label, label2id

def char_spans_to_token_labels(text, entities, tokenizer, label2id, max_length=512):
    enc = tokenizer(text, return_offsets_mapping=True, truncation=True, max_length=max_length)
    offsets = enc["offset_mapping"]
    labels = [-100] * len(offsets)
    ents = sorted([(e["start"], e["end"], e["label"]) for e in entities], key=lambda x: x[0])
    for i, (s,e) in enumerate(offsets):
        if s == e:
            continue
        tag = "O"
        for (es,ee,lab) in ents:
            if s >= es and e <= ee:
                prev_inside = False
                if i>0:
                    ps,pe = offsets[i-1]
                    prev_inside = (ps >= es and pe <= ee)
                tag = ("B-" if not prev_inside else "I-") + lab
                break
        labels[i] = label2id.get(tag, label2id["O"])
    enc.pop("offset_mapping")
    return enc, labels

def tokenize_dataset(examples, tokenizer, label2id, max_length=512):
    all_input_ids, all_attention_mask, all_labels = [], [], []
    for ex in examples:
        enc, lab = char_spans_to_token_labels(ex["text"], ex["entity_spans"], tokenizer, label2id, max_length=max_length)
        all_input_ids.append(enc["input_ids"])
        all_attention_mask.append(enc["attention_mask"])
        all_labels.append(lab)
    return Dataset.from_dict({"input_ids": all_input_ids, "attention_mask": all_attention_mask, "labels": all_labels})

def compute_metrics_builder(id2label):
    import numpy as np
    def align_preds(preds, labels):
        pred_ids = np.argmax(preds, axis=-1)
        y_pred, y_true = [], []
        for pred, lab in zip(pred_ids, labels):
            pred_l, lab_l = [], []
            for p, l in zip(pred, lab):
                if l != -100:
                    pred_l.append(id2label[p])
                    lab_l.append(id2label[l])
            y_pred.append(pred_l); y_true.append(lab_l)
        return y_pred, y_true
    def compute_metrics(p):
        preds, labels = p
        y_pred, y_true = align_preds(preds, labels)
        return {"precision": precision_score(y_true, y_pred), "recall": recall_score(y_true, y_pred), "f1": f1_score(y_true, y_pred)}
    return compute_metrics

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="roberta-base")
    ap.add_argument("--train", type=str, default="data/train.jsonl")
    ap.add_argument("--dev", type=str, default="data/dev.jsonl")
    ap.add_argument("--labels", type=str, default="labels.txt")
    ap.add_argument("--outdir", type=str, default="outputs/roberta_phi")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    label_list, id2label, label2id = build_label_list(args.labels)

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(args.model, num_labels=len(label_list), id2label=id2label, label2id=label2id)

    train_examples = list(read_jsonl(args.train))
    dev_examples = list(read_jsonl(args.dev))
    train_ds = tokenize_dataset(train_examples, tok, label2id)
    dev_ds = tokenize_dataset(dev_examples, tok, label2id)

    collator = DataCollatorForTokenClassification(tok)
    metrics_fn = compute_metrics_builder(id2label)

    training_args = TrainingArguments(
        output_dir=args.outdir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        num_train_epochs=args.epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=train_ds, eval_dataset=dev_ds, tokenizer=tok, data_collator=collator, compute_metrics=metrics_fn)
    trainer.train()
    eval_metrics = trainer.evaluate()
    print("Dev metrics:", eval_metrics)

    trainer.save_model(args.outdir)
    tok.save_pretrained(args.outdir)
    with open(os.path.join(args.outdir, "labels.txt"), "w", encoding="utf-8") as f:
        f.write("\\n".join(label_list)+"\\n")
