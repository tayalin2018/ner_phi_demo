# NER PHI/PII Anonymization Demo

A minimal, reproducible starter to fine-tune a token-classification model for **health/legal PHI** and serve it as a **redaction API**.

**Entities**: `MRN`, `INSURANCE_ID`, `ICD_CODE`, `CPT_CODE`, `DOCKET_NO`, `CASE_NO`, `PRESCRIPTION_ID`


---
## Results (synthetic test set)

Overall:
- **Micro avg** — precision: **0.820707**, recall: **0.890411**, F1: **0.854139**
- **Macro avg** — precision: **0.819262**, recall: **0.873440**, F1: **0.834260**
- **Weighted avg** — precision: **0.874033**, recall: **0.890411**, F1: **0.876667**
- Support: **365** tokens (BIO, strict span)

Per-label:

| label            | precision | recall   | f1        | support |
|------------------|-----------|----------|-----------|---------|
| CASE_NO          | 0.651163  | 0.700000 | 0.674699  | 40      |
| CPT_CODE         | 1.000000  | 1.000000 | 1.000000  | 64      |
| DOCKET_NO        | 1.000000  | 1.000000 | 1.000000  | 56      |
| ICD_CODE         | 1.000000  | 1.000000 | 1.000000  | 64      |
| INSURANCE_ID     | 1.000000  | 1.000000 | 1.000000  | 34      |
| MRN              | 0.820513  | 0.761905 | 0.790123  | 84      |
| PRESCRIPTION_ID  | 0.263158  | 0.652174 | 0.375000  | 23      |

> Notes: synthetic data only; strict span matching; see `out/per_label_test.csv` for the saved table.


---

## Quickstart

> Requires Python 3.10+

```bash
# 1) Install
pip install -r requirements.txt

# 2) Generate synthetic data
python -m data.generate_synth_data --train 800 --dev 160 --test 160

# 3) Train a fast baseline
python src/train_simple.py \
  --model distilroberta-base \
  --outdir outputs/phi_fast \
  --train data/train.jsonl \
  --dev data/dev.jsonl \
  --epochs 3 \
  --batch 8

# 4) Evaluate on test
python src/evaluate.py --model_dir outputs/phi_fast --data data/test.jsonl
# (example result on synthetic test: P≈0.82, R≈0.89, F1≈0.85)

# 5) Serve the anonymization API
export MODEL_DIR=outputs/phi_fast
uvicorn service.app:app --host 0.0.0.0 --port 8000
