# NER PHI/PII Anonymization Demo

A minimal, reproducible starter to fine-tune a token-classification model for **health/legal PHI** and serve it as a **redaction API**.

**Entities**: `MRN`, `INSURANCE_ID`, `ICD_CODE`, `CPT_CODE`, `DOCKET_NO`, `CASE_NO`, `PRESCRIPTION_ID`

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
