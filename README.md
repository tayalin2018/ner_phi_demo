# NER PHI/PII Anonymization Demo

Minimal starter to fine-tune a token-classification model for health/legal PHI entities and serve it as a redaction API.

## Labels
MRN, INSURANCE_ID, ICD_CODE, CPT_CODE, DOCKET_NO, CASE_NO, PRESCRIPTION_ID

## Quickstart
1) Install deps
```bash
pip install -r requirements.txt
```
2) Generate synthetic data
```bash
python data/generate_synth_data.py --train 300 --dev 60 --test 60
```
3) Train baseline (RoBERTa-base)
```bash
python src/train.py --model roberta-base --outdir outputs/roberta_phi --train data/train.jsonl --dev data/dev.jsonl
```
4) Evaluate
```bash
python src/evaluate.py --model_dir outputs/roberta_phi --data data/test.jsonl
```
5) Serve API
```bash
export MODEL_DIR=outputs/roberta_phi
uvicorn service.app:app --host 0.0.0.0 --port 8000
```
POST:
```bash
curl -X POST http://localhost:8000/anonymize -H "Content-Type: application/json" -d '{"text":"Patient MRN 004-77-9123 with dx E11.9 had CPT 99213 per docket 1:23-cv-0042."}'
```
Limitations & next steps

Synthetic data only: no real PHI; formats are representative but simplified.

Pattern drift: MRN/Insurance/Case/Docket formats vary by org/jurisdiction—rules may miss edge cases.

Boundary sensitivity: tokenization can affect span starts/ends; we use BIO with strict span matching.

Obfuscations: model trained on clean text; robustness to zero-width/homoglyphs not guaranteed (see planned test-only perturbations).

Multilingual: English-only; no transliteration or non-whitespace languages yet.

Long docs: inputs truncated at 512 tokens; chunking/overlap not implemented.

Next steps

Add programmatic weak labels (prompted ICD/CPT/MRN variants), plus hard negatives for false positives.

Per-label thresholds & abstention for high-recall types (e.g., MRN/Insurance).

Robustness: train with safe perturbations; keep spans aligned.

Ship an MLflow run + simple Docker image; consider a CI job to run tests.