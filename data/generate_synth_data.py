
import json, random, string, argparse, os
from evaluation.adversarial_perturb import perturb

LABELS = ["MRN","INSURANCE_ID","ICD_CODE","CPT_CODE","DOCKET_NO","CASE_NO","PRESCRIPTION_ID"]

def rand_icd():
    first = random.choice([c for c in string.ascii_uppercase if c not in "IOU"])
    second = random.choice(string.digits)
    third = random.choice(string.digits + string.ascii_uppercase)
    tail_len = random.choice([0,1,2,3,4])
    if tail_len>0:
        tail = "." + "".join(random.choice(string.digits+string.ascii_uppercase) for _ in range(tail_len))
    else:
        tail = ""
    return f"{first}{second}{third}{tail}"

def rand_cpt():
    chars = string.digits + string.ascii_uppercase
    base = "".join(random.choice(chars) for _ in range(5))
    if random.random()<0.3:
        base += "-" + "".join(random.choice(chars) for _ in range(2))
    return base

def rand_id(min_len=8, max_len=12):
    chars = string.ascii_uppercase + string.digits + "-"
    ln = random.randint(min_len, max_len)
    return "".join(random.choice(chars) for _ in range(ln))

def rand_docket():
    year2 = random.randint(20, 25)
    typ = random.choice(["cv","cr","mc","md","mj","po","bk","civ"])
    seq = random.randint(10, 9999)
    if random.random()<0.5:
        prefix = f"{random.randint(1,4)}:{year2}"
    else:
        prefix = f"{year2}"
    s = f"{prefix}-{typ}-{seq:04d}"
    if random.random()<0.4:
        s = "No. " + s
    return s

def place(text, slot):
    idx = text.index("{}")
    out = text.replace("{}", slot, 1)
    start = idx
    end = idx + len(slot)
    return out, start, end

CLINICAL_TEMPLATES = [
    "Patient MRN {} presented with dx {} and underwent CPT {}.",
    "Member ID {} authorized for procedure code {} per diagnosis {}.",
    "ICD {} recorded. CPT {} billed. Insurance ID {} verified.",
    "MRN {}: follow-up for condition {}, scheduled CPT {}.",
    "Insurance {} linked; Dx {}, CPT {} documented."
]

LEGAL_TEMPLATES = [
    "Filed under docket {} in S.D.N.Y. related to MRN {}.",
    "Hearing scheduled; {} references patient with Insurance ID {}.",
    "Court noted docket {} and case {}.",
    "Case {} consolidates claims under {}",
    "Prescription {} was entered as evidence under {}."
]

def make_example(ex_id):
    if random.random() < 0.55:
        t = random.choice(CLINICAL_TEMPLATES)
        icd = rand_icd()
        cpt = rand_cpt()
        mrn = rand_id()
        ins = rand_id()
        text, s1, e1 = place(t, mrn)  # MRN
        text, s2, e2 = place(text, icd)  # ICD
        text, s3, e3 = place(text, cpt)  # CPT
        spans = [
            {"start": s1, "end": e1, "label": "MRN"},
            {"start": s2, "end": e2, "label": "ICD_CODE"},
            {"start": s3, "end": e3, "label": "CPT_CODE"},
        ]
        if random.random()<0.5:
            text += f" Insurance ID {ins}."
            spans.append({"start": text.index(ins), "end": text.index(ins)+len(ins), "label": "INSURANCE_ID"})
    else:
        t = random.choice(LEGAL_TEMPLATES)
        docket = rand_docket()
        other = random.choice([rand_id(), rand_id(), rand_id()])
        text = t
        spans = []
        if "{}" in text:
            text, s1, e1 = place(text, docket)
            spans.append({"start": s1, "end": e1, "label": "DOCKET_NO"})
        if "{}" in text:
            text, s2, e2 = place(text, other)
            lab = random.choice(["CASE_NO","PRESCRIPTION_ID","MRN"])
            spans.append({"start": s2, "end": e2, "label": lab})
        if random.random()<0.5:
            extra = rand_id()
            text += f" Case {extra}."
            spans.append({"start": text.index(extra), "end": text.index(extra)+len(extra), "label": "CASE_NO"})
    if False:
        text = perturb(text)
    return {"id": f"ex_{ex_id:05d}", "text": text, "entity_spans": spans}

def write_split(path, n):
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            ex = make_example(i)
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    import argparse, os
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=300)
    ap.add_argument("--dev", type=int, default=60)
    ap.add_argument("--test", type=int, default=60)
    ap.add_argument("--outdir", type=str, default="data")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    write_split(os.path.join(args.outdir, "train.jsonl"), args.train)
    write_split(os.path.join(args.outdir, "dev.jsonl"), args.dev)
    write_split(os.path.join(args.outdir, "test.jsonl"), args.test)
    print("Wrote:", args.outdir)
