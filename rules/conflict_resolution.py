from typing import List, Tuple

PREFERRED = ["ICD_CODE","CPT_CODE","MRN","INSURANCE_ID","DOCKET_NO","PRESCRIPTION_ID","CASE_NO"]
PRIORITY = {lab:i for i,lab in enumerate(PREFERRED)}

def resolve(spans: List[Tuple[int,int,str]]) -> List[Tuple[int,int,str]]:
    spans = sorted(spans, key=lambda x: (x[0], x[1]-x[0], PRIORITY.get(x[2], 999)))
    out = []
    for s,e,l in spans:
        keep = True
        for i,(ss,ee,ll) in enumerate(out):
            if not (e <= ss or s >= ee):  # overlap
                if PRIORITY.get(l,999) < PRIORITY.get(ll,999) or (e-s) > (ee-ss):
                    out[i] = (s,e,l)
                    keep = False
                    break
                else:
                    keep = False
                    break
        if keep:
            out.append((s,e,l))
    return sorted(out, key=lambda x: x[0])
