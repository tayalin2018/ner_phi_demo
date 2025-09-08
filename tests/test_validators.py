from rules.validators import find_icd, find_cpt, find_docket, find_case, find_prescription, find_mrn, find_insurance_id

def span_text(text, spans):
    return [text[s:e] for s,e,_ in spans]

def test_icd_basic():
    t = "dx E11.9 and A00 plus Z99"
    found = span_text(t, find_icd(t))
    assert "E11.9" in found and "A00" in found and "Z99" in found

def test_cpt_basic():
    t = "codes 99213 and 123AB-26"
    found = span_text(t, find_cpt(t))
    assert "99213" in found and "123AB-26" in found

def test_docket():
    t = "See No. 1:23-cv-0042 and 23-cr-120"
    found = span_text(t, find_docket(t))
    assert "No. 1:23-cv-0042" in found and "23-cr-120" in found

def test_case_and_rx():
    t = "Case No. ABC-12345 and Prescription ID Z9Q-77"
    assert "ABC-12345" in span_text(t, find_case(t))[0]
    assert "Z9Q-77" in span_text(t, find_prescription(t))[0]

def test_mrn_insurance():
    t = "MRN 004-77-9123, Member ID AB12-9Z7K"
    mrn = span_text(t, find_mrn(t))
    ins = span_text(t, find_insurance_id(t))
    assert "004-77-9123" in mrn[0]
    assert "AB12-9Z7K" in ins[0]
