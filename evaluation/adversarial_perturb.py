import random

ZWSP = "\u200b"

def insert_zwsp(text: str, prob: float = 0.1) -> str:
    out = []
    for ch in text:
        out.append(ch)
        if random.random() < prob and ch.isalnum():
            out.append(ZWSP)
    return "".join(out)

def homoglyph_digits(text: str, prob: float = 0.15) -> str:
    trans = str.maketrans({
        "0":"０","1":"１","2":"２","3":"３","4":"４",
        "5":"５","6":"６","7":"７","8":"８","9":"９"
    })
    out = []
    for ch in text:
        if ch.isdigit() and random.random() < prob:
            out.append(ch.translate(trans))
        else:
            out.append(ch)
    return "".join(out)

def perturb(text: str) -> str:
    t = insert_zwsp(text, 0.08)
    t = homoglyph_digits(t, 0.1)
    return t
