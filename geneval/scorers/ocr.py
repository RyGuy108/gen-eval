"""Text legibility: can OCR read back the exact text the prompt asked for?

RapidOCR (PP-OCR detection + recognition, ONNX) reads every text line. For each required string we
find the best *approximate substring* match in the OCR output (Sellers' edit-distance DP, so the
target may sit inside a longer line like "OPEN - MORNING BREW CAFE - 7AM") and score

    char_acc = 1 - edit_distance(target, best_substring) / len(target)

after normalizing case, punctuation and whitespace. A garbled sign ("MORNIG BRWE CAFE") lands around
0.8, a missing one at 0. `exact` is true only if every required string is read back perfectly, which
is the bar a customer printing a poster actually has.
"""
from __future__ import annotations

import re

import numpy as np
from PIL import Image


def normalize(s: str) -> str:
    s = s.upper().replace("&", " AND ").replace("’", "'")
    s = re.sub(r"[^A-Z0-9' ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def substring_distance(pattern: str, text: str) -> int:
    """Minimum edit distance between `pattern` and any substring of `text` (Sellers 1980)."""
    if not pattern:
        return 0
    prev = np.zeros(len(text) + 1, dtype=np.int32)          # free start anywhere in text
    for i, pc in enumerate(pattern, 1):
        cur = np.empty_like(prev)
        cur[0] = i
        for j, tc in enumerate(text, 1):
            cur[j] = min(prev[j - 1] + (pc != tc), prev[j] + 1, cur[j - 1] + 1)
        prev = cur
    return int(prev.min())                                    # free end anywhere in text


def char_accuracy(target: str, text: str) -> float:
    t = normalize(target)
    if not t:
        return 1.0
    # Signs often break one phrase over two lines; spaces are compared too, so we also try the
    # text with spaces removed on both sides and keep the better reading.
    a = 1 - substring_distance(t, normalize(text)) / len(t)
    b = 1 - substring_distance(t.replace(" ", ""), normalize(text).replace(" ", "")) / len(t.replace(" ", ""))
    return float(max(0.0, a, b))


class OcrScorer:
    def __init__(self, min_conf: float = 0.5):
        from rapidocr_onnxruntime import RapidOCR
        self.engine = RapidOCR()
        self.min_conf = min_conf

    def read(self, img: Image.Image) -> list[tuple[str, float]]:
        arr = np.asarray(img.convert("RGB"))[:, :, ::-1].copy()
        res, _ = self.engine(arr)
        if not res:
            return []
        # reading order: top-to-bottom, then left-to-right
        res = sorted(res, key=lambda r: (round(min(p[1] for p in r[0]) / 20), min(p[0] for p in r[0])))
        return [(txt, float(conf)) for _, txt, conf in res if float(conf) >= self.min_conf]

    def score(self, imgs: list[Image.Image], targets: list[str]) -> dict:
        best = {"legibility": -1.0}
        for im in imgs:                          # video: best frame (text may appear mid-clip)
            lines = self.read(im)
            text = " ".join(t for t, _ in lines)
            per = [char_accuracy(t, text) for t in targets]
            r = {"legibility": float(np.mean(per)), "per_target": per,
                 "exact": all(p >= 0.999 for p in per), "ocr_text": text[:400]}
            if r["legibility"] > best["legibility"]:
                best = r
        return best
