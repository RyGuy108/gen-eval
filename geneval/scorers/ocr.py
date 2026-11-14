"""Text legibility: can OCR read back the exact text the prompt asked for?

RapidOCR (PP-OCR detection + recognition, ONNX) reads every text line. For each required string we
find the best *approximate substring* match in the OCR output (Sellers' edit-distance DP, so the
target may sit inside a longer line like "OPEN - MORNING BREW CAFE - 7AM") and score

    char_acc = 1 - edit_distance(target, best_substring) / len(target)

after normalizing case, punctuation and whitespace. A garbled sign ("MORNIG BRWE CAFE") lands around
0.8, a missing one at 0. `exact` is true only if every required string is read back perfectly, which
is the bar a customer printing a poster actually has.

Models also invent text nobody asked for: fake menu boards, lorem-ipsum signage, garbled small print.
`gibberish` counts OCR'd words (4+ letters, not part of a target) that are not English words by
wordfreq's frequency list. Real background text ("COFFEE", "PASTRIES") is fine; "TUNCKEOGFHCK" is
the artifact customers screenshot and complain about.

Two fixes came out of checking scores against the rendered images:
  * reading order: receipts and menus put prices in a right-hand column, and sorting boxes by a
    rounded y split "2 TACOS ... 8.50" into separate rows. Boxes are now clustered into rows by
    vertical overlap and each row is read left to right.
  * multi-view: heavy slanted thumbnail fonts ("I TRIED 7 AI MODELS") are often missed by the text
    detector on the color image but found on the grayscale or inverted image with a looser box
    threshold. Each target keeps its best reading across views. OCR does not invent the exact target
    string, so taking the max is safe; gibberish is counted on the primary view only.
OCR-confusable glyphs (0/O, 1/I/L, 5/S, 8/B, 2/Z) are folded before matching.
"""
from __future__ import annotations

import re

import numpy as np
from PIL import Image, ImageOps
from wordfreq import zipf_frequency


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


_FOLD = str.maketrans({"0": "O", "Q": "O", "1": "I", "L": "I", "5": "S", "8": "B", "2": "Z"})


def char_accuracy(target: str, text: str) -> float:
    t, x = normalize(target), normalize(text)
    if not t:
        return 1.0
    # Signs often break one phrase over two lines and OCR drops or adds spaces, so we also compare
    # with spaces removed, and with OCR-confusable glyphs folded; the best reading wins.
    best = 0.0
    for tt, xx in ((t, x), (t.replace(" ", ""), x.replace(" ", ""))):
        for a, b in ((tt, xx), (tt.translate(_FOLD), xx.translate(_FOLD))):
            best = max(best, 1 - substring_distance(a, b) / len(a))
    return float(max(0.0, best))


def _is_word(w: str) -> bool:
    return zipf_frequency(w.lower(), "en") >= 1.0


def _two_words(w: str) -> bool:
    """OCR often drops a space ("ITRIED", "NETWT"): accept a split into two common words."""
    return any(zipf_frequency(w[:i].lower(), "en") >= 2.5 and zipf_frequency(w[i:].lower(), "en") >= 2.5
               for i in range(3, len(w) - 2))


def gibberish_words(text: str, targets: list[str]) -> list[str]:
    """Invented non-words. Misreadings of the requested text are excluded here (they already cost
    legibility); this only counts junk the model added on its own."""
    want = normalize(" ".join(targets)).replace(" ", "")
    out = []
    for w in normalize(text).split():
        w = w.strip("'")
        if len(w) < 4 or not w.isalpha() or _is_word(w) or _two_words(w):
            continue
        if want and substring_distance(w, want) <= max(1, len(w) // 4):
            continue
        out.append(w)
    return out


class OcrScorer:
    def __init__(self, min_conf: float = 0.5):
        from rapidocr_onnxruntime import RapidOCR
        self.engine = RapidOCR()
        self.min_conf = min_conf

    def boxes(self, img: Image.Image, **kw) -> list[tuple[float, float, float, str]]:
        """[(y_center, height, x_left, text)] for confident detections."""
        arr = np.asarray(img.convert("RGB"))[:, :, ::-1].copy()
        res, _ = self.engine(arr, **kw)
        out = []
        for box, txt, conf in res or []:
            if float(conf) >= self.min_conf:
                ys = [p[1] for p in box]
                out.append(((min(ys) + max(ys)) / 2, max(ys) - min(ys), min(p[0] for p in box), txt))
        return out

    @staticmethod
    def reading_order(boxes) -> str:
        """Cluster boxes into rows by vertical overlap, read each row left to right."""
        rows: list[list] = []
        for b in sorted(boxes, key=lambda b: b[0]):
            for row in rows:
                if abs(row[0][0] - b[0]) <= 0.5 * min(row[0][1], b[1]):
                    row.append(b)
                    break
            else:
                rows.append([b])
        return " ".join(b[3] for row in rows for b in sorted(row, key=lambda b: b[2]))

    @staticmethod
    def row_pairs(boxes) -> list[str]:
        """Two boxes on roughly the same (possibly tilted) line, left one first: catches
        "2 TACOS" + "8.50" on a receipt photographed at an angle, where strict rows split them."""
        out = []
        for a in boxes:
            for b in boxes:
                if b[2] > a[2] and abs(a[0] - b[0]) <= 1.2 * max(a[1], b[1]):
                    out.append(f"{a[3]} {b[3]}")
        return out

    def read(self, img: Image.Image, **kw) -> list[tuple[str, float]]:
        return [(self.reading_order(self.boxes(img, **kw)), 1.0)]

    def views(self, img: Image.Image) -> list[str]:
        loose = dict(box_thresh=0.3, unclip_ratio=2.0, text_score=0.3)
        out = []
        for i, (im, kw) in enumerate(((img, {}), (ImageOps.grayscale(img).convert("RGB"), loose),
                                      (ImageOps.invert(img.convert("RGB")), loose))):
            bx = self.boxes(im, **kw)
            out.append(self.reading_order(bx))
            out.extend(self.row_pairs(bx) if i == 0 else [])
        return out

    def score(self, imgs: list[Image.Image], targets: list[str]) -> dict:
        best = {"legibility": -1.0}
        for im in imgs:                          # video: best frame (text may appear mid-clip)
            views = self.views(im)
            text = views[0]
            per = [max(char_accuracy(t, v) for v in views) for t in targets]
            junk = gibberish_words(text, targets)
            r = {"legibility": float(np.mean(per)), "per_target": per,
                 "exact": all(p >= 0.999 for p in per), "gibberish": len(junk),
                 "gibberish_words": junk[:20], "ocr_text": text[:400]}
            if r["legibility"] > best["legibility"]:
                best = r
        return best
