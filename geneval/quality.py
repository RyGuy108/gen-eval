"""Turn raw scorer metrics into one quality number in [0, 1] and a pass/fail per output.

The pass bars are what a paying customer would accept without regenerating:
  text       every required string read back with >= 90% character accuracy (a single OCR slip on a
             stylized font is tolerated, a misspelled word is not)
  adherence  the prompt beats every foil in the contrastive CLIP test
  character  SFace says same person (cos >= 0.363) AND the requested scene beats its foils, including
             the "plain studio headshot" foil, so returning the reference photo fails
  video      the *worst* sampled frame is still the same person, and the clip ends in the asked state

Quality is a geometric mean where two things must both hold, so a model can't buy points on one
axis with total failure on the other. Bars are versioned with the scorers; calibrate.py checks
them against hand labels.
"""
from __future__ import annotations

import math

QUALITY_VERSION = "q1"
TEXT_PASS = 0.90
FACE_SAME = 0.363
FACE_FULL = 0.60      # similarity at which identity counts as fully preserved


def _face01(s: float) -> float:
    return max(0.0, min(1.0, s / FACE_FULL))


def judge(suite: str, m: dict) -> tuple[float, bool]:
    """m is {scorer: metrics} for one generation."""
    if suite == "text":
        per = m["ocr"]["per_target"]
        return float(m["ocr"]["legibility"]), bool(min(per) >= TEXT_PASS)
    if suite == "adherence":
        c = m["clip"]
        return float(c["contrastive"]), bool(c["beats_all_foils"])
    if suite == "character":
        f, c = m["face"], m["clip"]
        q = math.sqrt(_face01(f["face_sim"]) * c["contrastive"])
        return q, bool(f["face_sim"] >= FACE_SAME and c["beats_all_foils"])
    if suite == "video":
        f, c = m["face"], m["clip"]
        ident = 0.5 * (f["face_sim"] + f["face_sim_min"])
        q = math.sqrt(_face01(ident) * c["contrastive"])
        return q, bool(f["face_sim_min"] >= FACE_SAME and c["beats_all_foils"])
    raise ValueError(suite)


def bars(suite: str, m: dict) -> dict[str, int]:
    """Pass/fail at the three quality bars the model router serves (draft < standard < premium).

    draft     usable as a rough draft: quality >= 0.5
    standard  the suite's normal pass bar (judge() above)
    premium   standard plus a stricter, suite-specific margin: no invented gibberish text; prompt wins
              its foils by a clear CLIP margin; identity well above the same-person threshold
    """
    q, ok = judge(suite, m)
    if suite == "text":
        prem = ok and m["ocr"].get("exact", False) and m["ocr"].get("gibberish", 0) == 0
    elif suite == "adherence":
        prem = ok and m["clip"].get("margin", 0) >= 0.02
    elif suite == "character":
        prem = ok and m["face"]["face_sim"] >= 0.5
    else:
        prem = ok and m["face"]["face_sim_min"] >= 0.45
    return {"draft": int(q >= 0.5), "standard": int(ok), "premium": int(prem)}
