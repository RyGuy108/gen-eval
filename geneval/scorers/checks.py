"""Prompt adherence as verifiable facts (GenEval-style), instead of whole-image CLIP similarity.

Calibration on v1 showed contrastive CLIP is worse than chance at counts and attribute binding
(AUC 0.42 vs hand labels): it scores a bag of words, so "three red apples and one green pear" and
"two red apples and two green pears" look the same to it. Following GenEval (Ghosh et al. 2023), each
adherence item lists the facts a correct image must satisfy, and each fact is checked with the tool
that is actually reliable for it:

  count     open-vocabulary detector (YOLOE, text-prompted) boxes for the object, after NMS
  color     dominant color of the object's segmentation mask (median hue of saturated pixels in HSV,
            brightness/saturation for black / white / gray / brown). CLIP on the box crop was tried
            first and called shadowed red apples "gray".
  position  box centers: left_of / right_of / above / below
  on_top_of A's box center sits in the upper part of B's box (bird on a bear's head, cat on books)
  wears     the accessory's box overlaps A more than any other listed animal / person
  style     CLIP zero-shot on the whole image over the given alternatives (CLIP is good at style)

Score = fraction of facts satisfied; pass = all facts satisfied. Each fact's result is stored so a
leaderboard can say *why* a model fails (it miscounts vs it binds colors wrong).
"""
from __future__ import annotations

import os
import pathlib

import numpy as np
from PIL import Image

from .clip import WEIGHTS

COLORS = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white", "gray"]


def _install_clip_tokenizer_shim() -> None:
    """Ultralytics imports the `clip` package only for `clip.clip.tokenize` when it builds YOLOE's
    MobileCLIP text prompts. MobileCLIP uses the standard CLIP BPE, which this repo already ships
    (clip_tokenizer.py), so we provide that function instead of installing the package."""
    import sys
    import types

    if "clip" in sys.modules:
        return
    import torch

    from .clip_tokenizer import ClipTokenizer
    tok = ClipTokenizer()

    def tokenize(texts, context_length: int = 77, truncate: bool = True):
        texts = [texts] if isinstance(texts, str) else list(texts)
        return torch.from_numpy(tok(texts, context_length))

    mod = types.ModuleType("clip")
    mod.clip = types.SimpleNamespace(tokenize=tokenize)
    mod.tokenize = tokenize
    sys.modules["clip"] = mod


class Detector:
    """Text-prompted open-vocabulary detector (YOLOE). Lazy import: torch is only needed when
    adherence checks run."""

    def __init__(self, weights: str | None = None, conf: float = 0.02, iou: float = 0.7):
        _install_clip_tokenizer_shim()
        from ultralytics import YOLOE
        w = weights or os.environ.get("GENEVAL_DETECTOR", str(WEIGHTS / "det" / "yoloe-11l-seg.pt"))
        self.model = YOLOE(w)
        self.conf, self.iou = conf, iou
        self._classes: tuple[str, ...] = ()

    def detect(self, img: Image.Image, names: list[str]) -> dict[str, list[dict]]:
        names = tuple(dict.fromkeys(names))
        if names != self._classes:
            self.model.set_classes(list(names), self.model.get_text_pe(list(names)))
            self._classes = names
        r = self.model.predict(img, conf=self.conf, iou=self.iou, verbose=False, agnostic_nms=False)[0]
        out = {n: [] for n in names}
        masks = None
        if r.masks is not None:
            masks = r.masks.data.cpu().numpy() > 0.5          # n x H' x W' at inference size
        for i, (box, cls, conf) in enumerate(zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())):
            d = {"box": [float(v) for v in box], "conf": float(conf)}
            if masks is not None:
                d["mask"] = masks[i]
            out[names[int(cls)]].append(d)
        return out


def dominant_color(px: np.ndarray) -> str:
    """px: N x 3 HSV in PIL's 0-255 ranges. Vote over pixels; highlights and deep shadows abstain."""
    h, s, v = px[:, 0] * 360 / 255, px[:, 1] / 255, px[:, 2] / 255
    lit = (v > 0.12) & (v < 0.97)
    chroma = lit & (s >= 0.3)
    votes = {}
    brown = chroma & (h >= 10) & (h < 35) & (v < 0.6)
    for name, lo, hi in (("red", 345, 361), ("red", 0, 10), ("orange", 10, 40), ("yellow", 40, 70),
                         ("green", 70, 170), ("blue", 170, 260), ("purple", 260, 300), ("pink", 300, 345)):
        votes[name] = votes.get(name, 0) + int((chroma & ~brown & (h >= lo) & (h < hi)).sum())
    votes["brown"] = int(brown.sum())
    # An object with a real color usually also has highlights, shadows and white patches (a blue
    # bird's belly); if a fair share of *all* its pixels is chromatic, only chromatic names compete.
    # (The share is over all pixels so a black cat's yellow eyes don't make it a yellow cat.)
    if chroma.sum() >= 0.25 * len(px):
        return max(votes, key=votes.get)
    grey = lit & ~chroma
    votes = {"black": int((v <= 0.12).sum()), "white": int((grey & (v >= 0.75)).sum()),
             "gray": int((grey & (v < 0.75)).sum())}
    return max(votes, key=votes.get)


COUNT_FLOOR = 0.05      # below this a box is noise even for rare / stylized classes
COUNT_REL = 0.5         # an instance must score at least half the best instance of its class


def _instances(ds: list[dict]) -> list[dict]:
    """Open-vocabulary scores are calibrated per class (a giraffe scores 0.97, a watercolor fox 0.07),
    so an absolute threshold either misses rare classes or overcounts common ones. Count the boxes
    that are within COUNT_REL of the class's best box instead."""
    if not ds:
        return []
    top = max(d["conf"] for d in ds)
    if top < COUNT_FLOOR:
        return []
    return [d for d in ds if d["conf"] >= max(COUNT_FLOOR, COUNT_REL * top)]


def _iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a["box"]
    bx0, by0, bx1, by1 = b["box"]
    iw, ih = max(0, min(ax1, bx1) - max(ax0, bx0)), max(0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    return inter / max(1e-6, (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter)


def _dedupe_countables(dets: dict, countable: set[str]) -> dict:
    """One physical object must not count as both an apple and a pear: among the classes being
    counted, a box that overlaps a more confident box of another counted class is dropped.
    Wearables (scarf, raincoat) are left alone, since they legitimately sit inside another box."""
    pool = sorted(((n, d) for n in countable for d in dets.get(n, [])), key=lambda x: -x[1]["conf"])
    kept: list[tuple[str, dict]] = []
    for n, d in pool:
        if all(not (n2 != n and _iou(d, d2) > 0.6) for n2, d2 in kept):
            kept.append((n, d))
    out = {n: v for n, v in dets.items() if n not in countable}
    for n in countable:
        out[n] = [d for n2, d in kept if n2 == n]
    return out


def _center(b):
    x0, y0, x1, y1 = b["box"]
    return (x0 + x1) / 2, (y0 + y1) / 2


def _overlap(a, b) -> float:
    ax0, ay0, ax1, ay1 = a["box"]
    bx0, by0, bx1, by1 = b["box"]
    iw, ih = max(0, min(ax1, bx1) - max(ax0, bx0)), max(0, min(ay1, by1) - max(ay0, by0))
    return iw * ih / max(1e-6, (ax1 - ax0) * (ay1 - ay0))


class CheckScorer:
    def __init__(self, clip, detector: Detector | None = None):
        self.clip = clip
        self.det = detector or Detector()

    def color_of(self, img: Image.Image, det: dict, obj: str = "") -> str:
        hsv = np.asarray(img.convert("HSV"), dtype=np.float32)
        if "mask" in det:
            m = np.asarray(Image.fromarray(det["mask"].astype(np.uint8) * 255).resize(img.size)) > 127
        else:
            m = np.zeros(hsv.shape[:2], bool)
            x0, y0, x1, y1 = (int(v) for v in det["box"])
            m[y0:y1, x0:x1] = True
        px = hsv[m]
        if len(px) < 20:
            return "unknown"
        return dominant_color(px)

    def score(self, imgs: list[Image.Image], checks: list[dict]) -> dict:
        img = imgs[-1] if len(imgs) > 1 else imgs[0]
        objects = set()
        for c in checks:
            for k in ("object", "a", "b", "item"):
                if isinstance(c.get(k), str):
                    objects.add(c[k])
            objects.update(c.get("among", []))
        dets = self.det.detect(img, sorted(objects)) if objects else {}
        dets = _dedupe_countables(dets, {c["object"] for c in checks if c["check"] == "count"})
        results = []
        for c in checks:
            kind = c["check"]
            ok, detail = False, {}
            if kind == "count":
                found = _instances(dets.get(c["object"], []))
                if "color" in c:
                    colors = [self.color_of(img, d, c["object"]) for d in found]
                    detail["colors"] = colors
                    n = sum(col == c["color"] for col in colors)
                    # the other instances must not be that color either (binding goes both ways)
                else:
                    n = len(found)
                detail["found"] = n
                ok = n == c["n"]
            elif kind == "color":
                found = sorted(dets.get(c["object"], []), key=lambda d: -d["conf"])
                if found:
                    col = self.color_of(img, found[0], c["object"])
                    detail["color"] = col
                    want = c["color"] if isinstance(c["color"], list) else [c["color"]]
                    ok = col in want
            elif kind in ("position", "on_top_of"):
                A = sorted(dets.get(c["a"], []), key=lambda d: -d["conf"])[:1]
                B = sorted(dets.get(c["b"], []), key=lambda d: -d["conf"])[:1]
                if A and B:
                    (ax, ay), (bx, by) = _center(A[0]), _center(B[0])
                    bx0, by0, bx1, by1 = B[0]["box"]
                    if kind == "on_top_of":
                        ok = bx0 <= ax <= bx1 and ay <= by0 + 0.5 * (by1 - by0)
                    else:
                        ok = {"left_of": ax < bx, "right_of": ax > bx, "above": ay < by, "below": ay > by}[c["rel"]]
                    detail = {"a": [round(ax), round(ay)], "b": [round(bx), round(by)]}
            elif kind == "wears":
                items = sorted(dets.get(c["item"], []), key=lambda d: -d["conf"])[:1]
                if items:
                    scores = {who: max([_overlap(items[0], d) for d in dets.get(who, [])] or [0.0])
                              for who in c["among"]}
                    detail["overlap"] = {k: round(v, 2) for k, v in scores.items()}
                    ok = max(scores, key=scores.get) == c["a"] and scores[c["a"]] > 0
            elif kind == "style":
                opts = [c["style"]] + c["vs"]
                e = self.clip.embed_images([img])
                t = self.clip.embed_texts([f"a {o}" for o in opts])
                detail["pick"] = opts[int(np.argmax(e @ t.T))]
                ok = detail["pick"] == c["style"]
            else:
                raise ValueError(kind)
            results.append({"check": c, "ok": bool(ok), **detail})
        frac = float(np.mean([r["ok"] for r in results])) if results else 1.0
        return {"facts": results, "fact_score": frac, "all_facts": all(r["ok"] for r in results)}
