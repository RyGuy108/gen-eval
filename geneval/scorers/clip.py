"""Prompt adherence with CLIP (ONNX, no torch at inference).

Two signals per output:
  clipscore    2.5 * max(cos(image, prompt), 0)  (Hessel et al. 2021). Good at "is this roughly the
               right scene", blind to counts, bindings and spatial relations.
  contrastive  the eval set gives each adherence prompt hand-written *foils*: the same caption with one
               fact broken (two cats -> three cats, red cube left of blue -> blue cube left of red). We
               softmax the image's similarity over [prompt] + foils with CLIP's logit scale and keep the
               probability of the true prompt. A model that draws a generic scene scores ~1/(k+1); one
               that gets the specific detail right scores near 1. This is what separates models that
               all have similar raw CLIPScore.
"""
from __future__ import annotations

import os
import pathlib

import numpy as np
from PIL import Image

from .clip_tokenizer import ClipTokenizer

MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
LOGIT_SCALE = 100.0
WEIGHTS = pathlib.Path(os.environ.get("GENEVAL_WEIGHTS", pathlib.Path(__file__).parents[2] / "weights"))


def preprocess(img: Image.Image, size: int = 224) -> np.ndarray:
    img = img.convert("RGB")
    w, h = img.size
    s = size / min(w, h)
    img = img.resize((max(size, round(w * s)), max(size, round(h * s))), Image.BICUBIC)
    w, h = img.size
    left, top = (w - size) // 2, (h - size) // 2
    img = img.crop((left, top, left + size, top + size))
    x = (np.asarray(img, dtype=np.float32) / 255.0 - MEAN) / STD
    return x.transpose(2, 0, 1)


def _norm(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


class ClipScorer:
    def __init__(self, arch: str | None = None, threads: int = 0):
        import onnxruntime as ort
        arch = arch or os.environ.get("GENEVAL_CLIP", "ViT-L-14")
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        d = WEIGHTS / "clip"
        self.visual = ort.InferenceSession(str(d / f"{arch}-visual.onnx"), so, providers=["CPUExecutionProvider"])
        self.textual = ort.InferenceSession(str(d / f"{arch}-textual.onnx"), so, providers=["CPUExecutionProvider"])
        self.size = self.visual.get_inputs()[0].shape[-1]
        self.size = self.size if isinstance(self.size, int) else 224
        self.tok = ClipTokenizer()
        self.arch = arch
        self._tcache: dict[str, np.ndarray] = {}

    def embed_images(self, imgs: list[Image.Image]) -> np.ndarray:
        x = np.stack([preprocess(i, self.size) for i in imgs])
        return _norm(self.visual.run(None, {self.visual.get_inputs()[0].name: x})[0])

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        missing = [t for t in dict.fromkeys(texts) if t not in self._tcache]
        if missing:
            ids = self.tok(missing)
            out = _norm(self.textual.run(None, {self.textual.get_inputs()[0].name: ids})[0])
            self._tcache.update(zip(missing, out))
        return np.stack([self._tcache[t] for t in texts])

    def score(self, imgs: list[Image.Image], prompt: str, foils: list[str] | None = None) -> dict:
        """Score one output (an image, or several frames of one video, averaged in embedding space
        for the contrastive test and per frame for clipscore)."""
        iv = self.embed_images(imgs)
        tv = self.embed_texts([prompt] + list(foils or []))
        cos = iv @ tv.T                                   # frames x (1 + k)
        out = {"clipscore": float(2.5 * np.clip(cos[:, 0], 0, None).mean()),
               "cos": float(cos[:, 0].mean())}
        if foils:
            mean_iv = _norm(iv.mean(0, keepdims=True))
            logits = LOGIT_SCALE * (mean_iv @ tv.T)[0]
            p = np.exp(logits - logits.max())
            p /= p.sum()
            out["contrastive"] = float(p[0])
            out["beats_all_foils"] = bool(logits[0] > logits[1:].max())
            out["n_foils"] = len(foils)
        return out
