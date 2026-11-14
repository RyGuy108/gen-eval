"""Character consistency: does the output show the same person as the reference?

YuNet detects faces and 5 landmarks, SFace aligns and embeds each face to 128-d, and the score is
cosine similarity to the reference identity. SFace's published threshold for "same person" is 0.363
(LFW); we report the raw similarity plus that pass/fail, and for multi-face outputs we take the best
matching face (the character may share the frame with others).

Failure modes are scored, not skipped: no face detected -> similarity 0, so a model that drops the
character or turns the face away is penalized rather than silently ignored.
"""
from __future__ import annotations

import os
import pathlib

import cv2
import numpy as np
from PIL import Image

WEIGHTS = pathlib.Path(os.environ.get("GENEVAL_WEIGHTS", pathlib.Path(__file__).parents[2] / "weights"))
SFACE_SAME = 0.363


def _bgr(img: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(img, np.ndarray):
        return img
    return cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR)


class FaceScorer:
    def __init__(self, det_score: float = 0.6, max_side: int = 640):
        d = WEIGHTS / "face"
        self.det = cv2.FaceDetectorYN.create(str(d / "face_detection_yunet_2023mar.onnx"), "",
                                             (320, 320), det_score, 0.3, 5000)
        self.rec = cv2.FaceRecognizerSF.create(str(d / "face_recognition_sface_2021dec.onnx"), "")
        self.max_side = max_side
        self._ref: dict[str, np.ndarray] = {}

    def faces(self, img) -> list[tuple[np.ndarray, np.ndarray]]:
        """[(embedding, detection_row)] for every face, largest first."""
        im = _bgr(img)
        h, w = im.shape[:2]
        s = min(1.0, self.max_side / max(h, w))
        if s < 1.0:
            im = cv2.resize(im, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
            h, w = im.shape[:2]
        self.det.setInputSize((w, h))
        _, dets = self.det.detect(im)
        if dets is None:
            return []
        dets = sorted(dets, key=lambda d: -d[2] * d[3])
        out = []
        for d in dets:
            aligned = self.rec.alignCrop(im, d)
            e = self.rec.feature(aligned).flatten()
            out.append((e / np.linalg.norm(e), d))
        return out

    def reference(self, path: str | pathlib.Path) -> np.ndarray:
        key = str(path)
        if key not in self._ref:
            fs = self.faces(Image.open(path))
            if not fs:
                raise ValueError(f"no face in reference {path}")
            self._ref[key] = fs[0][0]
        return self._ref[key]

    def score(self, imgs: list, ref_path: str | pathlib.Path) -> dict:
        """imgs: one image, or sampled frames of a video. For video we report the mean and the worst
        frame, because identity drift mid-clip is the failure customers notice."""
        ref = self.reference(ref_path)
        sims, found = [], 0
        for im in imgs:
            fs = self.faces(im)
            if fs:
                found += 1
                sims.append(max(float(e @ ref) for e, _ in fs))
            else:
                sims.append(0.0)
        sims = np.array(sims)
        return {"face_sim": float(sims.mean()), "face_sim_min": float(sims.min()),
                "face_found": found / len(imgs), "same_person": bool(sims.mean() >= SFACE_SAME)}
