"""Load an output as a list of PIL frames: one for an image, N evenly spaced for a video."""
from __future__ import annotations

import pathlib

import cv2
from PIL import Image

VIDEO_EXT = {".mp4", ".mov", ".webm", ".mkv"}


def is_video(path: str | pathlib.Path) -> bool:
    return pathlib.Path(path).suffix.lower() in VIDEO_EXT


def sample_frames(path: str | pathlib.Path, n: int = 8) -> list[Image.Image]:
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    # skip the very first frame: image-to-video models return the start image verbatim there,
    # which would inflate identity scores
    idx = sorted({min(total - 1, round(1 + i * (total - 2) / max(1, n - 1))) for i in range(n)})
    frames = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = cap.read()
        if ok:
            frames.append(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
    cap.release()
    if not frames:
        raise ValueError(f"could not decode {path}")
    return frames


def load(path: str | pathlib.Path, n_frames: int = 8) -> list[Image.Image]:
    if is_video(path):
        return sample_frames(path, n_frames)
    return [Image.open(path).convert("RGB")]
