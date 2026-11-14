"""The models under evaluation, one Target per (suite endpoint, model, settings).

`list_credits` is the documented "from N credits" price; the leaderboard always uses the
credits_charged the API actually reported for each render, and flags any model whose observed
price differs from the list price.
"""
from __future__ import annotations

from dataclasses import dataclass

from .providers.magichour import Target


@dataclass(frozen=True)
class ModelEntry:
    name: str
    list_credits: float
    generate: Target | None = None   # text-to-image
    edit: Target | None = None       # reference image + prompt
    animate: Target | None = None    # image-to-video


def _img(model: str, res: str, list_credits: float, edit_model: str | None = "same") -> ModelEntry:
    em = model if edit_model == "same" else edit_model
    return ModelEntry(
        name=model, list_credits=list_credits,
        generate=Target(f"gen:{model}@{res}", "ai_image_generator", model, res),
        edit=Target(f"edit:{em}@{res}", "ai_image_editor", em, res) if em else None)


def _vid(model: str, res: str, list_credits: float, seconds: float = 5) -> ModelEntry:
    return ModelEntry(name=model, list_credits=list_credits,
                      animate=Target(f"i2v:{model}@{res}/{seconds:g}s", "image_to_video", model, res, seconds))


IMAGE = [
    _img("nano-banana-pro", "1k", 150),
    _img("nano-banana-2", "1k", 50),
    _img("seedream-v4.5", "1k", 50),
    _img("qwen-image-2.1", "1k", 10, edit_model="qwen-edit"),
    _img("flux-2-klein", "1k", 5),
    _img("z-image-turbo", "1k", 5, edit_model=None),       # no edit endpoint
]

VIDEO = [
    _vid("kling-3.0", "720p", 0),
    _vid("seedance-2.0-mini", "480p", 0),
    _vid("ltx-2.5", "480p", 0),
]

# Evaluated in the first probe only (charged 100 credits at 1k against a 50 list price); add it back
# to IMAGE to include it in a full run.
EXTRA = [
    _img("gpt-image-2", "1k", 50),
]

ALL = {m.name: m for m in IMAGE + VIDEO + EXTRA}


def targets_for(endpoint: str, models: list[str] | None = None) -> list[Target]:
    out = []
    pool = IMAGE + VIDEO + [m for m in EXTRA if models and m.name in models]
    for m in pool:
        if models and m.name not in models:
            continue
        t = getattr(m, endpoint)
        if t is not None:
            out.append(t)
    return out


def model_of(target_key: str) -> str:
    return target_key.split(":", 1)[1].split("@", 1)[0]
