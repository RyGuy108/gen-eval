"""Versioned eval sets.

An eval set lives in evalsets/<version>/ as suites.yaml plus reference images. `freeze` writes
MANIFEST.json with a SHA-256 per item (its canonical JSON plus the bytes of any reference image) and
one hash for the whole set. `load` recomputes everything and refuses to run if any item drifted, so
two leaderboards with the same version string are guaranteed to have scored the same prompts.

Each item also has a `render_key`: the hash of only what the model sees (endpoint, prompt, aspect,
duration, reference bytes). A new version that changes how an item is *scored* (new foils, new
checks) keeps the render_key, so its run reuses the earlier renders and costs no credits.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
from dataclasses import dataclass, field

import yaml

ROOT = pathlib.Path(__file__).parents[1] / "evalsets"
SCORERS_FOR = {"text": ["ocr", "clip"], "adherence": ["clip"], "character": ["face", "clip"],
               "video": ["face", "clip"]}


@dataclass(frozen=True)
class Item:
    id: str
    suite: str
    task: str            # image | video
    endpoint: str        # generate | edit | animate
    prompt: str
    aspect_ratio: str = "1:1"
    seconds: float = 0.0
    ref: str | None = None          # path to reference image, relative to the eval set dir
    text: tuple[str, ...] = ()
    scene: str | None = None        # what CLIP should see (defaults to the prompt)
    foils: tuple[str, ...] = ()
    adherence_frames: str = "all"
    checks: tuple = ()              # verifiable facts (see scorers/checks.py), v2+
    hash: str = ""
    render_key: str = ""            # hash of only what the model sees; scoring changes keep it

    @property
    def scorers(self) -> list[str]:
        return SCORERS_FOR[self.suite] + (["checks"] if self.checks else [])

    def canonical(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k not in ("hash", "render_key")}
        d["text"], d["foils"], d["checks"] = list(self.text), list(self.foils), list(self.checks)
        if not d["checks"]:
            del d["checks"]          # keeps v1 item hashes stable
        return d

    def render_inputs(self) -> dict:
        return {"endpoint": self.endpoint, "prompt": self.prompt, "aspect_ratio": self.aspect_ratio,
                "seconds": self.seconds, "ref": self.ref}


@dataclass
class EvalSet:
    version: str
    dir: pathlib.Path
    items: list[Item]
    hash: str
    suites: dict = field(default_factory=dict)

    def by_id(self) -> dict[str, Item]:
        return {i.id: i for i in self.items}

    def ref_path(self, item: Item) -> pathlib.Path | None:
        return self.dir / item.ref if item.ref else None


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _parse(d: pathlib.Path) -> tuple[str, list[Item], dict]:
    raw = yaml.safe_load((d / "suites.yaml").read_text())
    defaults = raw.get("defaults", {})
    items, seen = [], set()
    for sname, s in raw["suites"].items():
        for it in s["items"]:
            if it["id"] in seen:
                raise ValueError(f"duplicate item id {it['id']}")
            seen.add(it["id"])
            ref = f"refs/{it['ref']}.jpg" if it.get("ref") else None
            if ref and not (d / ref).exists():
                raise FileNotFoundError(d / ref)
            if s["task"] == "image" and sname == "text" and not it.get("text"):
                raise ValueError(f"{it['id']}: text items need `text`")
            items.append(Item(
                id=it["id"], suite=sname, task=s["task"], endpoint=s["endpoint"], prompt=it["prompt"],
                aspect_ratio=it.get("aspect_ratio", defaults.get("aspect_ratio", "1:1")),
                seconds=float(it.get("seconds", s.get("seconds", 0))), ref=ref,
                text=tuple(it.get("text", ())), scene=it.get("scene"), foils=tuple(it.get("foils", ())),
                adherence_frames=s.get("adherence_frames", "all"), checks=tuple(it.get("checks", ()))))
    return raw["version"], items, {k: {kk: vv for kk, vv in v.items() if kk != "items"}
                                   for k, v in raw["suites"].items()}


def _hash_items(d: pathlib.Path, items: list[Item]) -> tuple[list[Item], dict[str, str], str]:
    hashed, per = [], {}
    for it in items:
        h = hashlib.sha256(json.dumps(it.canonical(), sort_keys=True).encode())
        rk = hashlib.sha256(json.dumps(it.render_inputs(), sort_keys=True).encode())
        if it.ref:
            ref = (d / it.ref).read_bytes()
            h.update(ref)
            rk.update(ref)
        per[it.id] = h.hexdigest()
        hashed.append(Item(**{**it.__dict__, "hash": per[it.id], "render_key": rk.hexdigest()}))
    whole = _sha(json.dumps(per, sort_keys=True).encode())
    return hashed, per, whole


def freeze(version: str, root: pathlib.Path = ROOT) -> dict:
    d = root / version
    v, items, _ = _parse(d)
    if v != version:
        raise ValueError(f"suites.yaml says {v}, directory is {version}")
    _, per, whole = _hash_items(d, items)
    manifest = {"version": version, "hash": whole, "n_items": len(items), "items": per}
    (d / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load(version: str = "v1", root: pathlib.Path = ROOT, verify: bool = True) -> EvalSet:
    d = root / version
    v, items, suites = _parse(d)
    items, per, whole = _hash_items(d, items)
    if verify:
        mpath = d / "MANIFEST.json"
        if not mpath.exists():
            raise RuntimeError(f"{version} is not frozen; run `python -m geneval.cli freeze {version}`")
        m = json.loads(mpath.read_text())
        changed = sorted(k for k in set(per) | set(m["items"]) if per.get(k) != m["items"].get(k))
        if changed:
            raise RuntimeError(f"eval set {version} was edited after freezing (items {changed}); "
                               f"published versions are immutable, create a new version instead")
    return EvalSet(version=v, dir=d, items=items, hash=whole, suites=suites)
