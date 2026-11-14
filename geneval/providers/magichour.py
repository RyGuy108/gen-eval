"""Magic Hour provider: one place that knows how each (endpoint, model) is called and polled.

Every create() returns a project id immediately; the caller persists it before polling so a
crash or failed download never loses a render that was already paid for. Downloads are
re-fetched from image_projects.get / video_projects.get on demand.
"""
from __future__ import annotations

import os
import pathlib
import time
import urllib.request
from dataclasses import dataclass, field

FINAL = {"complete", "error", "canceled"}


@dataclass(frozen=True)
class Target:
    """A model under evaluation, as one concrete API call configuration."""
    key: str                 # stable id used in results, e.g. "img:nano-banana-2@1k"
    endpoint: str            # ai_image_generator | ai_image_editor | image_to_video
    model: str
    resolution: str
    seconds: float = 0.0     # video only
    extra: dict = field(default_factory=dict)

    @property
    def kind(self) -> str:
        return "video" if self.endpoint == "image_to_video" else "image"


def _client():
    import magic_hour
    return magic_hour.Client(token=os.environ["MAGIC_HOUR_API_KEY"])


class MagicHour:
    def __init__(self, client=None):
        self.c = client or _client()
        self._uploads: dict[str, str] = {}

    def upload(self, path: str | pathlib.Path) -> str:
        """Upload a local reference once per process. The SDK does not auto-upload every asset
        field (end frames, multi-image edits), so we always pass already-uploaded paths."""
        key = str(pathlib.Path(path).resolve())
        if key not in self._uploads:
            self._uploads[key] = self.c.v1.files.upload_file(key)
        return self._uploads[key]

    def submit(self, t: Target, prompt: str, refs: list[str] | None = None,
               aspect_ratio: str = "1:1", name: str | None = None) -> str:
        v1 = self.c.v1
        refs = [self.upload(r) for r in (refs or [])]
        if t.endpoint == "ai_image_generator":
            r = v1.ai_image_generator.create(
                image_count=1, model=t.model, resolution=t.resolution, aspect_ratio=aspect_ratio,
                style={"prompt": prompt}, name=name, **t.extra)
        elif t.endpoint == "ai_image_editor":
            assert refs, "ai_image_editor needs a reference image"
            r = v1.ai_image_editor.create(
                assets={"image_file_paths": refs}, style={"prompt": prompt}, model=t.model,
                resolution=t.resolution, aspect_ratio=aspect_ratio, image_count=1, name=name,
                **t.extra)
        elif t.endpoint == "image_to_video":
            assert refs, "image_to_video needs a start frame"
            r = v1.image_to_video.create(
                assets={"image_file_path": refs[0]}, end_seconds=t.seconds, model=t.model,
                resolution=t.resolution, style={"prompt": prompt}, name=name, **t.extra)
        else:
            raise ValueError(t.endpoint)
        return r.id

    def get(self, kind: str, project_id: str):
        v1 = self.c.v1
        return (v1.video_projects if kind == "video" else v1.image_projects).get(id=project_id)

    def wait(self, kind: str, project_id: str, timeout: float = 900, every: float = 4.0):
        t0 = time.time()
        while True:
            p = self.get(kind, project_id)
            if p.status in FINAL:
                return p
            if time.time() - t0 > timeout:
                raise TimeoutError(f"{project_id} still {p.status} after {timeout:.0f}s")
            time.sleep(every)

    @staticmethod
    def download(project, out_dir: pathlib.Path, stem: str) -> list[pathlib.Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for i, d in enumerate(project.downloads or []):
            ext = pathlib.Path(d.url.split("?")[0]).suffix or ".bin"
            p = out_dir / f"{stem}_{i}{ext}"
            with urllib.request.urlopen(d.url, timeout=120) as resp, open(p, "wb") as f:
                f.write(resp.read())
            paths.append(p)
        return paths
