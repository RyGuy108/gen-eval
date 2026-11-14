"""Export scored generations as training data for the model router (RyGuy108/model-router).

The router trains on rows shaped like its simulator output (sim/world.py):

  {"id", "req": {task, prompt, resolution, aspect_ratio, seconds, has_image_input, quality},
   "outcomes": {model: [{"q": float, "draft": 0|1, "standard": 0|1, "premium": 0|1}, ...]}}

Here every number is measured on a real render instead of drawn from a hidden skill profile, and
each row carries `source` so real and simulated rows can be mixed or held out separately. One row
per (eval item, quality bar requested) would duplicate outcomes, so we emit one row per item with
`req.quality = "standard"`; the router reads all three bars from `outcomes` anyway.
"""
from __future__ import annotations

import json
from collections import defaultdict

from . import evalset, quality, rank


# Models in the router's catalog (router/catalog.py). Its trainer indexes outcomes by catalog name,
# so rows are filtered to these unless the caller passes its own list.
ROUTER_CATALOG = {"z-image-turbo", "flux-schnell", "qwen-image-2.1", "krea-2", "seedream-v4", "gpt-image-2",
                  "seedream-v5-pro", "nano-banana-pro", "kling-2.6", "ltx-2.5", "kling-3.0",
                  "seedance-2.0-mini", "veo3.1-lite", "seedance-2.0"}


def router_rows(run_ids: list[int], out_path: str, models: set[str] | None = ROUTER_CATALOG) -> int:
    rows = [r for r in rank.fetch(run_ids) if models is None or r["model"] in models]
    by_item = defaultdict(list)
    for r in rows:
        by_item[(r["evalset"], r["item_id"])].append(r)
    n = 0
    with open(out_path, "w") as f:
        for (ver, item_id), rs in sorted(by_item.items()):
            it = evalset.load(ver, verify=False).by_id()[item_id]
            res = rs[0]["target_key"].split("@", 1)[1].split("/", 1)[0]
            outcomes = defaultdict(list)
            for r in rs:
                metrics = {**r["metrics"]}
                outcomes[r["model"]].append({"q": round(r["q"], 4), **quality.bars(r["suite"], metrics),
                                             "credits": r["credits"]})
            if not outcomes:
                continue
            f.write(json.dumps({
                "id": f"{ver}:{item_id}", "source": f"gen-eval {ver}", "suite": it.suite,
                "req": {"task": it.task, "prompt": it.prompt, "resolution": res,
                        "aspect_ratio": it.aspect_ratio, "seconds": it.seconds,
                        "has_image_input": bool(it.ref), "quality": "standard"},
                "outcomes": outcomes}) + "\n")
            n += 1
    return n
