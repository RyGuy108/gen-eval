"""HTTP API: start runs, read leaderboards, serve outputs, and collect hand labels.

  POST /runs                         plan a run and enqueue its renders on RQ
  GET  /runs                         runs with progress and credits spent
  GET  /runs/{ids}/leaderboard       ids comma-separated; newest run wins per (item, model)
  GET  /runs/{ids}/calibration       scorer vs hand-label agreement
  GET  /generations/{id}/media       the rendered file
  GET  /label?runs=1,2               labeling page (keyboard: y / n / s)
  POST /labels                       {generation_id, labeler, acceptable, note}
"""
from __future__ import annotations

import json
import math
import pathlib

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from . import calibrate, db, evalset, rank, runner

app = FastAPI(title="gen-eval")


def _clean(o):
    if isinstance(o, float) and (math.isinf(o) or math.isnan(o)):
        return None
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    return o


def _ids(s: str) -> list[int]:
    try:
        return [int(x) for x in s.split(",") if x]
    except ValueError:
        raise HTTPException(400, "run ids must be integers")


class RunIn(BaseModel):
    name: str
    evalset: str = "v1"
    budget: int = Field(gt=0, le=50_000)
    models: list[str] | None = None
    suites: list[str] | None = None
    items: list[str] | None = None
    samples: int = Field(1, ge=1, le=8)


@app.post("/runs")
def create_run(body: RunIn):
    es = evalset.load(body.evalset)
    rid = runner.create_run(body.name, es, body.models, body.budget, body.suites, body.samples, body.items)
    return {"run_id": rid, "enqueued": runner.enqueue_run(rid)}


@app.get("/runs")
def runs():
    with db.connect() as c:
        return c.execute("""SELECT r.id, r.name, r.budget, r.created_at, e.version AS evalset,
                              count(g.*) AS renders, count(g.*) FILTER (WHERE g.status='complete') AS complete,
                              coalesce(sum(g.credits),0) AS credits
                            FROM runs r JOIN evalsets e ON e.hash=r.evalset_hash
                            LEFT JOIN generations g ON g.run_id=r.id GROUP BY r.id, e.version ORDER BY r.id""").fetchall()


@app.get("/runs/{ids}/leaderboard")
def leaderboard(ids: str):
    return JSONResponse(_clean(rank.leaderboard(_ids(ids))))


@app.get("/runs/{ids}/calibration")
def calibration(ids: str, labeler: str | None = None):
    return JSONResponse(_clean(calibrate.calibrate(_ids(ids), labeler)))


@app.get("/generations/{gid}/media")
def media(gid: int):
    with db.connect() as c:
        g = c.execute("SELECT paths FROM generations WHERE id=%s", (gid,)).fetchone()
    if not g or not g["paths"]:
        raise HTTPException(404)
    p = pathlib.Path(g["paths"][0]).resolve()
    if not p.is_relative_to(runner.ARTIFACTS.resolve()):
        raise HTTPException(403)
    return FileResponse(p)


class LabelIn(BaseModel):
    generation_id: int
    labeler: str = Field(min_length=1, max_length=64)
    acceptable: bool
    note: str | None = None


@app.post("/labels")
def add_label(body: LabelIn):
    with db.connect() as c:
        c.execute("""INSERT INTO labels (generation_id, labeler, acceptable, note) VALUES (%s,%s,%s,%s)
                     ON CONFLICT (generation_id, labeler) DO UPDATE SET acceptable=EXCLUDED.acceptable,
                     note=EXCLUDED.note, labeled_at=now()""",
                  (body.generation_id, body.labeler, body.acceptable, body.note))
        c.commit()
    return {"ok": True}


@app.get("/label/queue")
def label_queue(runs: str, labeler: str):
    """Outputs not yet labeled by this labeler, shuffled so model order can't bias the labels.
    Scores are deliberately not returned."""
    rows = rank.fetch(_ids(runs))
    with db.connect() as c:
        done = {r["generation_id"] for r in c.execute("SELECT generation_id FROM labels WHERE labeler=%s", (labeler,))}
    items = {}
    for r in rows:
        if r["evalset"] not in items:
            items[r["evalset"]] = evalset.load(r["evalset"], verify=False).by_id()
    import random
    q = [{"id": r["id"], "suite": r["suite"], "item": r["item_id"], "prompt": items[r["evalset"]][r["item_id"]].prompt,
          "text": list(items[r["evalset"]][r["item_id"]].text),
          "ref": items[r["evalset"]][r["item_id"]].ref, "video": r["suite"] == "video"}
         for r in rows if r["id"] not in done]
    random.Random(labeler).shuffle(q)
    return q


@app.get("/refs/{name}")
def ref(name: str):
    p = (evalset.ROOT / "v1" / "refs" / pathlib.Path(name).name)
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p)


LABEL_HTML = (pathlib.Path(__file__).parent / "label.html")


@app.get("/label", response_class=HTMLResponse)
def label_page():
    return LABEL_HTML.read_text()


@app.get("/healthz")
def healthz():
    return {"ok": True}


if __name__ == "__main__":
    print(json.dumps(app.openapi()["paths"].keys().__repr__()))
