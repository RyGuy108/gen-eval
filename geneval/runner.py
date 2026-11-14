"""Plan, render and score a run.

  plan    one `generations` row per (item, target, sample); idempotent
  render  RQ job per row: reserve credits against the run budget (atomic Lua in Redis), submit,
          persist the Magic Hour project id *before* polling, poll, download, settle the actual
          credits_charged against the reservation, enqueue scoring
  score   RQ job per row on a SimpleWorker so CLIP / OCR / face models load once per worker

A run is resumable: rows with a project id are re-polled rather than re-submitted, finished rows are
skipped, and the budget can never be overspent by concurrent workers.
"""
from __future__ import annotations

import json
import os
import pathlib
import time
import traceback

import redis

from . import db, evalset, media, quality, targets
from .providers.magichour import MagicHour, Target

ARTIFACTS = pathlib.Path(os.environ.get("GENEVAL_ARTIFACTS", pathlib.Path(__file__).parents[1] / "artifacts"))
REDIS_URL = os.environ.get("GENEVAL_REDIS", "redis://localhost:6379/0")
SCORER_VERSION = {"face": "yunet-sface-1", "ocr": "rapidocr-1.4.4-sellers-3", "clip": None}
N_FRAMES = 8

# KEYS[1]=spent KEYS[2]=reserved  ARGV[1]=budget ARGV[2]=amount -> 1 reserved / 0 over budget
RESERVE = """
local spent = tonumber(redis.call('GET', KEYS[1]) or '0')
local held = tonumber(redis.call('GET', KEYS[2]) or '0')
if spent + held + tonumber(ARGV[2]) > tonumber(ARGV[1]) then return 0 end
redis.call('INCRBY', KEYS[2], ARGV[2])
return 1
"""
# settle a reservation: release `held`, record `actual`
SETTLE = """
redis.call('DECRBY', KEYS[2], ARGV[1])
redis.call('INCRBY', KEYS[1], ARGV[2])
return redis.call('GET', KEYS[1])
"""


def _r():
    return redis.Redis.from_url(REDIS_URL)


class Budget:
    def __init__(self, run_id: int, budget: int, r=None):
        self.r = r or _r()
        self.budget = budget
        self.keys = [f"geneval:run:{run_id}:spent", f"geneval:run:{run_id}:held"]
        self._reserve = self.r.register_script(RESERVE)
        self._settle = self.r.register_script(SETTLE)

    def reserve(self, amount: int) -> bool:
        return bool(self._reserve(keys=self.keys, args=[self.budget, amount]))

    def settle(self, held: int, actual: int) -> int:
        return int(self._settle(keys=self.keys, args=[held, actual]))

    def spent(self) -> int:
        return int(self.r.get(self.keys[0]) or 0)


def estimate(model: str, t: Target) -> int:
    """Reservation size: max credits ever observed for this target, else the list price, else a
    conservative video default."""
    r = _r()
    seen = r.hget("geneval:observed_credits", t.key)
    if seen:
        return int(seen)
    lp = targets.ALL[model].list_credits if model in targets.ALL else 0
    return int(lp) if lp else 600


# ----------------------------------------------------------------------------------------- plan

def create_run(name: str, es: evalset.EvalSet, models: list[str] | None, budget: int,
               suites: list[str] | None = None, samples: int = 1,
               items: list[str] | None = None) -> int:
    with db.connect() as c:
        db.register_evalset(c, es)
        tlist = {}
        chosen = [it for it in es.items
                  if (not suites or it.suite in suites) and (not items or it.id in items)]
        for it in chosen:
            for t in targets.targets_for(it.endpoint, models):
                tlist[t.key] = t
        row = c.execute(
            "INSERT INTO runs (evalset_hash, name, targets, samples, budget) VALUES (%s,%s,%s,%s,%s) RETURNING id",
            (es.hash, name, json.dumps([t.__dict__ for t in tlist.values()]), samples, budget)).fetchone()
        run_id = row["id"]
        n = 0
        for it in chosen:
            for t in targets.targets_for(it.endpoint, models):
                for s in range(samples):
                    n += c.execute(
                        """INSERT INTO generations (run_id, item_id, item_hash, target_key, model, sample)
                           VALUES (%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING""",
                        (run_id, it.id, it.hash, t.key, targets.model_of(t.key), s)).rowcount
        c.commit()
    return run_id


def _target(run: dict, key: str) -> Target:
    for t in run["targets"]:
        if t["key"] == key:
            return Target(**t)
    raise KeyError(key)


# --------------------------------------------------------------------------------------- render

def render(gen_id: int, mh: MagicHour | None = None) -> str:
    with db.connect() as c:
        g = c.execute("SELECT g.*, r.targets, r.budget, e.version FROM generations g JOIN runs r ON r.id=g.run_id "
                      "JOIN evalsets e ON e.hash=r.evalset_hash WHERE g.id=%s FOR UPDATE OF g SKIP LOCKED",
                      (gen_id,)).fetchone()
        if g is None or g["status"] in ("complete", "skipped"):
            return "noop"
        es = evalset.load(g["version"])
        item = es.by_id()[g["item_id"]]
        if item.hash != g["item_hash"]:
            raise RuntimeError(f"item {item.id} changed since planning")
        t = _target(g, g["target_key"])
        mh = mh or MagicHour()
        budget = Budget(g["run_id"], g["budget"])
        held = 0
        pid = g["mh_project_id"]
        if not pid:
            held = estimate(g["model"], t)
            if not budget.reserve(held):
                c.execute("UPDATE generations SET status='skipped', error='budget' WHERE id=%s", (gen_id,))
                c.commit()
                return "skipped:budget"
            try:
                refs = [str(es.ref_path(item))] if item.ref else []
                pid = mh.submit(t, item.prompt, refs=refs, aspect_ratio=item.aspect_ratio,
                                name=f"geneval-{g['run_id']}-{item.id}-{t.model}")
            except Exception as e:  # rejected before rendering: nothing charged
                budget.settle(held, 0)
                c.execute("UPDATE generations SET status='error', error=%s WHERE id=%s", (str(e)[:500], gen_id))
                c.commit()
                return "error:submit"
            c.execute("UPDATE generations SET status='submitted', mh_project_id=%s, submitted_at=now() WHERE id=%s",
                      (pid, gen_id))
            c.commit()   # the paid-for project id is durable before we wait on it
    # poll outside the row lock / transaction
    p = mh.wait(t.kind, pid)
    credits = int(p.credits_charged or 0)
    budget = Budget(g["run_id"], g["budget"])
    budget.settle(held, credits)   # held is 0 when resuming an already-submitted row
    r = _r()
    if p.status == "complete" and credits:
        prev = int(r.hget("geneval:observed_credits", t.key) or 0)
        r.hset("geneval:observed_credits", t.key, max(prev, credits))
    with db.connect() as c:
        if p.status != "complete":
            c.execute("UPDATE generations SET status='error', credits=%s, error=%s, finished_at=now() WHERE id=%s",
                      (credits, json.dumps(p.error.model_dump() if p.error else p.status)[:500], gen_id))
            c.commit()
            return "error:render"
        out = ARTIFACTS / f"run{g['run_id']}" / item.id
        paths = MagicHour.download(p, out, f"{t.model}_s{g['sample']}")
        c.execute("UPDATE generations SET status='complete', credits=%s, paths=%s, finished_at=now() WHERE id=%s",
                  (credits, [str(x) for x in paths], gen_id))
        c.commit()
    return "complete"


# ---------------------------------------------------------------------------------------- score

_SCORERS: dict = {}


def scorers() -> dict:
    if not _SCORERS:
        from .scorers.clip import ClipScorer
        from .scorers.face import FaceScorer
        from .scorers.ocr import OcrScorer
        clip = ClipScorer()
        SCORER_VERSION["clip"] = f"{clip.arch}-onnx-2"
        _SCORERS.update(clip=clip, face=FaceScorer(), ocr=OcrScorer())
    return _SCORERS


def score_item(item: evalset.Item, path: str, ref: pathlib.Path | None) -> dict:
    S = scorers()
    frames = media.load(path, N_FRAMES)
    out = {}
    if "face" in item.scorers:
        out["face"] = S["face"].score(frames, ref)
    if "ocr" in item.scorers:
        out["ocr"] = S["ocr"].score(frames, list(item.text))
    if "clip" in item.scorers:
        adh = frames[-max(1, len(frames) // 3):] if item.adherence_frames == "last_third" else frames
        out["clip"] = S["clip"].score(adh, item.scene or item.prompt, list(item.foils) or None)
    q, ok = quality.judge(item.suite, out)
    out["quality"] = {"q": q, "pass": ok, "version": quality.QUALITY_VERSION}
    return out


def score(gen_id: int, force: bool = False) -> str:
    with db.connect() as c:
        g = c.execute("SELECT g.*, e.version FROM generations g JOIN runs r ON r.id=g.run_id "
                      "JOIN evalsets e ON e.hash=r.evalset_hash WHERE g.id=%s", (gen_id,)).fetchone()
        if g["status"] != "complete" or not g["paths"]:
            return "not-ready"
        es = evalset.load(g["version"])
        item = es.by_id()[g["item_id"]]
        S = scorers()
        if not force:
            have = {r["scorer"] for r in c.execute(
                "SELECT scorer FROM scores WHERE generation_id=%s AND scorer_version = ANY(%s)",
                (gen_id, list(SCORER_VERSION.values()) + [quality.QUALITY_VERSION]))}
            if {*item.scorers, "quality"} <= have:
                return "cached"
        out = score_item(item, g["paths"][0], es.ref_path(item))
        for name, metrics in out.items():
            ver = quality.QUALITY_VERSION if name == "quality" else SCORER_VERSION[name]
            c.execute("""INSERT INTO scores (generation_id, scorer, scorer_version, metrics) VALUES (%s,%s,%s,%s)
                         ON CONFLICT (generation_id, scorer, scorer_version) DO UPDATE SET metrics=EXCLUDED.metrics,
                         scored_at=now()""", (gen_id, name, ver, json.dumps(metrics)))
        c.commit()
    return "scored"


# ---------------------------------------------------------------------------------------- jobs

def render_then_score(gen_id: int) -> str:
    try:
        st = render(gen_id)
    except Exception:
        traceback.print_exc()
        raise
    if st == "complete":
        from rq import Queue
        Queue("score", connection=_r()).enqueue(score, gen_id, job_timeout=900)
    return st


def enqueue_run(run_id: int, per_model_spacing: float = 0.0) -> int:
    from rq import Queue
    q = Queue("render", connection=_r())
    with db.connect() as c:
        ids = [r["id"] for r in c.execute(
            "SELECT id FROM generations WHERE run_id=%s AND status IN ('pending','submitted') ORDER BY id", (run_id,))]
    for i in ids:
        q.enqueue(render_then_score, i, job_timeout=1800, result_ttl=3600)
        if per_model_spacing:
            time.sleep(per_model_spacing)
    return len(ids)


def run_local(run_id: int, workers: int = 6) -> dict:
    """Same pipeline without RQ: render in a thread pool (I/O bound), score in-process."""
    from collections import Counter
    from concurrent.futures import ThreadPoolExecutor
    with db.connect() as c:
        ids = [r["id"] for r in c.execute(
            "SELECT id FROM generations WHERE run_id=%s AND status IN ('pending','submitted') ORDER BY id", (run_id,))]
    mh = MagicHour()
    with ThreadPoolExecutor(workers) as ex:
        res = Counter(ex.map(lambda i: _safe_render(i, mh), ids))
    with db.connect() as c:
        done = [r["id"] for r in c.execute(
            "SELECT id FROM generations WHERE run_id=%s AND status='complete' ORDER BY id", (run_id,))]
    for i in done:
        res["score:" + score(i)] += 1
    return dict(res)


def _safe_render(i: int, mh: MagicHour) -> str:
    try:
        return render(i, mh)
    except Exception as e:
        traceback.print_exc()
        return f"exception:{type(e).__name__}"
