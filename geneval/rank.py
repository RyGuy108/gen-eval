"""Leaderboard: quality per credit, per suite, with uncertainty.

The headline metric is **credits per acceptable output** = mean credits charged / pass rate: what a
customer expects to spend to get one render they can use, including the retries. It ranks a cheap
model that usually fails below a pricier one that usually works, which "quality / credits" alone
does not. We also report mean quality, pass rate with a bootstrap CI over items, quality per 100
credits, and whether the model is on the cost/quality Pareto frontier for that suite.

Bootstrap resamples *items* (not renders), because item difficulty is the dominant source of
variance: the CI answers "would this ranking hold on a different prompt set like this one".
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from . import db

B = 2000


def fetch(run_ids: list[int], quality_version: str | None = None) -> list[dict]:
    """One row per (item, target, sample), newest run wins, with its quality score."""
    from .quality import QUALITY_VERSION
    qv = quality_version or QUALITY_VERSION
    with db.connect() as c:
        rows = c.execute(
            """SELECT DISTINCT ON (g.item_id, g.target_key, g.sample)
                      g.id, g.run_id, g.item_id, g.target_key, g.model, g.sample, g.credits, g.paths,
                      e.manifest->>'version' AS evalset, q.metrics AS quality,
                      (SELECT jsonb_object_agg(s.scorer, s.metrics) FROM scores s
                        WHERE s.generation_id = g.id AND s.scorer <> 'quality') AS metrics
               FROM generations g
               JOIN runs r ON r.id = g.run_id
               JOIN evalsets e ON e.hash = r.evalset_hash
               JOIN scores q ON q.generation_id = g.id AND q.scorer = 'quality' AND q.scorer_version = %s
               WHERE g.run_id = ANY(%s) AND g.status = 'complete'
               ORDER BY g.item_id, g.target_key, g.sample, g.run_id DESC, q.scored_at DESC""",
            (qv, run_ids)).fetchall()
    from . import evalset
    suites = {}
    for r in rows:
        if r["evalset"] not in suites:
            suites[r["evalset"]] = {i.id: i.suite for i in evalset.load(r["evalset"], verify=False).items}
        r["suite"] = suites[r["evalset"]][r["item_id"]]
        r["q"] = float(r["quality"]["q"])
        r["pass"] = bool(r["quality"]["pass"])
    return rows


def _boot(items: dict[str, list[float]], rng: np.random.Generator) -> tuple[float, float]:
    keys = list(items)
    if len(keys) < 2:
        return (float("nan"), float("nan"))
    per = np.array([np.mean(items[k]) for k in keys])
    idx = rng.integers(0, len(keys), size=(B, len(keys)))
    means = per[idx].mean(1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def pareto(points: dict[str, tuple[float, float]]) -> set[str]:
    """Models not dominated in (lower credits, higher quality)."""
    out = set()
    for m, (c, q) in points.items():
        if not any((c2 <= c and q2 >= q) and (c2 < c or q2 > q) for m2, (c2, q2) in points.items() if m2 != m):
            out.add(m)
    return out


def summarize(rows: list[dict], seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    by = defaultdict(list)
    for r in rows:
        by[(r["suite"], r["model"])].append(r)
    suites: dict[str, list[dict]] = defaultdict(list)
    for (suite, model), rs in sorted(by.items()):
        per_item_pass, per_item_q = defaultdict(list), defaultdict(list)
        for r in rs:
            per_item_pass[r["item_id"]].append(float(r["pass"]))
            per_item_q[r["item_id"]].append(r["q"])
        credits = float(np.mean([r["credits"] for r in rs]))
        pr = float(np.mean([r["pass"] for r in rs]))
        mq = float(np.mean([r["q"] for r in rs]))
        lo, hi = _boot(per_item_pass, rng)
        suites[suite].append({
            "model": model, "n": len(rs), "credits": credits, "pass_rate": pr, "pass_ci": [lo, hi],
            "quality": mq, "quality_ci": list(_boot(per_item_q, rng)),
            "credits_per_pass": credits / pr if pr > 0 else math.inf,
            "quality_per_100c": 100 * mq / credits if credits else math.inf,
        })
    for suite, ms in suites.items():
        front = pareto({m["model"]: (m["credits"], m["quality"]) for m in ms})
        for m in ms:
            m["pareto"] = m["model"] in front
        ms.sort(key=lambda m: (m["credits_per_pass"], -m["quality"]))
        for i, m in enumerate(ms, 1):
            m["rank"] = i
    return dict(suites)


def head_to_head(rows: list[dict], suite: str) -> dict:
    """Per pair of models: on items both rendered, how often A's quality beats B's."""
    q = defaultdict(dict)
    for r in rows:
        if r["suite"] == suite:
            q[r["model"]].setdefault(r["item_id"], []).append(r["q"])
    models = sorted(q)
    out = {}
    for a in models:
        for b in models:
            if a >= b:
                continue
            shared = set(q[a]) & set(q[b])
            if not shared:
                continue
            wins = sum(np.mean(q[a][i]) > np.mean(q[b][i]) + 1e-6 for i in shared)
            ties = sum(abs(np.mean(q[a][i]) - np.mean(q[b][i])) <= 1e-6 for i in shared)
            out[f"{a} vs {b}"] = {"n": len(shared), "a_wins": int(wins), "ties": int(ties)}
    return out


def leaderboard(run_ids: list[int]) -> dict:
    rows = fetch(run_ids)
    return {"runs": run_ids, "n_outputs": len(rows),
            "credits_spent": int(sum(r["credits"] or 0 for r in rows)),
            "suites": summarize(rows)}


def format_table(board: dict) -> str:
    lines = [f"{board['n_outputs']} scored outputs, {board['credits_spent']} credits"]
    for suite, ms in board["suites"].items():
        lines.append(f"\n== {suite}")
        lines.append(f"{'#':>2} {'model':<20} {'n':>3} {'credits':>7} {'pass':>5} {'95% CI':>13} "
                     f"{'quality':>7} {'cr/pass':>8} {'q/100cr':>8}  pareto")
        for m in ms:
            cpp = "inf" if math.isinf(m["credits_per_pass"]) else f"{m['credits_per_pass']:.0f}"
            ci = "" if math.isnan(m["pass_ci"][0]) else f"[{m['pass_ci'][0]:.2f},{m['pass_ci'][1]:.2f}]"
            lines.append(f"{m['rank']:>2} {m['model']:<20} {m['n']:>3} {m['credits']:>7.0f} {m['pass_rate']:>5.2f} "
                         f"{ci:>13} {m['quality']:>7.3f} {cpp:>8} {m['quality_per_100c']:>8.2f}  "
                         f"{'*' if m['pareto'] else ''}")
    return "\n".join(lines)
