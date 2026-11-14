"""Do the automatic scores agree with people? Validate every scorer against hand labels.

Labels are one bit per output ("would I ship this without regenerating?") from the `labels` table.
For each suite we report:

  agreement      accuracy / precision / recall of the scorer's pass bit vs the label
  auc            ROC AUC of the continuous quality score vs the label (threshold-free)
  P@1, P@3       best-of-N selection: per item, rank that item's outputs (one per model) by quality
                 and check whether the top 1 / top 3 are acceptable. Compared with the base rate of a
                 random pick, this is the lift a customer gets from "generate on several models, let
                 the harness pick".
  P@5 pooled     rank every output in the suite by quality; fraction of the top 5 that are acceptable
                 (the retrieval-style Precision@5: is the harness's "best of the suite" really good?)
  model rank     Spearman correlation between models ordered by scorer pass rate and by label pass
                 rate, i.e. would the human and automatic leaderboards agree
  threshold      the pass threshold on quality that maximizes agreement, to show if the default bar
                 is miscalibrated
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

from . import db, rank


def labels(labeler: str | None = None) -> dict[int, bool]:
    with db.connect() as c:
        q = "SELECT generation_id, acceptable FROM labels" + (" WHERE labeler=%s" if labeler else "")
        return {r["generation_id"]: r["acceptable"] for r in c.execute(q, (labeler,) if labeler else ())}


def auc(scores: list[float], y: list[bool]) -> float:
    pos = [s for s, t in zip(scores, y) if t]
    neg = [s for s, t in zip(scores, y) if not t]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def precision_at_k_per_item(rows: list[dict], k: int) -> float:
    by = defaultdict(list)
    for r in rows:
        by[r["item_id"]].append(r)
    vals = []
    for rs in by.values():
        top = sorted(rs, key=lambda r: -r["q"])[:k]
        vals.append(np.mean([r["label"] for r in top]))
    return float(np.mean(vals))


def calibrate(run_ids: list[int], labeler: str | None = None) -> dict:
    lab = labels(labeler)
    rows = [r | {"label": lab[r["id"]]} for r in rank.fetch(run_ids) if r["id"] in lab]
    out = {}
    for suite in sorted({r["suite"] for r in rows}):
        rs = [r for r in rows if r["suite"] == suite]
        y = np.array([r["label"] for r in rs])
        p = np.array([r["pass"] for r in rs])
        q = np.array([r["q"] for r in rs])
        tp = int((p & y).sum())
        res = {
            "n": len(rs), "base_rate": float(y.mean()),
            "accuracy": float((p == y).mean()),
            "precision": tp / max(1, int(p.sum())), "recall": tp / max(1, int(y.sum())),
            "auc": auc(q.tolist(), y.tolist()),
            "p_at_1": precision_at_k_per_item(rs, 1), "p_at_3": precision_at_k_per_item(rs, 3),
            "p_at_5_pooled": float(np.mean([r["label"] for r in sorted(rs, key=lambda r: -r["q"])[:5]])),
        }
        # best single threshold on q
        cands = sorted(set(np.round(q, 4)))
        best = max(cands, key=lambda t: ((q >= t) == y).mean()) if cands else 0
        res["best_threshold"] = float(best)
        res["best_threshold_accuracy"] = float(((q >= best) == y).mean())
        # model-level agreement
        by_m = defaultdict(lambda: [[], []])
        for r in rs:
            by_m[r["model"]][0].append(r["pass"])
            by_m[r["model"]][1].append(r["label"])
        if len(by_m) >= 3:
            a = [np.mean(v[0]) for v in by_m.values()]
            b = [np.mean(v[1]) for v in by_m.values()]
            res["model_rank_spearman"] = float(spearmanr(a, b).statistic)
        res["models"] = {m: {"scorer_pass": float(np.mean(v[0])), "label_pass": float(np.mean(v[1]))}
                         for m, v in sorted(by_m.items())}
        out[suite] = res
    return out
