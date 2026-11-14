"""gen-eval command line.

  python -m geneval.cli migrate
  python -m geneval.cli freeze v1
  python -m geneval.cli plan --name nightly --budget 8000 [--models a,b] [--suites text,adherence]
  python -m geneval.cli run RUN_ID [--local] [--workers 6]
  python -m geneval.cli score RUN_ID [--force]
  python -m geneval.cli leaderboard RUN_ID [--json out.json]
  python -m geneval.cli export-router RUN_ID out.jsonl
"""
from __future__ import annotations

import argparse
import json


def _csv(s):
    return [x for x in s.split(",") if x] if s else None


def main(argv=None):
    ap = argparse.ArgumentParser(prog="geneval")
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("migrate")
    f = sp.add_parser("freeze"); f.add_argument("version")
    p = sp.add_parser("plan")
    p.add_argument("--name", required=True); p.add_argument("--evalset", default="v1")
    p.add_argument("--budget", type=int, required=True); p.add_argument("--models")
    p.add_argument("--suites"); p.add_argument("--items"); p.add_argument("--samples", type=int, default=1)
    p.add_argument("--reuse", help="comma-separated earlier run ids whose identical renders are reused")
    r = sp.add_parser("run"); r.add_argument("run_id", type=int)
    r.add_argument("--local", action="store_true"); r.add_argument("--workers", type=int, default=6)
    s = sp.add_parser("score"); s.add_argument("run_id", type=int); s.add_argument("--force", action="store_true")
    lb = sp.add_parser("leaderboard"); lb.add_argument("run_ids", type=int, nargs="+"); lb.add_argument("--json")
    il = sp.add_parser("import-labels"); il.add_argument("path"); il.add_argument("--labeler", required=True)
    il.add_argument("--runs", required=True)
    cb = sp.add_parser("calibrate"); cb.add_argument("run_ids", type=int, nargs="+"); cb.add_argument("--labeler")
    cb.add_argument("--json")
    ex = sp.add_parser("export-router"); ex.add_argument("run_ids", type=int, nargs="+"); ex.add_argument("--out", required=True)
    ex.add_argument("--all-models", action="store_true", help="also models outside the router catalog")
    a = ap.parse_args(argv)

    from . import db, evalset, runner
    if a.cmd == "migrate":
        db.migrate(); print("ok")
    elif a.cmd == "freeze":
        m = evalset.freeze(a.version); print(f"{a.version} {m['hash'][:12]} ({m['n_items']} items)")
    elif a.cmd == "plan":
        es = evalset.load(a.evalset)
        rid = runner.create_run(a.name, es, _csv(a.models), a.budget, _csv(a.suites), a.samples, _csv(a.items),
                                [int(x) for x in _csv(a.reuse) or []])
        print(rid)
    elif a.cmd == "run":
        print(runner.run_local(a.run_id, a.workers) if a.local else f"enqueued {runner.enqueue_run(a.run_id)}")
    elif a.cmd == "score":
        with db.connect() as c:
            ids = [r["id"] for r in c.execute(
                "SELECT id FROM generations WHERE run_id=%s AND status='complete' ORDER BY id", (a.run_id,))]
        from collections import Counter
        print(dict(Counter(runner.score(i, force=a.force) for i in ids)))
    elif a.cmd == "leaderboard":
        from . import rank
        board = rank.leaderboard(a.run_ids)
        print(rank.format_table(board))
        if a.json:
            open(a.json, "w").write(json.dumps(board, indent=2))
    elif a.cmd == "import-labels":
        from . import rank
        gens = {(r["item_id"], r["model"]): r["id"] for r in rank.fetch(_csv(a.runs) and [int(x) for x in _csv(a.runs)])}
        n = 0
        with db.connect() as c:
            for line in open(a.path):
                lab = json.loads(line)
                gid = gens.get((lab["item_id"], lab["model"]))
                if gid is None:
                    print("no generation for", lab["item_id"], lab["model"]); continue
                c.execute("""INSERT INTO labels (generation_id, labeler, acceptable, note) VALUES (%s,%s,%s,%s)
                             ON CONFLICT (generation_id, labeler) DO UPDATE SET acceptable=EXCLUDED.acceptable,
                             note=EXCLUDED.note""", (gid, a.labeler, lab["acceptable"], lab.get("note")))
                n += 1
            c.commit()
        print(f"imported {n} labels")
    elif a.cmd == "calibrate":
        from . import calibrate
        res = calibrate.calibrate(a.run_ids, a.labeler)
        for suite, r in res.items():
            print(f"{suite:<10} n={r['n']:<3} base={r['base_rate']:.2f} acc={r['accuracy']:.2f} "
                  f"prec={r['precision']:.2f} rec={r['recall']:.2f} auc={r['auc']:.2f} "
                  f"P@1={r['p_at_1']:.2f} P@3={r['p_at_3']:.2f} P@5pool={r['p_at_5_pooled']:.2f} "
                  f"rho={r.get('model_rank_spearman', float('nan')):.2f} best_t={r['best_threshold']:.3f}"
                  f"({r['best_threshold_accuracy']:.2f})")
        if a.json:
            open(a.json, "w").write(json.dumps(res, indent=2, default=float))
    elif a.cmd == "export-router":
        from . import export
        n = export.router_rows(a.run_ids, a.out, None if a.all_models else export.ROUTER_CATALOG)
        print(f"wrote {n} rows -> {a.out}")


if __name__ == "__main__":
    main()
