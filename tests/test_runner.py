"""Runner integration test against real Postgres + Redis with a fake Magic Hour client."""
import os
import pathlib
import types
import uuid

import pytest

psycopg = pytest.importorskip("psycopg")
redis = pytest.importorskip("redis")

from geneval import db, evalset, runner  # noqa: E402

DSN = os.environ.get("GENEVAL_TEST_DSN", "postgresql://geneval:geneval@localhost:5432/geneval_test")
REDIS = os.environ.get("GENEVAL_TEST_REDIS", "redis://localhost:6379/15")   # never the live db


def _up():
    try:
        psycopg.connect(DSN, connect_timeout=2).close()
        redis.Redis.from_url(REDIS).ping()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _up(), reason="needs Postgres and Redis")


class FakeMH:
    """Charges `cost` per render, writes a tiny PNG, counts submissions."""

    def __init__(self, tmp: pathlib.Path, cost: int = 50):
        self.tmp, self.cost, self.submitted = tmp, cost, []
        from PIL import Image
        self.png = tmp / "out.png"
        Image.new("RGB", (8, 8), "red").save(self.png)

    def submit(self, t, prompt, refs=None, aspect_ratio="1:1", name=None):
        pid = uuid.uuid4().hex
        self.submitted.append((t.key, prompt))
        return pid

    def wait(self, kind, pid, timeout=900, every=4.0):
        return types.SimpleNamespace(status="complete", credits_charged=self.cost, error=None,
                                     downloads=[types.SimpleNamespace(url=self.png.as_uri())])

    download = staticmethod(runner.MagicHour.download)


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DSN", DSN)
    monkeypatch.setattr(runner, "ARTIFACTS", tmp_path / "artifacts")
    monkeypatch.setattr(runner, "REDIS_URL", REDIS)
    redis.Redis.from_url(REDIS).flushdb()
    with db.connect(DSN) as c:
        c.execute("DROP TABLE IF EXISTS labels, scores, generations, runs, evalsets CASCADE")
        c.commit()
    db.migrate(DSN)
    # connect()'s default dsn was bound at import time; route every call to the test DB
    monkeypatch.setattr(db, "connect", lambda dsn=DSN, _c=db.connect: _c(DSN))
    yield tmp_path


def test_budget_caps_spend_and_rows_resume(env):
    es = evalset.load("v1")
    rid = runner.create_run("t", es, ["z-image-turbo"], budget=120, suites=["text"])
    with db.connect() as c:
        ids = [r["id"] for r in c.execute("SELECT id FROM generations WHERE run_id=%s ORDER BY id", (rid,))]
    assert len(ids) == len([i for i in es.items if i.suite == "text"])
    mh = FakeMH(env, cost=50)
    res = [runner.render(i, mh) for i in ids]
    # The first render reserves the 5-credit list price but is charged 50. From then on the
    # reservation uses the observed price, so the third render (100 spent + 50) is refused.
    assert res.count("complete") == 2
    assert res.count("skipped:budget") == len(ids) - 2
    assert runner.Budget(rid, 120).spent() == 100

    # re-running is a no-op for finished rows and never re-submits
    before = len(mh.submitted)
    assert all(runner.render(i, mh) == "noop" for i in ids)
    assert len(mh.submitted) == before


def test_submitted_row_is_polled_not_resubmitted(env):
    es = evalset.load("v1")
    rid = runner.create_run("t", es, ["z-image-turbo"], budget=1000, items=["t01"])
    with db.connect() as c:
        gid = c.execute("SELECT id FROM generations WHERE run_id=%s", (rid,)).fetchone()["id"]
        c.execute("UPDATE generations SET status='submitted', mh_project_id='already-paid' WHERE id=%s", (gid,))
        c.commit()
    mh = FakeMH(env, cost=5)
    assert runner.render(gid, mh) == "complete"
    assert mh.submitted == []
    with db.connect() as c:
        g = c.execute("SELECT * FROM generations WHERE id=%s", (gid,)).fetchone()
    assert g["credits"] == 5 and pathlib.Path(g["paths"][0]).exists()
