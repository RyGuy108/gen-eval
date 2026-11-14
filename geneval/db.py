from __future__ import annotations

import json
import os
import pathlib
from contextlib import contextmanager

import psycopg
from psycopg.rows import dict_row

DSN = os.environ.get("GENEVAL_DSN", "postgresql://geneval:geneval@localhost:5432/geneval")
MIGRATIONS = pathlib.Path(__file__).parents[1] / "migrations"


@contextmanager
def connect(dsn: str = DSN):
    with psycopg.connect(dsn, row_factory=dict_row, autocommit=False) as conn:
        yield conn


def migrate(dsn: str = DSN) -> None:
    with connect(dsn) as c:
        for f in sorted(MIGRATIONS.glob("*.sql")):
            c.execute(f.read_text())
        c.commit()


def register_evalset(c, es) -> None:
    manifest = json.loads((es.dir / "MANIFEST.json").read_text())
    c.execute("INSERT INTO evalsets (hash, version, manifest) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
              (es.hash, es.version, json.dumps(manifest)))
