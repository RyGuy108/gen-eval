-- gen-eval schema

CREATE TABLE IF NOT EXISTS evalsets (
    hash        text PRIMARY KEY,
    version     text NOT NULL,
    manifest    jsonb NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS runs (
    id            bigserial PRIMARY KEY,
    evalset_hash  text NOT NULL REFERENCES evalsets(hash),
    name          text NOT NULL,
    targets       jsonb NOT NULL,           -- [{key, endpoint, model, resolution, seconds}]
    samples       int NOT NULL DEFAULT 1,   -- renders per (item, target)
    budget        int NOT NULL,             -- hard credit cap for the run
    status        text NOT NULL DEFAULT 'open',
    created_at    timestamptz NOT NULL DEFAULT now()
);

-- One row per render. The unique key makes a run resumable: re-enqueueing only creates
-- what is missing, and a row that already has an mh_project_id is polled, never re-submitted.
CREATE TABLE IF NOT EXISTS generations (
    id             bigserial PRIMARY KEY,
    run_id         bigint NOT NULL REFERENCES runs(id),
    item_id        text NOT NULL,
    item_hash      text NOT NULL,
    target_key     text NOT NULL,
    model          text NOT NULL,
    sample         int NOT NULL DEFAULT 0,
    status         text NOT NULL DEFAULT 'pending',  -- pending|submitted|complete|error|skipped
    mh_project_id  text,
    credits        int,
    paths          text[] NOT NULL DEFAULT '{}',
    error          text,
    submitted_at   timestamptz,
    finished_at    timestamptz,
    UNIQUE (run_id, item_id, target_key, sample)
);
CREATE INDEX IF NOT EXISTS generations_status ON generations (run_id, status);

CREATE TABLE IF NOT EXISTS scores (
    generation_id  bigint NOT NULL REFERENCES generations(id) ON DELETE CASCADE,
    scorer         text NOT NULL,          -- face | ocr | clip
    scorer_version text NOT NULL,
    metrics        jsonb NOT NULL,
    scored_at      timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (generation_id, scorer, scorer_version)
);

-- Human judgements used to validate the scorers (see geneval/calibrate.py).
CREATE TABLE IF NOT EXISTS labels (
    generation_id  bigint NOT NULL REFERENCES generations(id) ON DELETE CASCADE,
    labeler        text NOT NULL,
    acceptable     boolean NOT NULL,
    note           text,
    labeled_at     timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (generation_id, labeler)
);
