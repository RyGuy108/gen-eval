-- A row copied from an earlier run whose item renders identically (same render_key):
-- scored again under the new eval set version, never re-rendered or re-charged.
ALTER TABLE generations ADD COLUMN IF NOT EXISTS reused_from bigint REFERENCES generations(id);
