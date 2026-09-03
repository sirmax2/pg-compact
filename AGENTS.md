# AGENTS.md

Guidance for AI agents working in this repository. Keep it accurate: if you
change behaviour that contradicts something here, update this file in the same
change.

## What this project is

`pg-compact` is a zero-lock **online** table and TOAST bloat compaction tool
for PostgreSQL. It shrinks a bloated relation in place by relocating tail
rows/chunks into free space nearer the front (via ordinary `UPDATE`s that the
free space map serves front-first) and letting `VACUUM` truncate the emptied
tail — no `AccessExclusiveLock`, no second on-disk copy, unlike
`VACUUM FULL` / `pg_repack` / `CLUSTER`.

Python package under `src/pg_compact/`, CLI entry point `pg-compact`
(`pg_compact.cli:main`). Supports PostgreSQL 12–17.

## Environment & commands

- Python 3.9+; dependencies via `psycopg[binary]`, optional `rich` for the
  interactive UI. Dev tools: `pytest`, `pytest-timeout`, `ruff`, `mypy`.
- Use the project virtualenv. On this machine it is `.venv/` (Windows:
  `.venv\Scripts\python.exe`). Run module code with `PYTHONPATH=src`.
- Lint:  `python -m ruff check src/ tests/`  (line-length 125; rules E,F,W,I,UP)
- Types: `python -m mypy src`   (config in `pyproject.toml`)
- Tests: `python -m pytest tests/ --pg-version=pg17 -q`
  - Tests run against **real** PostgreSQL containers, not mocks. Start them
    with `tests/docker-compose.yml` (services `pg12` on port 55412, `pg17` on
    55417, db `pgcompact_test`, user/pw `postgres`). They are NOT spun up
    automatically by pytest.
  - `--pg-version=pg12|pg17` restricts to one server; omit to run both.
  - There may also be a `pg14-test` container (port 55414) holding a restored
    real database (`calc`) used for ad-hoc manual investigation — not part of
    the automated suite.
- Always run ruff + mypy + tests before committing. Keep them green.

## Module map

- `cli.py` — argument parsing, interactive/CI detection, top-level flow.
- `compactor.py` — the compaction engine: the shared tail-to-head relocation
  loop (`_compact_relation`), heap and TOAST phase drivers, progress emission,
  skip decision, `CompactionConfig`/`CompactionTarget`/`ProgressUpdate`.
- `stats.py` — bloat measurement (`BloatStats`), the occupied/ideal-pages
  model, and the online-reclaimable estimate.
- `fsm_predict.py` — FSM category arithmetic (a transcription of
  `freespace.c`), the placement predicate, and the live fragmentation-map
  builder. This is where the "what will the FSM do" logic lives.
- `toast.py` — TOAST-specific rewrite SQL and the two strategies (PG17 native
  `pg_column_toast_chunk_id`, PG12–16 bulk `pageinspect` chunk map).
- `db.py` — psycopg helpers (connection, sizes, `pg_freespace`, VACUUM, etc.).
- `disk_guard.py` — free-disk monitoring / pause logic.
- `reindex.py` — `REINDEX CONCURRENTLY` at the end.
- `ui.py` — `rich` banner, live progress bar + fragmentation map, summary.
- `logging_utils.py` — log function type, byte/duration formatting.

## Domain invariants — read before changing compaction or reporting

These are hard-won and easy to get wrong. Several were bugs.

1. **Forcing a real re-TOAST.** `col = (col::text)::type` is a **no-op**:
   PostgreSQL recognises the identity cast and passes the original TOAST
   pointer through unchanged, so `heap_update` keeps it and nothing relocates.
   The working form builds a genuinely new datum by concatenation:
   `col = (left(col::text || 'X', -1) || right(col::text, 0))::type`
   (value-preserving, but a fresh string that must be re-TOASTed). See
   `toast.py::_toast_rewrite_assignments`. Do not "simplify" it back.

2. **FSM placement is front-first.** `fsm_search` returns the lowest-numbered
   page whose free space (rounded **down** to a 1/256-of-page = 32-byte
   category) is at least the request (rounded **up** to a category). So
   relocation can drain the tail even when free space is spread uniformly —
   the FSM pulls new chunks toward the front. A hole a little under one whole
   chunk/tuple (e.g. 1920–2016 B for a ~2 KB TOAST chunk) is one category
   short and is never offered; that sub-chunk remainder is the irreducible
   floor a full rewrite also keeps (~8% on one measured column).

3. **Phase-skip and progress target key on the ONLINE estimate**, not the full
   `VACUUM FULL` floor. Using the floor would start phases that make no
   progress and set unreachable targets. Progress is reported **per phase**
   (heap, then TOAST), each on its own relation's scale.

4. **Bloat breakdown is a reliable 3-part split that sums to 100%:**
   Live data / Unusable padding (internal name: alignment waste) / Bloat
   (the full-rewrite floor). The Bloat line is itemised into two **indented
   sub-lines that sum to Bloat** (not extra entries in the 100% split):
   "pg-compact frees now" (the online figure) and "needs VACUUM FULL"
   (= floor − online). The online part cannot be computed exactly up front,
   only bounded from one FSM scan, so it stays a sub-line of Bloat and never
   enters the top-level 100% sum. Both output paths (plain `compactor.py` log +
   rich `ui.py` banner) must use the same "Bloat / pg-compact frees now /
   needs VACUUM FULL" wording, and the same label must mean the same number
   everywhere (do not label the floor "Reclaimable" in one place and the online
   figure "Reclaimable" in another — that was a real bug).

5. **The online estimate is an upper bound.** `sum(avail)` over pages at/above
   the relocation unit's FSM threshold. Accurate for whole-chunk-sized holes;
   can overshoot when partial (sub-chunk) holes dominate or when free space
   already sits ahead of all live data. Present it as such.

6. **UI flicker.** The live display is a `rich.Live` driven via
   `get_renderable` (Live pulls on its own timer; do NOT also call
   `live.update()` per engine event — that double-renders and flickers). The
   fragmentation map is a single coloured line refreshed only after each
   `VACUUM`. The spinner is intentional (shows liveness during long quiet
   phases); keep `refresh_per_second` modest.

## Conventions

- Match the existing style; keep functions typed where the surrounding code
  is. mypy runs with `check_untyped_defs` but not `disallow_untyped_defs`.
- Prefer `psycopg.sql` composition for identifiers/literals; never format
  user/identifier values into SQL strings by hand.
- Verify claims about PostgreSQL behaviour against a real server (the
  containers above) rather than assuming — this codebase has repeatedly turned
  up behaviour that contradicts intuition (see invariants 1 and 2).
- Do not create scratch scripts in the repo root for investigation without
  cleaning them up; drop any temp tables you create in the shared containers.
- Git: commit only when asked; stage specific files; never push to
  main/master unless explicitly requested.
