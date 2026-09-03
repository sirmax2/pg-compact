# pg-compact

Reduce PostgreSQL table and TOAST bloat online, without heavy locks and
without needing extra disk space, by relocating rows out of the tail of a
table and letting `VACUUM` truncate the now-empty pages.

`pg-compact` is a from-scratch Python rewrite of an earlier shell-script
prototype. It follows the same general approach as
[`pgcompacttable`](https://github.com/dataegret/pgcompacttable) and
[`pgtoolkit`'s `pgcompact`](https://github.com/grayhemp/pgtoolkit), but
with its own implementation, its own CLI, and its own test suite.

## How it works

1. `VACUUM` alone can only truncate pages from the *end* of a relation's
   file if they're completely empty - it never moves rows around to make
   that happen.
2. `pg-compact` walks the relation from the last page backwards in
   windows, relocating the rows in each tail window into free space
   nearer the front of the file. PostgreSQL's free space map (FSM) serves
   new tuple/chunk allocations from the front, so a relocated row's new
   version lands on an earlier page while its old version becomes dead on
   the tail.
3. Because those relocations leave the tail full of *dead* tuples (not
   empty pages), a `VACUUM` is what finally lets PostgreSQL physically
   truncate the trailing pages and return the space to the filesystem.
   `pg-compact` defers `VACUUM` and runs it periodically - relocating
   many windows first, then truncating the whole accumulated dead tail at
   once (see "Deferred-VACUUM loop" below).
4. Both the heap and, if present, the TOAST relation are compacted by the
   same loop with strategy-specific callbacks - see "TOAST-aware
   compaction" below for why TOAST needs its own row-selection strategy.
5. Once the relation is compacted, indexes are rebuilt with
   `REINDEX TABLE CONCURRENTLY` (PostgreSQL 12+) so index bloat doesn't
   erase the space savings.

No `pg_repack`-style table rewrite, no `VACUUM FULL`, no doubling of disk
usage, and no `AccessExclusiveLock` at any point during the main run
(`REINDEX ... CONCURRENTLY` briefly needs `SHARE UPDATE EXCLUSIVE`, which
still allows reads and writes).

### Deferred-VACUUM loop

Relocating a tail row leaves its old tuple/chunk dead on the tail while
the new version lands in an earlier free page. As long as earlier free
space can absorb the relocated rows, the relation's physical size does
**not** grow between vacuums - so `pg-compact` rewrites many trailing
windows back-to-back without vacuuming, then runs a single `VACUUM` to
truncate the whole accumulated dead tail at once.

A cursor walks down from the physical tail toward the relation's ideal
(bloat-free) page count. Each window's size is bounded by the *landing
capacity* - the free pages ahead of the tail that can absorb the
relocated rows - so new versions never land back inside the window being
emptied. `VACUUM` fires once a configurable number of pages have been
cleared, plus a final pass at the end. The landing capacity is measured
once per `VACUUM` (via `pg_freespace()`) and estimated between vacuums,
avoiding a costly full FSM scan on every iteration.

### Algorithm in detail

Block size below is PostgreSQL's page size (8 KB by default).

**Top level (`compact_table`)**

1. **Guards.** Skip the table if it does not exist, if it has
   `ENABLE ALWAYS`/`ENABLE REPLICA` `UPDATE` triggers (to avoid firing
   arbitrary trigger logic), or if another `pg-compact` already holds the
   per-table advisory lock.
2. **Initial `VACUUM`** (unless `--no-initial-vacuum`) so the FSM reflects
   the current free space before anything is measured.
3. **Measure bloat** for the heap and, separately, the TOAST relation,
   with one formula for both:
   - `free_bytes` = `sum(avail)` from `pg_freespace()`.
   - `occupied = size − free_bytes`; `ideal_pages = ceil(occupied /
     usable_per_page)` is the floor a full rewrite could reach.
   - `reclaimable = (relpages − ideal_pages) × block_size` — the genuine
     bloat; the free space that still would not pack away is
     `alignment_waste` (structural per-page padding, not reclaimable).
4. **Skip decision.** Unless `--force`, stop early if *both* the heap and
   the TOAST reclaimable percentages are below `--min-compact-percent`
   (default 10%), or if the table is smaller than `--min-compact-pages`
   (default 10). `--dry-run` reports the estimate here and stops.
5. **Heap phase** (only if the heap is worth it): run the shared loop
   below over the heap relation, with `session_replication_role = replica`
   and `lock_timeout` set. Skipped when the heap's reclaimable percent is
   below threshold (e.g. when its free space is entirely alignment waste).
6. **TOAST phase** (unless `--no-toast-compact`, and only if the table has
   TOASTable columns and the TOAST relation is non-empty and bloated): run
   the shared loop over the TOAST relation.
7. **Finalize:** `VACUUM`, then `REINDEX TABLE CONCURRENTLY` (unless
   `--no-reindex`), then `ANALYZE`. A statement-timeout here is
   non-fatal - the compaction already succeeded. Release the advisory
   lock.

**Shared compaction loop (`_compact_relation`)**

The loop is the same for the heap and TOAST; only three callbacks differ
(how to find the rows in a window, how to rewrite them, and how to
`VACUUM`). Let `ideal` be the relation's ideal page count and `cursor`
start at the current physical page count.

1. **Prime the FSM.** Run one `VACUUM` up front so a never-vacuumed or
   freshly bulk-deleted relation reports its reusable space, and measure
   the initial landing capacity.
2. **Loop while `cursor > ideal`:**
   1. **Check landing capacity.** Use the cached estimate; if it dips
      below the trial window size, re-measure exactly with `pg_freespace()`.
      If there is no free space ahead of the cursor and dead pages have
      accumulated, `VACUUM` to reclaim them and re-measure; if there is
      still none, stop (the remaining bloat is interleaved with live data
      - see limitations).
   2. **Size the window.** `window = min(2000, cursor - ideal, 20000,
      landing_estimate)` pages, then `win_start = cursor - window`.
   3. **Collect** the ctids of rows whose data occupies
      `[win_start, cursor)` (strategy-specific - see below).
   4. **Rewrite** those rows in one batched `UPDATE`, relocating them into
      free space ahead. Throttling is adaptive: if this rewrite ran
      markedly slower per row than the recent average (lock contention, an
      I/O spike), pause for a fraction of its duration (capped by
      `--throttle-delay-s`) to give the server a breather. A normal
      multi-second window never trips it; `--no-throttle` disables it.
   5. **Advance.** Move `cursor` down to `win_start` (its rows are now
      dead-on-tail, pending truncation) and decrement the landing estimate
      by the window span.
   6. **Disk check** (throttled to ~30 s): measure free space; pause until
      it recovers if it is below `--min-free-disk-mb`.
   7. **Periodic `VACUUM`.** Once `50000` pages have been cleared since the
      last `VACUUM` (clamped down to the relation's reclaimable size, so
      small relations get a single `VACUUM` at the end instead), run it -
      truncating the accumulated dead tail - and refresh the landing
      estimate. A single `VACUUM` may truncate nothing even while
      relocation is working, because PostgreSQL only truncates once at
      least ~1000 empty pages have accumulated contiguously at the file's
      end; so progress is judged by whether the physical size trends down
      across several vacuums, not by any one truncation. If several
      consecutive vacuums produce no net reduction, the tail can no longer
      be relocated (internal fragmentation) and the loop stops.
3. **Final `VACUUM`** to truncate any remaining dead tail.

**Row selection per relation**

- **Heap.** Drain a tail window inside a *single* transaction:
  `UPDATE ... SET col = col WHERE ctid >= win_start` repeatedly until the
  rows leave the window, committing only when the window is drained (or
  rolling back if stuck). The single transaction is what forces the page
  to fill and PostgreSQL to place the next row version on an earlier page,
  defeating both HOT updates and opportunistic pruning (see the HOT-update
  subtlety above).
- **TOAST.** Rewrite the TOASTed column(s) with a concatenation expression
  (`col = (left(col::text || 'X', -1) || right(col::text, 0))::type`) so the
  value is genuinely re-TOASTed into new chunks that the FSM places at the
  front. A plain `col = (col::text)::type` does **not** work — PostgreSQL
  recognises the identity cast and passes the original TOAST pointer through
  unchanged, so nothing moves (see the note under "Known limitations").
  Finding which heap rows have chunks in the window differs by version: PG17+
  evaluates `pg_column_toast_chunk_id()` inline; PG12-16 builds a one-time
  `chunk_id -> heap ctid` map with a single `pageinspect` heap scan up front,
  then looks each window up in it.

### Alignment-aware bloat reporting

The FSM's "free space" is not all reclaimable. Some of it is structural
per-page padding that survives even a full rewrite — for the heap, the
bytes left at the end of a page too small for another row; for TOAST, the
gap left when a page's chunks don't divide the page evenly. `pg-compact`
separates that from the genuinely reclaimable bloat and reports a three-part
breakdown that sums to 100% of the relation size, for the heap and its TOAST
relation:

```
Heap (17.6 GB, 2312338 pages, via FSM):
  Live data                  16.5 GB   93.5%
  Unusable padding           49.6 MB    0.3%  never reclaimable
  Bloat                       1.1 GB    6.3%
    pg-compact frees now    704.6 MB    3.9%
    needs VACUUM FULL       431.7 MB    2.4%
TOAST (40.1 GB, via FSM):
  Live data                  36.2 GB   90.3%
  Unusable padding          109.0 MB    0.3%  never reclaimable
  Bloat                       3.8 GB    9.5%
    pg-compact frees now     86.8 KB    0.0%
    needs VACUUM FULL         3.8 GB    9.5%
```

Both are derived cheaply from one identity: the *occupied* bytes are
`size − free`, and packing that volume as tightly as a full rewrite would
(`ceil(occupied / usable_per_page)` pages) gives the floor `VACUUM FULL` /
`pg_repack` could reach. The pages above that floor are the **bloat**;
the free space that still would not pack away is **alignment waste**
(displayed as *"Unusable padding"*, never reclaimable by anything); the floor
itself is the live data plus unavoidable per-page overhead.

The breakdown splits the relation size into three parts that sum to 100%:

- **Live data** — rows/values in use plus unavoidable per-page overhead.
- **Unusable padding** — structural per-page alignment; no tool reclaims it.
- **Bloat** — the genuine bloat a full rewrite (`VACUUM FULL` / `pg_repack`)
  would return.

`pg-compact`, working in place, frees a *subset* of the **Bloat** now, so the
Bloat line is itemised into two indented sub-lines that sum to it:

- **pg-compact frees now** — what it can relocate into FSM-served holes and
  then truncate off the tail, in place and without a heavy lock. This is a
  cheaply computed **upper bound**, not an exact figure.
- **needs VACUUM FULL** — the remainder, free space only a full rewrite
  (`VACUUM FULL` / `pg_repack`) can pack away.

The "frees now" bound comes from one FSM scan: the free space on pages whose
available bytes reach the relocation unit's rounded-up FSM category (a full
~2 KB chunk for TOAST; the mean live-tuple footprint for the heap). Because the
FSM pulls relocated chunks to the earliest fitting page, `pg-compact` reaches
close to this bound when the free space is in whole-chunk-sized holes — even
if they're spread uniformly through the file. Where free space is dominated by
partial (sub-chunk) holes, or already sits ahead of all live data, the online
result can be much lower. So the *"frees ~X now"* figure is an **upper bound**;
the actual result is always shown as before → after when the run completes. A
phase is skipped when its *"can free now"* estimate is below the threshold.

### The HOT-update subtlety (heap phase)

For the **heap**, a no-op `UPDATE table SET col = col` on a column that is
part of an index would normally trigger a HOT (Heap-Only Tuple) update,
keeping the new row version on the *same* page and defeating relocation.
PostgreSQL's opportunistic page pruning can also reclaim a dead tuple's
space as soon as the producing transaction commits, so per-statement
commits would let rows shuffle in place forever without ever moving to an
earlier page. The heap phase therefore drains a tail window inside a
single transaction - repeatedly updating until the rows leave the window,
committing only once the window is drained (or rolling back if it gets
stuck) - which forces the page to fill and PostgreSQL to place the next
row version on an earlier page.

TOAST does not have this problem: rewritten chunks always go through the
TOAST relation's own FSM, which serves them from the front of the file.

### TOAST-aware compaction

This is the main reason `pg-compact` exists rather than just using a
generic no-op-`UPDATE` script: PostgreSQL's `UPDATE` has a well
documented shortcut - if a `TOAST`ed column's value isn't changing, its
on-disk bytes are left completely untouched, and only the TOAST pointer
is copied into the new heap tuple. A plain no-op `UPDATE` on some other,
small column shrinks the *heap* but does nothing at all for `TOAST`
storage, and even a literal `SET big_col = big_col` on the TOASTed
column itself doesn't help - PostgreSQL detects the value is
byte-for-byte identical and skips rewriting it.

`pg-compact` forces a real (if momentary) change to a TOASTed column's
bytes using a universal text round-trip that reproduces the exact
original value for any TOASTable type:

```sql
col = (left(col::text || 'X', -1) || right(col::text, 0))::original_type
```

The TOAST rewrite walks the TOAST relation from tail to head in windows,
rewriting the heap rows whose TOAST chunks occupy each trailing window so
their new chunks land in earlier free space. It uses the same
deferred-VACUUM loop as the heap: many windows are rewritten before a
`VACUUM` truncates the accumulated dead tail.

Two strategies are used to identify which heap rows to rewrite:

- **PG17+**: the built-in `pg_column_toast_chunk_id()` function extracts
  the chunk_id from a TOAST pointer inline — fast per-row evaluation, so
  the rows for a window are found with a single query against the TOAST
  relation.
- **PG12-16**: `pg_column_toast_chunk_id()` doesn't exist, so a one-time
  bulk heap scan via `pageinspect` reads every heap page, using C-level
  byte-pattern matching to extract TOAST pointers, and builds a temporary
  `chunk_id -> heap ctid` map. Each window then looks up its rows in that
  map. Building the map costs one full heap scan up front (~80 s for a
  2.3 M-page heap), amortised across the whole run.

This phase only runs when the table's TOAST relation actually has
reclaimable bloat (checked independently of the main heap) and is
skipped outright, even with `--force`, when the TOAST relation is empty.
Use `--no-toast-compact` to disable this phase entirely.

## Bloat estimation

`pg-compact` estimates reclaimable space from the relation size and
`pg_freespacemap` alone, with no dependency on `pgstattuple` and no
per-column statistics:

- `free_bytes` = `sum(avail)` from `pg_freespace()` (~100 ms plus one FSM
  scan that grows with page count).
- `occupied` = `size − free_bytes` — data plus unavoidable per-page
  overhead.
- `ideal_pages` = `ceil(occupied / usable_per_page)` — the floor a full
  rewrite could reach; `reclaimable = (relpages − ideal_pages) × 8192`,
  and the free space that still would not pack away is alignment waste.

The same formula is used for both the heap and the TOAST relation.

## Requirements

- PostgreSQL 12 or newer (for `REINDEX ... CONCURRENTLY` on tables).
- Python 3.9+.
- The following extensions must be installed before running `pg-compact`.
  It checks on startup and exits with the exact `CREATE EXTENSION`
  commands if any are missing:
  - **`pg_freespacemap`** — free space analysis and bloat estimation.
    Always required.
  - **`pageinspect`** — targeted TOAST compaction on PG12-16 (bulk heap
    scan to map chunk_ids to heap rows). Required on PG12-16 only; not
    needed on PG17+ where `pg_column_toast_chunk_id()` is a built-in.
- A user with enough privileges to `VACUUM`, `UPDATE`, and
  `REINDEX ... CONCURRENTLY` the target table (table owner or superuser),
  and to `CREATE EXTENSION` if the required extensions are not yet
  installed.
- Optional: the [`rich`](https://github.com/Textualize/rich) package for
  a live progress bar and summary table in an interactive terminal.

## Installation

```bash
pipx install pg-compact
# or
pip install pg-compact

# with the optional rich-based interactive UI:
pipx install "pg-compact[ui]"
```

## Usage

Connection options follow the same conventions as `psql`: `-h/-p/-U/-d`,
with `$PGHOST`/`$PGPORT`/`$PGUSER`/`$PGPASSWORD`/`$PGDATABASE` and
`~/.pgpass` as fallbacks. Everything is optional except the target table.

```bash
# Compact a table, using $PGHOST/$PGPORT/etc. from the environment
pg-compact -t public.orders

# Explicit connection options, like psql
pg-compact -h db.internal -p 5432 -U app_user -d billing -t public.orders

# See what would happen without changing anything
pg-compact -t public.orders --dry-run --verbose

# Compact regardless of the default bloat/size thresholds
pg-compact -t public.orders --force
```

Exit code is `0` on success (including "nothing to do"), non-zero if the
run failed or a table could not be fully compacted.

### Options

Connection options:

| Flag | Description |
|---|---|
| `-h, --host HOST` | Database server host or socket directory. Falls back to `$PGHOST`, then the libpq default. |
| `-p, --port PORT` | Database server port. Falls back to `$PGPORT`, then 5432. |
| `-U, --user USER` | Database user name. Falls back to `$PGUSER`, then the OS user. |
| `-W, --password PASSWORD` | Database password. Prefer `$PGPASSWORD` or `~/.pgpass` instead. |
| `-d, --dbname DBNAME` | Database name to connect to. Falls back to `$PGDATABASE`. |

Target:

| Flag | Description |
|---|---|
| `-t, --table [SCHEMA.]TABLE` | Table to compact (required). Defaults to the `public` schema if unqualified. |

Behavior (all optional, sane defaults):

| Flag | Default | Description |
|---|---|---|
| `-f, --force` | off | Compact even if below the minimum bloat/size thresholds. |
| `-n, --dry-run` | off | Report the bloat estimate and planned action without modifying the table. |
| `--no-reindex` | off | Skip `REINDEX TABLE CONCURRENTLY` after compaction. |
| `--no-initial-vacuum` | off | Skip the `VACUUM` run before measuring the table. |
| `--min-compact-percent PERCENT` | `10.0` | Minimum estimated reclaimable percent required to proceed. |
| `--min-compact-pages PAGES` | `10` | Minimum table size in pages required to proceed. |
| `--no-throttle` | off | Never throttle. By default throttling is adaptive (pauses only when a rewrite runs markedly slower per row than the recent average). |
| `--throttle-delay-s SEC` | `1.5` | Maximum pause after a slow rewrite; the actual pause scales with how long the rewrite took, up to this cap. |

Output:

| Flag | Description |
|---|---|
| `-v, --verbose` | Show detailed progress messages. |
| `-q, --quiet` | Show only errors and the final result. |
| `--interactive` | Show a startup banner, live progress bar, and summary table. Default: auto-detect a terminal (requires `rich`). |
| `--no-interactive` | Plain line-oriented output only, no banner/progress bar/summary table. |
| `-V, --version` | Print the version and exit. |
| `--help` | Show the full option reference and exit. |

Safety:

| Flag | Default | Description |
|---|---|---|
| `--min-free-disk-mb MB` | `10240` | Pause if free disk space drops below this. |
| `--no-disk-check` | off | Skip the disk space guard entirely. |
| `--lock-timeout-ms MS` | `5000` | Max time to wait for a row lock before backing off and retrying. |
| `--no-toast-compact` | off | Skip rewriting TOASTed columns; only compact the main heap. |

### Interactive vs. CI output

By default, `pg-compact` detects whether it's running in a real terminal
and, if the optional `rich` package is installed, shows a startup banner,
a live progress bar, and a summary table at the end. Piped output,
redirected files, and `cron`/`systemd` all fall back automatically to
plain timestamped log output. `--interactive`/`--no-interactive` override
the automatic detection either way.

### Live fragmentation map

Above the progress bar, the interactive display shows a
disk-defragmenter-style map of the relation currently being compacted
(the heap, then its TOAST storage), laid out front (left) to tail
(right). Each block is a range of pages, coloured by its dominant state
and refreshed after every `VACUUM` (one cheap FSM scan):

| Colour | Glyph | Meaning |
|---|---|---|
| green | `█` | live data - pages packed with rows/chunks in use |
| cyan | `░` | reclaimable now - free space in holes big enough for `pg-compact` to relocate into and then truncate |
| yellow | `▓` | needs `VACUUM FULL` - free space trapped in sub-tuple/sub-chunk holes the free space map never offers a relocation, so only a full rewrite can pack it away |
| dim | `·` | (near-)empty pages at the tail - about to be truncated |

As compaction proceeds you'll see the tail blocks drain (turn empty, then
disappear as `VACUUM` truncates them) while the file shrinks. A relation
that is mostly green with yellow mixed throughout is telling you its
remaining bloat is fragmentation only a full rewrite can reclaim.

### Disk space guard

The no-op `UPDATE`s generate real WAL. As it works, `pg-compact`
periodically checks free disk space and pauses rather than continuing to
write more WAL if space is low. The measured free space is shown in the
progress output. See `src/pg_compact/disk_guard.py` for the
multi-strategy approach (data directory `df`, WAL monitoring, graceful
fallback).

### Lock contention throttling

`--lock-timeout-ms` bounds how long a no-op UPDATE waits for a row lock,
so `pg-compact` backs off and retries after a short pause instead of
queuing indefinitely. Deadlocks are handled the same way.

### Ctrl+C during REINDEX

If interrupted during `REINDEX TABLE CONCURRENTLY`, PostgreSQL can leave
a partially built index marked invalid. `pg-compact` warns about this and
cleans it up automatically on the next run.

## Known limitations

- **Heap windows that will not drain.** The heap phase relocates rows by
  draining a tail window inside a single transaction, which defeats the
  HOT-update/pruning trap in the common case (see "The HOT-update
  subtlety" above). If a window still cannot be emptied within a bounded
  number of attempts — for example when there is no free space earlier in
  the file to move its rows into — that window is rolled back and left
  unchanged, and compaction of the heap stops there. `VACUUM FULL` or
  `pg_repack` can reclaim what remains.

- **How the FSM placement makes online compaction work (and its floor).**
  Compaction relocates a row or TOAST value by `UPDATE`ing it: PostgreSQL
  re-TOASTs the value into fresh chunks and the free space map (FSM) decides
  which page they land on. The FSM's `fsm_search` walks its tree preferring
  low block numbers, so it serves the **earliest** page whose free space is
  at least the rounded-up request size. Repeatedly rewriting the values that
  currently sit nearest the tail therefore pulls their chunks toward the
  front of the file and drains the tail, which `VACUUM` then truncates — this
  works **even when the free space is spread uniformly through the file**, not
  only when it happens to sit at the tail. (Verified: a uniformly fragmented
  40k-value TOAST relation shrank 156 MB → 83 MB online, versus 78 MB for
  `VACUUM FULL`.)

  The one thing an `UPDATE` cannot do is fill a hole **smaller** than one
  whole chunk: the FSM only offers a page for an allocation when its free
  space, rounded down to a 1/256-of-a-page category (32 bytes with the default
  8 KB block), is at least the request rounded **up** to a category. So a hole
  a little under one chunk (e.g. 1920–2016 bytes for the ~2 KB TOAST chunk) is
  one category short and is never offered. That sub-chunk remainder — the
  partial tail chunk every value leaves — is the true irreducible floor:
  measured at ~8% on one real column, and a full sequential rewrite
  (`VACUUM FULL` / `pg_repack`) leaves the *same* remainder, because it too
  cuts values into `TOAST_MAX_CHUNK_SIZE` chunks. So `pg-compact` reaches
  within a few percent of what a full rewrite achieves, online, without the
  `AccessExclusiveLock` or the second on-disk copy.

  > Note: forcing a real re-TOAST matters. `SET col = (col::text)::type` is a
  > no-op — PostgreSQL recognises the identity cast and passes the original
  > TOAST pointer through unchanged, so nothing moves. `pg-compact` builds a
  > genuinely new datum via concatenation
  > (`left(col::text || 'X', -1) || right(col::text, 0)`) so the value is
  > actually rewritten and relocated.

  `pg-compact` detects when it can no longer shrink the relation and stops
  rather than churning. (Smaller TOAST chunks would leave less such waste,
  but `TOAST_MAX_CHUNK_SIZE` is fixed at compile time.)

- Reclaiming space is rate-limited by how fast rows can be relocated and
  by `VACUUM`; large tables take proportionally longer than `pg_repack`
  or `VACUUM FULL`, in exchange for not blocking and not needing extra
  disk space.

- A long-running transaction on another connection (holding an old MVCC
  snapshot) limits how much space `VACUUM` can actually reclaim.

- Tables with `ENABLE ALWAYS` or `ENABLE REPLICA` triggers on `UPDATE`
  are skipped to avoid firing arbitrary trigger logic.

## Running the tests

```bash
docker compose -f tests/docker-compose.yml up -d
pip install -e ".[dev]"
pytest tests/
```

Runs against PostgreSQL 12 and 17. Use `pytest tests/ --pg-version=pg17`
to run against only one.

## License

MIT, see [LICENSE](LICENSE).
