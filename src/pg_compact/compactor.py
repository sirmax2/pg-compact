"""The core compaction engine.

Both heap and TOAST compaction follow the same deferred-VACUUM loop:

1. Compute the ideal relation size from the FSM
   (``relpages - free_bytes / block_size``).
2. Walk a cursor down from the physical tail toward the ideal size. Each
   iteration collects the ctids of rows on a tail window and rewrites
   them with a batched UPDATE, so their new versions land in earlier free
   space (the FSM serves allocations from the front) while the old
   versions become dead on the tail.
3. The window is sized from the landing capacity (free pages ahead of the
   tail) so relocated rows never fall back inside the window being
   emptied.
4. VACUUM is deferred: because the relation does not grow while free space
   absorbs the relocated rows, many windows are rewritten before a single
   VACUUM truncates the whole accumulated dead tail at once (every
   ``_VACUUM_EVERY_PAGES`` cleared pages, plus a final pass).
5. Stop when the cursor reaches the ideal size or there is no free space
   left ahead of the tail (bloat fully interleaved with live data).

``session_replication_role = replica`` is set for the duration so that
ordinary (non-ALWAYS) triggers do not fire for these synthetic updates.
After both phases, indexes are rebuilt with ``REINDEX TABLE CONCURRENTLY``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

import psycopg
from psycopg import sql

from pg_compact import db
from pg_compact.fsm_predict import FULL_TOAST_CHUNK_FOOTPRINT
from pg_compact.logging_utils import LogFn, fmt_bytes, fmt_duration, noop_log
from pg_compact.stats import BloatStats, get_bloat_stats, get_toast_bloat_stats, get_toast_relation

MIN_COMPACT_PAGES = 10
MIN_COMPACT_PERCENT = 10.0

# Tail window (pages) processed per iteration.  Sized from the landing
# capacity (free pages ahead of the tail) so relocated rows' new tuples
# land on earlier free pages rather than back inside the window being
# emptied.  _MAX caps the window on very large relations; _INITIAL is
# only used to probe landing capacity before the real window is computed.
_MAX_TAIL_WINDOW_PAGES = 20000
_INITIAL_TAIL_WINDOW_PAGES = 2000

# Run VACUUM to truncate the dead tail once this many pages have been
# cleared (rows relocated) since the last VACUUM.  Larger = fewer VACUUMs
# (faster) but more transient dead space; must stay well under the
# available free space so the relation never grows.
_VACUUM_EVERY_PAGES = 50000

# Plateau detection.  After relocating N pages of tail rows, a productive
# VACUUM truncates about N pages (measured ratio ~1.0); once relocation
# runs out of usable free space ahead, the emptied pages no longer land at
# the tail and VACUUM truncates almost nothing (ratio ~0) even though we
# cleared a full batch.  So we stop when a VACUUM that followed a
# substantial batch of clearing reclaims less than this fraction of it.
# The gap between the two regimes is wide, so one such VACUUM is decisive.
_PLATEAU_TRUNCATE_RATIO = 0.2

# PG12-16 bulk TOAST path: rebuild the static chunk map when this fraction
# of the chunk_ids found on a window's pages are missing from it.  Rewrites
# re-TOAST values into new chunk_ids not in the map; once enough of a
# window's chunks are unmapped, collect_ctids would silently skip those
# rows, so the map must be refreshed to stay complete.
_CHUNK_MAP_STALE_RATIO = 0.8

# Fraction of a rewrite's duration to sleep afterwards when throttling, so
# the server gets idle time in proportion to the load just imposed (a
# duty cycle).  0.5 means at most ~1/3 of wall-clock is spent pausing.
# The pause is additionally capped by --throttle-delay-s.
_THROTTLE_RATIO = 0.5

# Adaptive throttle: a rewrite is "abnormally slow" (and triggers
# throttling) when it takes more than this multiple of the recent average
# per-row cost.  This self-calibrates to the relation's own throughput, so
# a normal multi-second window never trips it, but a rewrite that suddenly
# runs several times slower per row (lock contention, an I/O spike) does.
_THROTTLE_SLOW_FACTOR = 3.0
# EWMA smoothing for the per-row baseline (weight of each new sample).
_THROTTLE_EWMA_ALPHA = 0.3
# Warm-up: don't throttle until we have at least this many samples, so the
# baseline is meaningful.
_THROTTLE_MIN_SAMPLES = 3
# Never throttle a rewrite quicker than this regardless of the per-row
# math: a fast statement imposes little sustained load, and small tail
# windows have a high per-row cost (fixed overhead over few rows) that
# would otherwise look "abnormally slow".
_THROTTLE_MIN_ELAPSED_S = 5.0

# Status icons surfaced through ProgressUpdate.status_icon so the
# interactive UI can label what the loop is currently doing.  The UI
# matches on these exact code points (see ui.py).
_ICON_PAUSE = "\u23f8"  # throttle or waiting for disk
_ICON_VACUUM = "\U0001f9f9"  # running VACUUM

# Phase label the UI checks to render "waiting disk" rather than
# "throttle" for the pause icon.
_PHASE_WAITING_DISK = "waiting for disk"


class Outcome(str, Enum):
    COMPLETED = "completed"
    SKIPPED_LOCKED = "skipped_locked"
    SKIPPED_EMPTY = "skipped_empty"
    SKIPPED_TRIGGERS = "skipped_triggers"
    SKIPPED_BELOW_THRESHOLD = "skipped_below_threshold"
    INCOMPLETE_STUCK = "incomplete_stuck"


@dataclass
class CompactionConfig:
    force: bool = False
    dry_run: bool = False
    min_compact_pages: int = MIN_COMPACT_PAGES
    min_compact_percent: float = MIN_COMPACT_PERCENT
    throttle: bool = True  # adaptive throttle on abnormally slow rewrites
    throttle_delay_s: float = 1.5
    initial_vacuum: bool = True
    reindex: bool = True
    toast_compact: bool = True
    min_free_disk_mb: int = 10240
    disk_check: bool = True
    lock_timeout_ms: int = 5000


@dataclass
class CompactionResult:
    outcome: Outcome
    schema: str
    table: str
    size_before: db.SizeStats | None = None
    size_after: db.SizeStats | None = None
    bloat_before: BloatStats | None = None
    bloat_after: BloatStats | None = None
    toast_bloat_before: BloatStats | None = None
    reindexed: bool = False
    toast_columns_rewritten: list = field(default_factory=list)
    message: str = ""


@dataclass(frozen=True)
class ProgressUpdate:
    """A structured progress snapshot, decoupled from any particular UI toolkit."""

    phase: str
    pages_done: int
    pages_total: int
    current_size_bytes: int | None = None
    before_size_bytes: int | None = None
    target_size_bytes: int | None = None
    free_disk_bytes: int | None = None
    min_free_disk_bytes: int | None = None
    status_icon: str = ""
    # Progress rate and ETA, computed from real page progress (the same
    # figures the flat status log shows).  Rendered by the UI instead of
    # letting rich extrapolate from the bar's fill, which disagrees.
    mb_per_s: float | None = None
    eta_seconds: float | None = None
    # Live fragmentation map: a list of single-char cell codes (front->tail)
    # from fsm_predict.build_fragmentation_map, plus a short label.  Refreshed
    # periodically (after each VACUUM) since it costs one FSM scan; None means
    # "unchanged, keep showing the previous map".
    frag_map: list[str] | None = None
    frag_map_label: str | None = None


ProgressFn = Callable[[ProgressUpdate], None]


def _noop_progress(update: ProgressUpdate) -> None:  # pragma: no cover
    pass


# ---------------------------------------------------------------------------
# CompactionTarget — pluggable callbacks for the shared loop
# ---------------------------------------------------------------------------


@dataclass
class CompactionTarget:
    """Everything the shared compaction loop needs."""

    # The relation being shrunk (for pg_relation_size / FSM queries).
    relation: str

    # Target page count — stop when page_count <= ideal_pages.
    ideal_pages: int

    # Collect ctids of rows that live on tail pages [from_page .. to_page].
    collect_ctids: Callable[[psycopg.Connection, int, int], list[str]]

    # Rewrite the collected rows.  Returns number of rows touched.
    rewrite: Callable[[psycopg.Connection, list[str]], int]

    # Vacuum the relation after a batch of rewrites.
    vacuum: Callable[[psycopg.Connection], None]

    # Optional one-time setup / teardown (e.g. build chunk map).
    setup: Callable[[psycopg.Connection], None] | None = None
    teardown: Callable[[psycopg.Connection], None] | None = None

    # Label for log messages.
    phase_name: str = "compacting"

    # On-page footprint of the relocation unit, used to classify the live
    # fragmentation map's cells (reclaimable-now vs sub-unit holes).  A full
    # TOAST chunk for TOAST; the mean live-tuple footprint for the heap.
    map_unit_footprint: int = 4096


# ---------------------------------------------------------------------------
# Shared compaction loop
# ---------------------------------------------------------------------------


def _compact_relation(
    conn: psycopg.Connection,
    target: CompactionTarget,
    config: CompactionConfig,
    log: LogFn,
    progress: ProgressFn = _noop_progress,
    before_total_bytes: int = 0,
    target_bytes: int = 0,
) -> Outcome:
    """Batched tail-to-head compaction with deferred VACUUM.

    Relocating a tail row leaves its old chunks/tuples dead on the tail
    while the new version lands in an earlier free page (FSM serves from
    the front).  As long as earlier free space can absorb them, the
    relation does *not* grow — so we can rewrite many trailing windows
    back-to-back without vacuuming, then run a single VACUUM to truncate
    the whole accumulated dead tail at once.

    VACUUM is triggered when either:
      * enough tail pages have been cleared (``_VACUUM_EVERY_PAGES``), or
      * the landing capacity (free space ahead of the tail) runs low.
    """
    block_size = 8192
    pages_before = db.get_relation_page_count(conn, target.relation)
    pages_current = pages_before
    rows_rewritten = 0
    reclaimable_pages = max(pages_before - target.ideal_pages, 0)
    start_time = time.monotonic()
    last_status_log = start_time
    status_log_interval = 30.0

    # Scale the VACUUM cadence to the relation.  The module constant is
    # tuned for very large tables; on small relations it would exceed the
    # whole reclaimable range, so a single final VACUUM does the job.
    vacuum_every = max(1, min(_VACUUM_EVERY_PAGES, reclaimable_pages))

    # Cursor walks down from the physical tail toward ideal_pages.  Pages
    # in (cursor, pages_current] have had their rows relocated but still
    # hold dead tuples until the next VACUUM.
    cursor = pages_current
    cleared_since_vacuum = 0

    # Cached landing capacity (free pages available ahead of the tail).
    # Measuring it exactly via pg_freespace() means materializing the whole
    # FSM fork — ~4.5 s on a 6 M-page relation — so calling it every
    # iteration would dominate the runtime.  Instead we measure it after
    # each VACUUM (which is what changes the FSM), then decrement the
    # estimate as windows consume free space, and only re-measure when the
    # estimate runs low.  Set by the priming VACUUM below.
    landing_estimate = 0

    # Throttle the disk-space check to the same cadence as the status log:
    # the DF tier shells out to `df` via COPY FROM PROGRAM, which is too
    # heavy to run on every sub-second iteration of a large relation.
    last_disk_check = 0.0
    disk_check_interval = 30.0
    last_free_disk_bytes: int | None = None

    # Adaptive-throttle baseline: EWMA of per-row rewrite time, and how many
    # rewrites we've timed.  Used to detect abnormally slow rewrites without
    # any fixed absolute threshold (see _THROTTLE_SLOW_FACTOR).
    ewma_per_row = 0.0
    throttle_samples = 0

    # Live fragmentation map state.  Rebuilt after each VACUUM (when the FSM
    # actually changes); a fresh map is attached to the next progress update
    # and then cleared, so intervening updates keep showing the last one.
    frag_map_pending: list[str] | None = None
    # The relocation unit footprint that classifies reclaimable-now vs
    # sub-unit holes on the map (a full TOAST chunk for TOAST; a whole page
    # for the heap, so only genuinely large holes read as reclaimable).
    from pg_compact.fsm_predict import build_fragmentation_map
    map_unit = target.map_unit_footprint

    def _refresh_frag_map() -> None:
        """Rebuild the fragmentation map (one FSM scan); safe to call rarely."""
        nonlocal frag_map_pending
        try:
            pages = db.get_relation_page_count(conn, target.relation)
            frag_map_pending = build_fragmentation_map(
                conn, target.relation, pages, map_unit, ncells=60,
            )
        except Exception:  # a map failure must never break compaction
            frag_map_pending = None

    def _rate_and_eta() -> tuple[float, float]:
        """(MB/s, ETA seconds) from real page progress.  Single source of
        truth so the flat log and the interactive bar agree."""
        now = time.monotonic()
        done = pages_before - cursor
        rate_pages = done / (now - start_time) if now > start_time else 0.0
        remaining = max(cursor - target.ideal_pages, 0)
        eta_s = (remaining / rate_pages) if rate_pages > 0 else 0.0
        return rate_pages * block_size / 1e6, eta_s

    def _emit_progress(status_icon: str = "", phase: str | None = None) -> None:
        """Send a progress snapshot to the UI, optionally labelling the
        current activity (throttle, waiting for disk, or VACUUM)."""
        # Report the whole-table effective size (see _effective_total_bytes)
        # so before/current/target are on the same scale and match the flat
        # status log.
        nonlocal frag_map_pending
        mb_s, eta_s = _rate_and_eta()
        progress(ProgressUpdate(
            phase=phase or target.phase_name,
            pages_done=pages_before - cursor,
            pages_total=reclaimable_pages,
            current_size_bytes=_effective_total_bytes(),
            before_size_bytes=before_total_bytes or None,
            target_size_bytes=target_bytes or None,
            free_disk_bytes=last_free_disk_bytes,
            min_free_disk_bytes=(
                config.min_free_disk_mb * 1024 * 1024 if config.disk_check else None
            ),
            status_icon=status_icon,
            mb_per_s=mb_s,
            eta_seconds=eta_s,
            frag_map=frag_map_pending,
            frag_map_label=(target.phase_name if frag_map_pending is not None else None),
        ))
        # A pending map is delivered once; later updates keep showing the last.
        frag_map_pending = None

    def _do_vacuum() -> tuple[int, int]:
        """VACUUM and return (pages_before_vacuum, pages_after_vacuum).

        Measures the real physical size just before VACUUM (it can differ
        from the last-known pages_current: between vacuums the relation may
        have grown if relocations spilled past the available free space),
        so callers can report an accurate before -> after.
        """
        nonlocal pages_current, cursor, cleared_since_vacuum, landing_estimate
        # Surface the VACUUM in the interactive UI: it can take seconds on a
        # large relation, and otherwise the display would appear stalled.
        _emit_progress(status_icon=_ICON_VACUUM)
        before_vacuum = db.get_relation_page_count(conn, target.relation)
        target.vacuum(conn)
        new_pages = db.get_relation_page_count(conn, target.relation)
        pages_current = new_pages
        cursor = min(cursor, new_pages)
        cleared_since_vacuum = 0
        # VACUUM changes the FSM; re-measure the true landing capacity and
        # refresh the live fragmentation map from the same fresh FSM state.
        landing_estimate = db.relation_free_pages_before(conn, target.relation, pages_current)
        _refresh_frag_map()
        return before_vacuum, new_pages

    def _effective_total_bytes() -> int:
        """Whole-table size projected after the pending VACUUM truncates the
        rows relocated so far.  Kept consistent with _emit_progress so the
        flat log and the interactive bar report the same figures."""
        relocated_bytes = (pages_before - cursor) * block_size
        if before_total_bytes:
            return max(before_total_bytes - relocated_bytes, 0)
        return cursor * block_size

    def _log_status(free_disk_bytes: int | None = None) -> None:
        nonlocal last_status_log
        now = time.monotonic()
        if now - last_status_log < status_log_interval:
            return
        # Progress is measured by how far the cursor has walked down, not
        # by physical size: with deferred VACUUM the file stays flat between
        # truncations, but the rows in (cursor, pages_before] are already
        # relocated.  Report whole-table figures (before_total_bytes and the
        # projected effective size) so this matches the interactive bar.
        done = pages_before - cursor
        pct = (100.0 * done / reclaimable_pages) if reclaimable_pages else 0.0
        mb_s, eta_s = _rate_and_eta()
        before_bytes = before_total_bytes or (pages_before * block_size)
        target_disp = target_bytes or (target.ideal_pages * block_size)
        disk_str = f", disk free {fmt_bytes(free_disk_bytes)}" if free_disk_bytes is not None else ""
        # When a live progress UI is consuming updates, the bar already
        # shows this same line continuously, so demote the flat copy to
        # debug (visible only with -v).  Without a live UI this periodic
        # line is the primary progress output, so keep it at info.
        level = "debug" if progress is not _noop_progress else "info"
        log(level,
            f"{target.phase_name}: {pct:.1f}% - "
            f"{fmt_bytes(before_bytes)} -> {fmt_bytes(_effective_total_bytes())} "
            f"(target {fmt_bytes(target_disp)}), "
            f"{mb_s:.1f} MB/s, ETA {fmt_duration(eta_s)}{disk_str}.")
        last_status_log = now

    if target.setup:
        # setup() can be slow (e.g. the PG12-16 chunk-map build scans the
        # whole heap, ~1-2 min on a large table).  Label the bar so it
        # doesn't sit on "starting..." with no explanation.
        _emit_progress(phase=f"{target.phase_name}: preparing")
        target.setup(conn)
    try:
        try:
            # Prime the FSM: a relation that has never been vacuumed (or was
            # just bulk-deleted) reports no free space, which would stall the
            # landing-capacity check on the first iteration.  A cheap VACUUM
            # records the reusable space left by dead tuples so the loop can
            # see where relocated rows may land.
            _emit_progress(phase=f"{target.phase_name}: initial VACUUM", status_icon=_ICON_VACUUM)
            _do_vacuum()
            # Deliver the first fragmentation map right after priming, so the
            # interactive display shows the starting layout even if the loop
            # below does little work.
            _emit_progress()

            max_iterations = reclaimable_pages + 100
            for _iteration in range(1, max_iterations + 1):
                if cursor <= target.ideal_pages:
                    break

                # Landing capacity check, using the cached estimate to avoid
                # a per-iteration pg_freespace() scan.  When the estimate
                # runs low, re-measure exactly; if it is genuinely low,
                # vacuum to reclaim the dead tail and expose fresh space.
                if landing_estimate < _INITIAL_TAIL_WINDOW_PAGES:
                    landing_estimate = db.relation_free_pages_before(
                        conn, target.relation, cursor,
                    )
                if landing_estimate < 1:
                    if cleared_since_vacuum > 0:
                        # Dead tail is holding space that VACUUM can reclaim.
                        _do_vacuum()
                        continue
                    # Even after vacuuming, no free space lies ahead of the
                    # cursor.  Remaining bloat is interleaved with live rows
                    # (internal fragmentation) and cannot be reclaimed by
                    # relocating tail rows.
                    log("notice", f"{target.phase_name}: no landing space ahead of tail; stopping.")
                    break

                # Window at the current tail.  Size it so relocated rows fit
                # in the free space ahead: bounded by the landing estimate,
                # the initial trial size, and _MAX for very large relations.
                window_pages = min(
                    _INITIAL_TAIL_WINDOW_PAGES,
                    max(cursor - target.ideal_pages, 1),
                    _MAX_TAIL_WINDOW_PAGES,
                    max(landing_estimate, 1),
                )
                win_start = max(cursor - window_pages, target.ideal_pages)
                win_end = cursor - 1

                ctids = target.collect_ctids(conn, win_start, win_end)

                throttling = False
                if ctids:
                    start = time.monotonic()
                    touched = target.rewrite(conn, ctids)
                    elapsed = time.monotonic() - start
                    rows_rewritten += touched

                    # Adaptive throttle.  A normal window on a large relation
                    # always takes several seconds, so an absolute time
                    # threshold would fire every iteration.  Instead compare
                    # this rewrite's per-row cost against a running average:
                    # throttle only when it ran markedly slower than the
                    # recent norm (lock contention, an I/O spike), which is
                    # when giving the server a breather actually helps.  The
                    # pause is proportional to the rewrite's duration, capped
                    # by --throttle-delay-s.  Disabled entirely by
                    # --no-throttle.
                    if config.throttle and touched > 0:
                        per_row = elapsed / touched
                        abnormal = (
                            throttle_samples >= _THROTTLE_MIN_SAMPLES
                            and elapsed >= _THROTTLE_MIN_ELAPSED_S
                            and per_row > ewma_per_row * _THROTTLE_SLOW_FACTOR
                        )
                        if abnormal:
                            pause = min(config.throttle_delay_s, elapsed * _THROTTLE_RATIO)
                            if pause > 0:
                                throttling = True
                                time.sleep(pause)
                        # Update the baseline from this sample (after the
                        # decision, so a spike doesn't mask itself).
                        if throttle_samples == 0:
                            ewma_per_row = per_row
                        else:
                            ewma_per_row = (
                                _THROTTLE_EWMA_ALPHA * per_row
                                + (1 - _THROTTLE_EWMA_ALPHA) * ewma_per_row
                            )
                        throttle_samples += 1

                # Advance the cursor past this window regardless — its rows
                # are relocated (dead tail), pending VACUUM to truncate.
                window_span = cursor - win_start
                cleared_since_vacuum += window_span
                cursor = win_start
                # The relocated rows consumed roughly a window's worth of
                # free pages ahead; decrement the cached estimate so we know
                # when to re-measure.  (Truncation via VACUUM refreshes it.)
                landing_estimate = max(landing_estimate - window_span, 0)

                # Disk-space check, throttled to disk_check_interval.  The
                # measured free space feeds the progress update and status
                # log so it is always visible; if space runs low, block
                # until it recovers.
                if config.disk_check and (
                    time.monotonic() - last_disk_check >= disk_check_interval
                ):
                    from pg_compact.disk_guard import (
                        check_disk_space,
                        wait_for_disk_space,
                    )
                    status = check_disk_space(conn, config.min_free_disk_mb)
                    if not status.ok:
                        # Keep the UI alive while blocked on disk, labelled
                        # so it renders "waiting disk" rather than "throttle".
                        wait_for_disk_space(
                            conn, config.min_free_disk_mb, log,
                            on_wait=lambda: _emit_progress(
                                status_icon=_ICON_PAUSE, phase=_PHASE_WAITING_DISK,
                            ),
                        )
                        status = check_disk_space(conn, config.min_free_disk_mb)
                    if status.free_mb is not None:
                        last_free_disk_bytes = int(status.free_mb * 1024 * 1024)
                    last_disk_check = time.monotonic()

                _log_status(last_free_disk_bytes)
                _emit_progress(status_icon=_ICON_PAUSE if throttling else "")

                # Periodic VACUUM to truncate the accumulated dead tail, and
                # the plateau check.  A productive VACUUM truncates about as
                # many pages as we just cleared (the relocated rows emptied
                # the tail); once relocation runs out of usable free space
                # ahead, the emptied pages no longer land at the tail and the
                # VACUUM truncates almost nothing despite a full batch.  The
                # two regimes are far apart, so a single VACUUM that reclaims
                # less than _PLATEAU_TRUNCATE_RATIO of what we cleared means
                # the relation can no longer be shrunk online — stop.
                if cleared_since_vacuum >= vacuum_every:
                    cleared = cleared_since_vacuum
                    before_vac, after_vac = _do_vacuum()
                    truncated = before_vac - after_vac
                    log("info",
                        f"{target.phase_name}: VACUUM truncated {truncated} pages "
                        f"({before_vac} -> {after_vac}).")
                    if truncated < cleared * _PLATEAU_TRUNCATE_RATIO:
                        log("notice",
                            f"{target.phase_name}: no landing space left ahead of tail; stopping.")
                        break

            # Final VACUUM to truncate whatever dead tail remains.
            if cleared_since_vacuum > 0:
                before_vac, after_vac = _do_vacuum()
                log("info",
                    f"{target.phase_name}: final VACUUM truncated {before_vac - after_vac} pages "
                    f"({before_vac} -> {after_vac}).")
        except psycopg.errors.QueryCanceled:
            conn.rollback()
            log("notice",
                f"{target.phase_name}: cancelled (timeout); partial progress kept.")
    finally:
        if target.teardown:
            target.teardown(conn)

    shrunk = pages_before - pages_current
    if shrunk > 0:
        log("info",
            f"{target.phase_name}: {pages_before} -> {pages_current} pages "
            f"({shrunk} reclaimed, {rows_rewritten} rows rewritten).")
    else:
        log("notice",
            f"{target.phase_name}: no pages reclaimed after {rows_rewritten} row rewrites.")

    return Outcome.COMPLETED if shrunk > 0 else Outcome.INCOMPLETE_STUCK


# ---------------------------------------------------------------------------
# Heap target
# ---------------------------------------------------------------------------


MAX_DRAIN_ATTEMPTS = 50


def _make_heap_target(
    conn: psycopg.Connection,
    schema: str,
    table: str,
    column: str,
    ideal_pages: int,
) -> CompactionTarget:
    """Build a CompactionTarget for heap compaction.

    The rewrite callback uses drain-within-transaction: it repeatedly
    UPDATEs the same page range inside a single transaction until all
    rows leave.  This is necessary because ``SET col = col`` on an
    indexed column does not change the value, so PG may apply a HOT
    update and keep the row on the same page.  Within a transaction,
    dead tuples are not vacuumed, so the page fills up and PG is
    forced to place the next version on an earlier page.
    """
    qname = db.qualified_name(conn, schema, table)
    tbl_ident = sql.Identifier(schema, table)
    col_ident = sql.Identifier(column)

    def collect_ctids(c: psycopg.Connection, from_page: int, to_page: int) -> list[str]:
        with c.cursor() as cur:
            cur.execute(
                sql.SQL(
                    "SELECT ctid FROM ONLY {tbl} "
                    "WHERE ctid >= {lo}::tid AND ctid < {hi}::tid"
                ).format(
                    tbl=tbl_ident,
                    lo=sql.Literal(f"({from_page},0)"),
                    hi=sql.Literal(f"({to_page + 1},0)"),
                )
            )
            return [str(row[0]) for row in cur.fetchall()]

    def rewrite(c: psycopg.Connection, ctids: list[str]) -> int:
        """Drain rows from tail pages within a single transaction.

        Repeatedly UPDATE rows still on pages >= from_page until they all
        leave or MAX_DRAIN_ATTEMPTS is reached.  On failure the transaction
        is rolled back (rows stay put, no harm done).

        A safety guard aborts early if the relation is *growing* during the
        drain: that means the new row versions are being appended to fresh
        pages instead of relocating into earlier free space (no usable free
        space ahead), which would otherwise multiply row versions and bloat
        the file — the drain is only productive when there is room ahead.
        """
        if not ctids:
            return 0
        # Determine from_page from the lowest ctid in the list.
        from_page = min(int(t.strip("()").split(",")[0]) for t in ctids)
        update_q = sql.SQL(
            "UPDATE ONLY {tbl} SET {col} = {col} "
            "WHERE ctid >= {lo}::tid "
            "RETURNING ctid"
        ).format(
            tbl=tbl_ident,
            col=col_ident,
            lo=sql.Literal(f"({from_page},0)"),
        )
        pages_before_drain = db.get_relation_page_count(c, qname)
        total = 0
        with c.transaction():
            for _attempt in range(MAX_DRAIN_ATTEMPTS):
                with c.cursor() as cur:
                    cur.execute(update_q)
                    rows = cur.fetchall()
                if not rows:
                    return total
                total += len(rows)
                # Check if all returned ctids are below from_page.
                still_stuck = any(
                    int(str(r[0]).strip("()").split(",")[0]) >= from_page
                    for r in rows
                )
                if not still_stuck:
                    return total
                # Abort if the relation is growing: the rows are not
                # relocating into earlier free space, they are extending the
                # file.  Rolling back avoids leaving that bloat behind.
                if db.get_relation_page_count(c, qname) > pages_before_drain:
                    raise psycopg.Rollback()
            # Max attempts reached — roll back to avoid leaving dead tuples.
            raise psycopg.Rollback()
        return total  # after rollback: 0 effectively

    def vacuum(c: psycopg.Connection) -> None:
        db.vacuum_relation(c, qname)

    # Map classification unit for the heap = the mean live-tuple on-page
    # footprint (so the map's "reclaimable now" matches the online metric,
    # which the drain-in-transaction loop can actually fill).
    from pg_compact.fsm_predict import onpage_footprint

    avg = db.avg_row_size(conn, schema, table)
    map_unit = onpage_footprint(avg) if avg else 2048

    return CompactionTarget(
        relation=qname,
        ideal_pages=ideal_pages,
        collect_ctids=collect_ctids,
        rewrite=rewrite,
        vacuum=vacuum,
        phase_name="heap compaction",
        map_unit_footprint=map_unit,
    )


# ---------------------------------------------------------------------------
# TOAST targets
# ---------------------------------------------------------------------------


def _make_toast_target_native(
    conn: psycopg.Connection,
    schema: str,
    table: str,
    columns: list,
    toast_relation: str,
    ideal_pages: int,
) -> CompactionTarget:
    """TOAST CompactionTarget for PG17+ using pg_column_toast_chunk_id."""
    from pg_compact.toast import _toast_rewrite_assignments

    tbl_ident = sql.Identifier(schema, table)
    map_column = columns[0]
    col_ident = sql.Identifier(map_column.name)
    assignments = _toast_rewrite_assignments(columns)

    def collect_ctids(c: psycopg.Connection, from_page: int, to_page: int) -> list[str]:
        with c.cursor() as cur:
            cur.execute(
                sql.SQL(
                    "SELECT t.ctid FROM {tbl} t "
                    "WHERE pg_column_toast_chunk_id(t.{col}) IN ("
                    "  SELECT DISTINCT chunk_id FROM {toast} "
                    "  WHERE ctid >= {lo}::tid AND ctid < {hi}::tid "
                    "  AND chunk_seq = 0"
                    ")"
                ).format(
                    tbl=tbl_ident,
                    col=col_ident,
                    toast=sql.SQL(toast_relation),
                    lo=sql.Literal(f"({from_page},0)"),
                    hi=sql.Literal(f"({to_page + 1},0)"),
                )
            )
            return [str(row[0]) for row in cur.fetchall()]

    def rewrite(c: psycopg.Connection, ctids: list[str]) -> int:
        ctid_lits = sql.SQL(", ").join(sql.Literal(t) + sql.SQL("::tid") for t in ctids)
        q = sql.SQL(
            "UPDATE ONLY {tbl} SET {assigns} WHERE ctid = ANY(ARRAY[{ctids}])"
        ).format(
            tbl=tbl_ident,
            assigns=sql.SQL(", ").join(assignments),
            ctids=ctid_lits,
        )
        with c.cursor() as cur:
            cur.execute(q)
            return cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0

    def vacuum(c: psycopg.Connection) -> None:
        db.vacuum_relation(c, toast_relation)

    return CompactionTarget(
        relation=toast_relation,
        ideal_pages=ideal_pages,
        collect_ctids=collect_ctids,
        rewrite=rewrite,
        vacuum=vacuum,
        phase_name="TOAST compaction",
        map_unit_footprint=FULL_TOAST_CHUNK_FOOTPRINT,
    )


def _make_toast_target_bulk(
    conn: psycopg.Connection,
    schema: str,
    table: str,
    columns: list,
    toast_relation: str,
    toast_relid: int,
    ideal_pages: int,
    log: LogFn,
) -> CompactionTarget:
    """TOAST CompactionTarget for PG12-16 using pageinspect bulk scan."""
    from pg_compact.toast import (
        _toast_rewrite_assignments,
        build_chunk_map,
        create_chunk_map_function,
        drop_chunk_map,
        lookup_chunk_map,
    )

    tbl_ident = sql.Identifier(schema, table)
    assignments = _toast_rewrite_assignments(columns)

    def _build(c: psycopg.Connection) -> int:
        heap_pages = db.get_size_stats(c, schema, table).page_count
        log("notice", f"Building chunk map ({heap_pages} pages)...")
        n = build_chunk_map(c, schema, table, toast_relid, heap_pages, log)
        log("notice", f"Chunk map ready ({n} chunks).")
        return n

    def setup(c: psycopg.Connection) -> None:
        create_chunk_map_function(c)
        _build(c)

    def teardown(c: psycopg.Connection) -> None:
        drop_chunk_map(c)

    def collect_ctids(c: psycopg.Connection, from_page: int, to_page: int) -> list[str]:
        chunk_ids = db.get_toast_chunk_ids_on_pages(c, toast_relation, from_page, to_page)
        if not chunk_ids:
            return []
        pairs = lookup_chunk_map(c, chunk_ids)
        # The map is built once, but each rewrite re-TOASTs its value into
        # NEW chunk_ids that land in free space below the tail.  When the
        # cursor later reaches those pages, their new chunk_ids are absent
        # from the static map and the rows would be silently skipped, so the
        # region never drains.  Detect that staleness cheaply — many on-page
        # chunk_ids not found in the map — and rebuild before proceeding, so
        # collect_ctids stays complete.
        if len(pairs) < len(chunk_ids) * _CHUNK_MAP_STALE_RATIO:
            drop_chunk_map(c)
            _build(c)
            pairs = lookup_chunk_map(c, chunk_ids)
        return [ctid_str for ctid_str, _ in pairs]

    def rewrite(c: psycopg.Connection, ctids: list[str]) -> int:
        ctid_lits = sql.SQL(", ").join(sql.Literal(t) + sql.SQL("::tid") for t in ctids)
        q = sql.SQL(
            "UPDATE ONLY {tbl} SET {assigns} WHERE ctid = ANY(ARRAY[{ctids}])"
        ).format(
            tbl=tbl_ident,
            assigns=sql.SQL(", ").join(assignments),
            ctids=ctid_lits,
        )
        with c.cursor() as cur:
            cur.execute(q)
            return cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0

    def vacuum(c: psycopg.Connection) -> None:
        db.vacuum_relation(c, toast_relation)

    return CompactionTarget(
        relation=toast_relation,
        ideal_pages=ideal_pages,
        collect_ctids=collect_ctids,
        rewrite=rewrite,
        vacuum=vacuum,
        setup=setup,
        teardown=teardown,
        phase_name="TOAST compaction",
        map_unit_footprint=FULL_TOAST_CHUNK_FOOTPRINT,
    )


# ---------------------------------------------------------------------------
# TOAST dispatch
# ---------------------------------------------------------------------------


def _run_toast_compaction(
    conn: psycopg.Connection,
    schema: str,
    table: str,
    columns: list,
    toast_relation: str,
    config: CompactionConfig,
    log: LogFn,
    progress: ProgressFn = _noop_progress,
    before_total_bytes: int = 0,
    target_bytes: int = 0,
) -> Outcome:
    """Pick the right TOAST strategy and run the shared compaction loop."""
    toast_pages = db.get_relation_page_count(conn, toast_relation)
    if toast_pages == 0:
        return Outcome.COMPLETED

    # Base the target on *usable* free space only.  Sub-chunk holes (free
    # space smaller than one TOAST chunk, scattered between live chunks)
    # cannot be reclaimed online, so counting them would set an unreachable
    # target and make the loop churn against structural fragmentation.
    usable_free, _subchunk_holes = db.relation_free_space_split(conn, toast_relation)
    ideal_pages = max(toast_pages - usable_free // 8192, 0)

    if db.has_toast_chunk_id_func(conn):
        log("notice", "TOAST: targeting chunks via pg_column_toast_chunk_id (PG17+).")
        target = _make_toast_target_native(
            conn, schema, table, columns, toast_relation, ideal_pages,
        )
    else:
        log("notice", "TOAST: targeting chunks via pageinspect scan.")
        qname = db.qualified_name(conn, schema, table)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT reltoastrelid::bigint FROM pg_class WHERE oid = %s::regclass",
                (qname,),
            )
            toast_relid: int = cur.fetchone()[0]  # type: ignore[index]
        target = _make_toast_target_bulk(
            conn, schema, table, columns, toast_relation,
            toast_relid, ideal_pages, log,
        )

    return _compact_relation(
        conn, target, config, log, progress,
        before_total_bytes, target_bytes,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def compact_table(
    conn: psycopg.Connection,
    schema: str,
    table: str,
    config: CompactionConfig,
    log: LogFn = noop_log,
    progress: ProgressFn = _noop_progress,
) -> CompactionResult:
    result = CompactionResult(outcome=Outcome.COMPLETED, schema=schema, table=table)

    if not db.table_exists(conn, schema, table):
        result.outcome = Outcome.SKIPPED_EMPTY
        result.message = f'Table "{schema}"."{table}" does not exist, skipping.'
        return result

    if db.has_blocking_triggers(conn, schema, table):
        result.outcome = Outcome.SKIPPED_TRIGGERS
        result.message = (
            "Has ENABLE ALWAYS/REPLICA UPDATE triggers; skipping to avoid "
            "firing them during compaction."
        )
        return result

    locked = db.try_advisory_lock(conn, schema, table)
    if not locked:
        result.outcome = Outcome.SKIPPED_LOCKED
        result.message = "Another pg-compact (or advisory-lock holder) is already working on this table."
        return result

    try:
        if config.initial_vacuum and not config.dry_run:
            log("info", "Running initial VACUUM...")
            db.vacuum(conn, schema, table)

        size_before = db.get_size_stats(conn, schema, table)
        result.size_before = size_before

        if size_before.page_count <= 1:
            result.outcome = Outcome.SKIPPED_EMPTY
            result.message = "Table is empty or a single page; nothing to compact."
            return result

        bloat = get_bloat_stats(conn, schema, table, log)

        result.bloat_before = bloat

        # The interactive banner (rendered by the CLI before this runs)
        # already shows the size and bloat breakdown, so demote these lines
        # to debug when a live UI is active to avoid printing it all twice;
        # keep them at info in non-interactive mode where this is the only
        # place the breakdown appears.
        stat_level = "debug" if progress is not _noop_progress else "info"

        log(stat_level,
            f"Current size: {fmt_bytes(size_before.total_bytes)} "
            f"(heap {fmt_bytes(size_before.table_bytes)}"
            f"{f', TOAST {fmt_bytes(size_before.toast_bytes)}' if size_before.toast_bytes else ''}"
            f", indexes {fmt_bytes(size_before.indexes_bytes)}).")

        # Log heap bloat breakdown.
        heap_bytes = size_before.table_bytes

        def _val(v: int) -> str:
            return f"{fmt_bytes(v):>10}"

        def _pct(v: float) -> str:
            return f"{v:>5.1f}%"

        def _pct_of_heap(v: int) -> str:
            return _pct(100.0 * v / heap_bytes) if heap_bytes else _pct(0.0)

        # Breakdown of the relation size (three top-level parts sum to 100%):
        #   * Live data        — rows plus unavoidable per-page overhead
        #   * Unusable padding — structural per-page alignment; no tool frees it
        #   * Bloat            — the genuine bloat a full rewrite (VACUUM FULL /
        #                        pg_repack) would return.
        # Bloat is itemised into two indented sub-lines that sum to it:
        #   - "pg-compact frees now" — what this tool relocates + truncates in
        #     place (the online figure; an upper bound from one FSM scan)
        #   - "needs VACUUM FULL"    — the remainder only a full rewrite frees
        # so "can free now" is never confused with the full-rewrite total.
        heap_live = max(heap_bytes - bloat.free_bytes, 0)
        heap_vf_only = max(bloat.reclaimable_bytes - bloat.online_reclaimable_bytes, 0)
        log(stat_level, f"Heap ({fmt_bytes(heap_bytes)}, "
            f"{size_before.page_count} pages, via FSM):")
        log(stat_level, f"  Live data               {_val(heap_live)}  {_pct_of_heap(heap_live)}")
        log(stat_level, f"  Unusable padding        {_val(bloat.alignment_waste_bytes)}  "
            f"{_pct_of_heap(bloat.alignment_waste_bytes)}  never reclaimable")
        log(stat_level, f"  Bloat                   {_val(bloat.reclaimable_bytes)}  "
            f"{_pct_of_heap(bloat.reclaimable_bytes)}")
        log(stat_level, f"    pg-compact frees now  {_val(bloat.online_reclaimable_bytes)}  "
            f"{_pct_of_heap(bloat.online_reclaimable_bytes)}")
        log(stat_level, f"    needs VACUUM FULL     {_val(heap_vf_only)}  "
            f"{_pct_of_heap(heap_vf_only)}")

        # Check TOAST bloat early so the skip decision accounts for it.
        toast_bloat: BloatStats | None = None
        if config.toast_compact:
            toast_bloat = get_toast_bloat_stats(conn, schema, table)
            if toast_bloat is not None and toast_bloat.free_percent > 0:
                toast_bytes = size_before.toast_bytes or 0

                def _tpct_of(v: int) -> str:
                    return _pct(100.0 * v / toast_bytes) if toast_bytes else _pct(0.0)

                # Same breakdown as the heap: Live data / Unusable padding /
                # Bloat sum to 100%, with Bloat itemised into the "frees now"
                # subset and the "needs VACUUM FULL" remainder (which sum to it).
                toast_live = max(toast_bytes - toast_bloat.free_bytes, 0)
                toast_vf_only = max(
                    toast_bloat.reclaimable_bytes - toast_bloat.online_reclaimable_bytes, 0
                )
                log(stat_level, f"TOAST ({fmt_bytes(toast_bytes)}, via FSM):")
                log(stat_level, f"  Live data               {_val(toast_live)}  {_tpct_of(toast_live)}")
                log(stat_level, f"  Unusable padding        {_val(toast_bloat.alignment_waste_bytes)}  "
                    f"{_tpct_of(toast_bloat.alignment_waste_bytes)}  never reclaimable")
                log(stat_level, f"  Bloat                   {_val(toast_bloat.reclaimable_bytes)}  "
                    f"{_tpct_of(toast_bloat.reclaimable_bytes)}")
                log(stat_level, f"    pg-compact frees now  {_val(toast_bloat.online_reclaimable_bytes)}  "
                    f"{_tpct_of(toast_bloat.online_reclaimable_bytes)}")
                log(stat_level, f"    needs VACUUM FULL     {_val(toast_vf_only)}  "
                    f"{_tpct_of(toast_vf_only)}")

        # Skip only if BOTH heap AND toast are below threshold.
        # Worth compacting is judged by what pg-compact can actually free in
        # place (the "can free now" / online figure), NOT the full VACUUM
        # FULL floor — otherwise we'd start a phase whose bloat is almost
        # entirely "needs VACUUM FULL" and make no progress.
        heap_online_pct = (
            100.0 * bloat.online_reclaimable_bytes / heap_bytes if heap_bytes else 0.0
        )
        heap_worth = config.force or heap_online_pct >= config.min_compact_percent
        toast_online_pct = 0.0
        if toast_bloat is not None and (size_before.toast_bytes or 0) > 0:
            toast_online_pct = 100.0 * toast_bloat.online_reclaimable_bytes / size_before.toast_bytes
        toast_worth = toast_online_pct >= config.min_compact_percent
        if not config.force and not heap_worth and not toast_worth:
            result.outcome = Outcome.SKIPPED_BELOW_THRESHOLD
            msg_parts = [f"heap {heap_online_pct:.1f}%"]
            if toast_bloat is not None:
                msg_parts.append(f"TOAST {toast_online_pct:.1f}%")
            result.message = (
                f"Can free now ({', '.join(msg_parts)}) < {config.min_compact_percent:.0f}% minimum; "
                "skipping (--force to override).")
            return result

        if not config.force and size_before.page_count < config.min_compact_pages:
            result.outcome = Outcome.SKIPPED_BELOW_THRESHOLD
            result.message = (
                f"{size_before.page_count} pages < {config.min_compact_pages}-page minimum; "
                "skipping (--force to override).")
            return result

        if config.dry_run:
            toast_online = toast_bloat.online_reclaimable_bytes if toast_bloat else 0
            toast_reclaimable = toast_bloat.reclaimable_bytes if toast_bloat else 0
            online_total = bloat.online_reclaimable_bytes + toast_online
            reclaimable_total = bloat.reclaimable_bytes + toast_reclaimable
            result.message = (
                f"Dry run: {fmt_bytes(reclaimable_total)} bloat; "
                f"frees ~{fmt_bytes(online_total)} now (upper bound), "
                f"rest needs VACUUM FULL."
            )
            return result

        column = db.pick_update_column(conn, schema, table)
        if column is None:
            result.outcome = Outcome.SKIPPED_EMPTY
            result.message = "Table has no ordinary columns to drive the no-op UPDATE."
            return result

        idx_col = db.pick_indexed_column(conn, schema, table)
        if idx_col is not None:
            column = idx_col
        log("info", f"Driving relocation via column {column!r}.")

        # Progress is reported PER PHASE: heap and TOAST run sequentially with
        # very different mechanics and throughput, so a shared whole-table bar
        # would mix their rates/ETA and jump between phases.  Each phase gets
        # its own before -> target on its own relation's scale (the map is
        # already per-phase); the overall before/after is shown in the final
        # summary.  Targets use the ONLINE-achievable size (subtract only the
        # "Reclaimable now" part), never the full VACUUM FULL floor, so the
        # bar can actually reach 100%.
        toast_online = toast_bloat.online_reclaimable_bytes if toast_bloat is not None else 0

        # --- Heap compaction ---
        heap_outcome = Outcome.COMPLETED
        if heap_worth:
            ideal_heap_pages = max(
                size_before.page_count - bloat.online_reclaimable_bytes // 8192, 0
            )
            heap_target_bytes = max(size_before.table_bytes - bloat.online_reclaimable_bytes, 0)

            heap_target = _make_heap_target(conn, schema, table, column, ideal_heap_pages)

            db.set_replica_role(conn)
            db.set_lock_timeout(conn, config.lock_timeout_ms)
            try:
                heap_outcome = _compact_relation(
                    conn, heap_target, config, log, progress,
                    before_total_bytes=size_before.table_bytes,
                    target_bytes=heap_target_bytes,
                )
            finally:
                db.reset_replication_role(conn)
        else:
            log("notice", f"Heap: can free now < {config.min_compact_percent:.0f}%; skipping heap phase.")

        # --- TOAST compaction ---
        if config.toast_compact:
            from pg_compact.toast import get_toastable_columns

            toastable_columns = get_toastable_columns(conn, schema, table)
            if toastable_columns:
                toast_rel = get_toast_relation(conn, schema, table)
                toast_relation_bytes = db.get_relation_size(conn, toast_rel)
                toast_bloat_now = get_toast_bloat_stats(conn, schema, table)
                if toast_bloat_now is not None:
                    result.toast_bloat_before = toast_bloat_now
                    toast_bloat = toast_bloat_now

                if toast_relation_bytes == 0:
                    should_rewrite = False
                elif config.force:
                    should_rewrite = True
                else:
                    should_rewrite = toast_bloat is None or toast_bloat.free_percent >= config.min_compact_percent

                if should_rewrite:
                    result.toast_columns_rewritten = [c.name for c in toastable_columns]
                    log("info", f"Rewriting TOASTed column(s): {', '.join(c.name for c in toastable_columns)}...")

                    assert toast_rel is not None
                    # Per-phase scale: TOAST relation's own size and its
                    # online-achievable target.
                    toast_before = toast_relation_bytes
                    toast_target_bytes = max(toast_before - toast_online, 0)
                    db.set_replica_role(conn)
                    try:
                        toast_outcome = _run_toast_compaction(
                            conn, schema, table, toastable_columns,
                            toast_rel, config, log, progress,
                            before_total_bytes=toast_before,
                            target_bytes=toast_target_bytes,
                        )
                    finally:
                        db.reset_replication_role(conn)
                elif toast_bloat is not None:
                    toast_outcome = Outcome.COMPLETED
                    log("notice",
                        f"TOAST: free {toast_bloat.free_percent:.1f}% < "
                        f"{config.min_compact_percent:.0f}%; skipping TOAST rewrite.")
                else:
                    toast_outcome = Outcome.COMPLETED
        else:
            toast_outcome = Outcome.COMPLETED

        # Overall outcome: COMPLETED if either phase made progress.
        if heap_outcome == Outcome.COMPLETED or toast_outcome == Outcome.COMPLETED:
            result.outcome = Outcome.COMPLETED
        else:
            result.outcome = heap_outcome

        with conn.cursor() as cur:
            cur.execute("RESET lock_timeout")

        try:
            log("info", "Running final VACUUM...")
            db.vacuum(conn, schema, table)

            if config.reindex:
                from pg_compact.reindex import reindex_table

                log("info", "Reindexing (REINDEX CONCURRENTLY)...")
                result.reindexed = reindex_table(conn, schema, table, log)

            log("info", "Running final ANALYZE...")
            db.analyze_table(conn, schema, table)
        except psycopg.errors.QueryCanceled:
            # A statement timeout cancelled final maintenance (e.g. VACUUM
            # blocked by a long-running transaction).  Not fatal — the
            # compaction itself succeeded; final cleanup can be retried.
            conn.rollback()
            log("notice", "Final maintenance cancelled (timeout); skipping.")

        result.size_after = db.get_size_stats(conn, schema, table)
        result.bloat_after = get_bloat_stats(conn, schema, table)

        return result
    finally:
        db.advisory_unlock(conn, schema, table)
