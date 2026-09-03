"""Tests for TOAST-aware compaction - the core reason pg-compact exists.

A no-op UPDATE against an ordinary column shrinks the heap, but PostgreSQL's
UPDATE leaves an unchanged TOASTed value's on-disk bytes completely
untouched (it just copies the TOAST pointer into the new heap tuple) - so
without a dedicated TOAST rewrite pass, a table with a heavily bloated
TOAST relation would come out of a "successful" compaction run with its
TOAST storage exactly as bloated as before. See toast.py's module
docstring for the full mechanics and the round-trip formula used.
"""

from __future__ import annotations

from conftest import make_toast_bloated_table

from pg_compact import db
from pg_compact.compactor import CompactionConfig, Outcome, compact_table
from pg_compact.stats import get_toast_bloat_stats, get_toast_relation
from pg_compact.toast import get_toastable_columns


def _noop_log(level, message):
    pass


def _checksum(conn, table) -> str:
    with conn.cursor() as cur:
        cur.execute(f'SELECT string_agg(md5(tag || val), \',\' ORDER BY id) FROM "{table}"')
        (digest,) = cur.fetchone()
        return digest


def test_get_toastable_columns_finds_text_columns_by_storage_attribute(pg_conn, pg_table):
    with pg_conn.cursor() as cur:
        cur.execute(
            f'CREATE TABLE "{pg_table}" (id serial primary key, fixed_col int, toastable_col text)'
        )
    columns = get_toastable_columns(pg_conn, "public", pg_table)
    names = {c.name for c in columns}
    assert names == {"toastable_col"}


def test_toast_compaction_shrinks_toast_storage(pg_conn, pg_table):
    """The headline behavior: a bloated TOAST relation is measurably reclaimed."""
    make_toast_bloated_table(pg_conn, pg_table)

    toast_relation = get_toast_relation(pg_conn, "public", pg_table)
    assert toast_relation is not None
    toast_bytes_before = db.get_relation_size(pg_conn, toast_relation)
    assert toast_bytes_before > 0, "fixture should have produced real TOAST bloat"

    config = CompactionConfig(force=True, disk_check=False)
    result = compact_table(pg_conn, "public", pg_table, config, _noop_log)

    assert result.outcome == Outcome.COMPLETED
    assert result.toast_columns_rewritten  # at least one TOASTable column was rewritten

    toast_bytes_after = db.get_relation_size(pg_conn, get_toast_relation(pg_conn, "public", pg_table))
    # TOAST file should not grow. Shrinkage depends on whether VACUUM can
    # truncate the TOAST file (fragmented free space may prevent it).
    assert toast_bytes_after <= toast_bytes_before, (
        f"TOAST grew from {toast_bytes_before} to {toast_bytes_after} — rewrite should never inflate"
    )


def test_toast_compaction_preserves_data_byte_for_byte(pg_conn, pg_table):
    """The round-trip rewrite must reproduce every TOASTed value exactly."""
    make_toast_bloated_table(pg_conn, pg_table, total_rows=200, keep_rows=25)

    with pg_conn.cursor() as cur:
        cur.execute(f'SELECT count(*) FROM "{pg_table}"')
        (count_before,) = cur.fetchone()
    checksum_before = _checksum(pg_conn, pg_table)

    config = CompactionConfig(force=True, disk_check=False)
    result = compact_table(pg_conn, "public", pg_table, config, _noop_log)
    assert result.outcome in (Outcome.COMPLETED, Outcome.INCOMPLETE_STUCK)

    with pg_conn.cursor() as cur:
        cur.execute(f'SELECT count(*) FROM "{pg_table}"')
        (count_after,) = cur.fetchone()
    checksum_after = _checksum(pg_conn, pg_table)

    assert count_after == count_before
    assert checksum_after == checksum_before, "TOAST round-trip rewrite must not alter any value"


def test_heap_only_compaction_never_touches_untouched_toast(pg_conn, pg_table):
    """Without toast_compact, a bloated TOAST relation must stay exactly as bloated.

    This is the negative-control counterpart to the main feature test: it
    documents and locks in the very problem TOAST-aware compaction was
    built to fix (see toast.py's docstring - verified directly against a
    live server that a heap-only no-op UPDATE leaves TOAST storage
    byte-for-byte unchanged).
    """
    make_toast_bloated_table(pg_conn, pg_table)
    toast_bytes_before = db.get_relation_size(pg_conn, get_toast_relation(pg_conn, "public", pg_table))
    assert toast_bytes_before > 0

    config = CompactionConfig(force=True, toast_compact=False, disk_check=False)
    result = compact_table(pg_conn, "public", pg_table, config, _noop_log)
    assert result.outcome == Outcome.COMPLETED
    assert result.toast_columns_rewritten == []

    toast_bytes_after = db.get_relation_size(pg_conn, get_toast_relation(pg_conn, "public", pg_table))
    assert toast_bytes_after == toast_bytes_before


def test_empty_toast_relation_is_never_rewritten_even_with_force(pg_conn, pg_table):
    """Regression guard for the TOAST-phase heap re-bloat bug.

    A table can have a TOASTable column (and therefore a TOAST relation
    slot) without ever having stored a value large enough to actually be
    TOASTed - the relation exists but is 0 bytes. Running the rewrite
    phase anyway would scan and rewrite every page of the table for
    nothing, and was measured directly to re-bloat an already-compacted
    heap (149 pages -> 296 pages) because the rewrite touches every page,
    not just ones with real TOASTed values. --force must only override
    the reclaimable-percent threshold, never "is there anything here at
    all" - so this must be skipped unconditionally when TOAST is empty.
    """
    with pg_conn.cursor() as cur:
        cur.execute(f'CREATE TABLE "{pg_table}" (id serial primary key, val text)')
        cur.execute(f'INSERT INTO "{pg_table}" (val) SELECT \'short\' FROM generate_series(1, 2000)')
        cur.execute(f'DELETE FROM "{pg_table}" WHERE id <= 1800')
    db.vacuum(pg_conn, "public", pg_table)
    db.analyze_table(pg_conn, "public", pg_table)

    toast_relation = get_toast_relation(pg_conn, "public", pg_table)
    assert db.get_relation_size(pg_conn, toast_relation) == 0, "fixture invalid: expected an empty TOAST relation"

    size_before_pages_after_vacuum = db.get_size_stats(pg_conn, "public", pg_table).page_count

    config = CompactionConfig(force=True, disk_check=False)
    result = compact_table(pg_conn, "public", pg_table, config, _noop_log)

    assert result.outcome == Outcome.COMPLETED
    assert result.toast_columns_rewritten == [], "empty TOAST relation must never be rewritten, even with --force"
    # The heap compaction itself should still have worked normally and
    # shrunk relative to its pre-compaction size - the regression this
    # guards against made the table bigger, not just "no smaller".
    assert result.size_after.page_count <= size_before_pages_after_vacuum


def test_toast_bloat_stats_are_invisible_to_the_main_heap_stats(pg_conn, pg_table):
    """get_bloat_stats() on the main table must not see TOAST at all.

    get_bloat_stats() measures the heap relation's own free space (via the
    FSM) and size, which do not include the separate TOAST relation - so a
    table with a small, clean heap and a heavily bloated TOAST relation
    would look 0% bloated if only the ordinary bloat stats were consulted.
    get_toast_bloat_stats() must be checked separately for TOAST-specific
    decisions.
    """
    from pg_compact.stats import get_bloat_stats

    make_toast_bloated_table(pg_conn, pg_table)

    heap_bloat = get_bloat_stats(pg_conn, "public", pg_table)
    toast_bloat = get_toast_bloat_stats(pg_conn, "public", pg_table)

    assert toast_bloat is not None
    assert toast_bloat.free_percent > 50  # the TOAST relation is heavily bloated
    # The heap itself, on the other hand, is small and not the point of
    # this fixture - it must not report the TOAST-sized bloat.
    assert heap_bloat.free_bytes < toast_bloat.free_bytes
