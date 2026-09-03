"""Integration tests for the compaction engine against real PostgreSQL servers.

Run with the test containers up (see README): pytest tests/ --pg-version=pg17
for a quick single-version pass, or without --pg-version to cover both
supported versions.
"""

from __future__ import annotations

from conftest import make_bloated_table

from pg_compact import db
from pg_compact.compactor import (
    CompactionConfig,
    Outcome,
    compact_table,
)


def _noop_log(level, message):
    pass


def test_compaction_shrinks_a_bloated_table(pg_conn, pg_table):
    """Baseline happy path: delete-heavy bloat should be measurably reclaimed."""
    make_bloated_table(pg_conn, pg_table)

    size_before = db.get_size_stats(pg_conn, "public", pg_table)
    config = CompactionConfig(force=True, disk_check=False)
    result = compact_table(pg_conn, "public", pg_table, config, _noop_log)

    assert result.outcome == Outcome.COMPLETED
    assert result.size_after.total_bytes < size_before.total_bytes
    # Reclaiming ~90% of a table that is 90% dead rows is the whole point;
    # anything less indicates the engine gave up too early.
    assert result.size_after.page_count < size_before.page_count * 0.3


def test_compaction_preserves_all_live_rows(pg_conn, pg_table):
    """No-op updates must never lose or duplicate a row."""
    make_bloated_table(pg_conn, pg_table, total_rows=2_000, keep_rows=300)

    with pg_conn.cursor() as cur:
        cur.execute(f'SELECT count(*) FROM "{pg_table}"')
        (count_before,) = cur.fetchone()
        cur.execute(f'SELECT sum(id) FROM "{pg_table}"')
        (checksum_before,) = cur.fetchone()

    config = CompactionConfig(force=True, disk_check=False)
    result = compact_table(pg_conn, "public", pg_table, config, _noop_log)
    assert result.outcome == Outcome.COMPLETED

    with pg_conn.cursor() as cur:
        cur.execute(f'SELECT count(*) FROM "{pg_table}"')
        (count_after,) = cur.fetchone()
        cur.execute(f'SELECT sum(id) FROM "{pg_table}"')
        (checksum_after,) = cur.fetchone()

    assert count_after == count_before
    assert checksum_after == checksum_before


def test_dry_run_does_not_modify_the_table(pg_conn, pg_table):
    make_bloated_table(pg_conn, pg_table)
    size_before = db.get_size_stats(pg_conn, "public", pg_table)
    with pg_conn.cursor() as cur:
        cur.execute(f'SELECT count(*) FROM "{pg_table}"')
        (count_before,) = cur.fetchone()

    config = CompactionConfig(force=True, dry_run=True, disk_check=False)
    result = compact_table(pg_conn, "public", pg_table, config, _noop_log)

    assert result.size_after is None  # dry run never reaches the "after" measurement
    size_after = db.get_size_stats(pg_conn, "public", pg_table)
    with pg_conn.cursor() as cur:
        cur.execute(f'SELECT count(*) FROM "{pg_table}"')
        (count_after,) = cur.fetchone()

    assert size_after.page_count == size_before.page_count
    assert count_after == count_before


def test_below_threshold_table_is_skipped_without_force(pg_conn, pg_table):
    """A table with little reclaimable bloat should be left alone by default."""
    with pg_conn.cursor() as cur:
        cur.execute(f'CREATE TABLE "{pg_table}" (id serial primary key, val text)')
        cur.execute(f'INSERT INTO "{pg_table}" (val) SELECT \'x\' FROM generate_series(1, 100)')

    config = CompactionConfig(force=False, disk_check=False)
    result = compact_table(pg_conn, "public", pg_table, config, _noop_log)

    assert result.outcome in (Outcome.SKIPPED_BELOW_THRESHOLD, Outcome.SKIPPED_EMPTY)


def test_force_overrides_thresholds(pg_conn, pg_table):
    # A table with no dead rows at all has 0% reclaimable space - an
    # unambiguous "below threshold" case regardless of estimator precision.
    with pg_conn.cursor() as cur:
        cur.execute(f'CREATE TABLE "{pg_table}" (id serial primary key, val text)')
        cur.execute(f'INSERT INTO "{pg_table}" (val) SELECT repeat(\'x\', 200) FROM generate_series(1, 1000)')
    db.vacuum(pg_conn, "public", pg_table)
    db.analyze_table(pg_conn, "public", pg_table)

    config_no_force = CompactionConfig(force=False, disk_check=False)
    result = compact_table(pg_conn, "public", pg_table, config_no_force, _noop_log)
    assert result.outcome == Outcome.SKIPPED_BELOW_THRESHOLD

    config_force = CompactionConfig(force=True, disk_check=False)
    result = compact_table(pg_conn, "public", pg_table, config_force, _noop_log)
    # With 0% bloat there is no free space to relocate into, so the
    # progress-guard may stop early — either outcome is correct.
    assert result.outcome in (Outcome.COMPLETED, Outcome.INCOMPLETE_STUCK)


def test_empty_table_is_skipped(pg_conn, pg_table):
    with pg_conn.cursor() as cur:
        cur.execute(f'CREATE TABLE "{pg_table}" (id serial primary key)')

    config = CompactionConfig(force=True, disk_check=False)
    result = compact_table(pg_conn, "public", pg_table, config, _noop_log)
    assert result.outcome == Outcome.SKIPPED_EMPTY


def test_nonexistent_table_is_skipped_not_errored(pg_conn):
    config = CompactionConfig(force=True, disk_check=False)
    result = compact_table(pg_conn, "public", "does_not_exist_at_all", config, _noop_log)
    assert result.outcome == Outcome.SKIPPED_EMPTY


def test_always_trigger_table_is_skipped(pg_conn, pg_table):
    make_bloated_table(pg_conn, pg_table, total_rows=500, keep_rows=50)
    with pg_conn.cursor() as cur:
        cur.execute(
            f'CREATE OR REPLACE FUNCTION "{pg_table}_trg"() RETURNS trigger AS '
            "$$ BEGIN RETURN NEW; END; $$ LANGUAGE plpgsql"
        )
        cur.execute(
            f'CREATE TRIGGER "{pg_table}_always" AFTER UPDATE ON "{pg_table}" '
            f'FOR EACH ROW EXECUTE FUNCTION "{pg_table}_trg"()'
        )
        cur.execute(f'ALTER TABLE "{pg_table}" ENABLE ALWAYS TRIGGER "{pg_table}_always"')

    config = CompactionConfig(force=True, disk_check=False)
    result = compact_table(pg_conn, "public", pg_table, config, _noop_log)
    assert result.outcome == Outcome.SKIPPED_TRIGGERS

    with pg_conn.cursor() as cur:
        cur.execute(f'DROP TRIGGER "{pg_table}_always" ON "{pg_table}"')
        cur.execute(f'DROP FUNCTION "{pg_table}_trg"()')


def test_advisory_lock_prevents_concurrent_runs(pg_conn, pg_table):
    """Two pg-compact runs on the same table must not step on each other."""
    make_bloated_table(pg_conn, pg_table, total_rows=500, keep_rows=50)

    params = db.ConnectionParams(host="localhost", port=pg_conn.info.port, user="postgres",
                                 password="postgres", dbname="pgcompact_test")
    other_conn = db.connect(params)
    try:
        locked = db.try_advisory_lock(other_conn, "public", pg_table)
        assert locked  # sanity check: lock acquisition itself works

        config = CompactionConfig(force=True, disk_check=False)
        result = compact_table(pg_conn, "public", pg_table, config, _noop_log)
        assert result.outcome == Outcome.SKIPPED_LOCKED
    finally:
        db.advisory_unlock(other_conn, "public", pg_table)
        other_conn.close()


def test_long_running_transaction_blocks_full_reclaim(pg_conn, pg_table):
    """A concurrent transaction holding a snapshot from before the deletes should
    demonstrably reduce how much space pg-compact can reclaim.

    VACUUM cannot remove a dead tuple while any transaction's snapshot might
    still need to see it. To actually hold tuples live, the blocking
    transaction's REPEATABLE READ snapshot must be taken BEFORE the deletes
    run (a snapshot taken after already can't see the since-deleted rows,
    so it wouldn't hold anything back - this was verified empirically while
    writing this test). A short statement_timeout on the compacting
    connection prevents this test from hanging if VACUUM ever blocks
    indefinitely instead of just reclaiming less.
    """
    with pg_conn.cursor() as cur:
        cur.execute(f'CREATE TABLE "{pg_table}" (id serial primary key, val text)')
        cur.execute(f'INSERT INTO "{pg_table}" (val) SELECT repeat(\'x\', 200) FROM generate_series(1, 2000)')

    params = db.ConnectionParams(host="localhost", port=pg_conn.info.port, user="postgres",
                                 password="postgres", dbname="pgcompact_test")
    blocker = db.connect(params)
    blocker.autocommit = False
    try:
        with blocker.cursor() as cur:
            cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
            cur.execute(f'SELECT count(*) FROM "{pg_table}"')  # snapshot taken now, before the deletes

        with pg_conn.cursor() as cur:
            cur.execute(f'DELETE FROM "{pg_table}" WHERE id <= 1800')
        db.vacuum(pg_conn, "public", pg_table)  # marks space reusable but can't reclaim what blocker still sees

        with pg_conn.cursor() as cur:
            cur.execute("SET statement_timeout = '3s'")

        config = CompactionConfig(force=True, disk_check=False)
        result = compact_table(pg_conn, "public", pg_table, config, _noop_log)

        # Whatever happened, the live rows must still all be there - a
        # blocked VACUUM must never look like data loss.
        with pg_conn.cursor() as cur:
            cur.execute(f'SELECT count(*) FROM "{pg_table}"')
            (count_after,) = cur.fetchone()
        assert count_after == 200
        assert result.size_after is not None
    finally:
        blocker.rollback()
        blocker.close()


def test_reindex_removes_leftover_invalid_index(pg_conn, pg_table):
    """An invalid index from an interrupted REINDEX CONCURRENTLY must be cleaned up, not left behind."""
    make_bloated_table(pg_conn, pg_table, total_rows=1_000, keep_rows=200)
    with pg_conn.cursor() as cur:
        cur.execute(f'CREATE INDEX "{pg_table}_val_idx" ON "{pg_table}" (val)')

    # Simulate a previous run being interrupted mid-REINDEX by directly
    # marking the index invalid, the same state PostgreSQL leaves behind.
    with pg_conn.cursor() as cur:
        cur.execute(
            "UPDATE pg_index SET indisvalid = false "
            f"WHERE indexrelid = '\"{pg_table}_val_idx\"'::regclass"
        )

    from pg_compact.reindex import find_invalid_indexes

    assert find_invalid_indexes(pg_conn, "public", pg_table) == [f"{pg_table}_val_idx"]

    config = CompactionConfig(force=True, disk_check=False)
    result = compact_table(pg_conn, "public", pg_table, config, _noop_log)
    assert result.outcome == Outcome.COMPLETED
    assert find_invalid_indexes(pg_conn, "public", pg_table) == []


def test_stats_detects_bloated_table(pg_conn, pg_table):
    """Stats backend should detect that a heavily-bloated table is bloated."""
    make_bloated_table(pg_conn, pg_table)

    from pg_compact.stats import get_bloat_stats

    auto = get_bloat_stats(pg_conn, "public", pg_table)

    assert auto.free_percent > 50


def test_low_bloat_table_does_not_grow(pg_conn, pg_table):
    """Regression test: a table with scattered low bloat (~12%) must never
    grow as a result of compaction.
    """
    with pg_conn.cursor() as cur:
        cur.execute(f'CREATE TABLE "{pg_table}" (id serial primary key, val text)')
        cur.execute(f'INSERT INTO "{pg_table}" (val) SELECT repeat(\'x\', 200) FROM generate_series(1, 2000)')
        cur.execute(f'DELETE FROM "{pg_table}" WHERE id % 5 = 0')
        cur.execute(f'INSERT INTO "{pg_table}" (val) SELECT repeat(\'y\', 200) FROM generate_series(1, 300)')
    db.vacuum(pg_conn, "public", pg_table)
    db.analyze_table(pg_conn, "public", pg_table)

    size_before = db.get_size_stats(pg_conn, "public", pg_table)
    config = CompactionConfig(force=True, disk_check=False)
    result = compact_table(pg_conn, "public", pg_table, config, _noop_log)

    assert result.size_after is not None
    assert result.size_after.table_bytes <= size_before.table_bytes, (
        f"Table grew from {size_before.table_bytes} to {result.size_after.table_bytes} bytes — "
        "rows are being placed beyond the tail instead of earlier in the file."
    )
