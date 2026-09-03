"""Tests for reindex.py, particularly the is_reindex_active flag used by the
CLI's Ctrl+C handler to decide whether an "invalid index may be left behind"
warning applies.
"""

from __future__ import annotations

import psycopg
import pytest

from pg_compact import reindex


def _noop_log(level, message):
    pass


@pytest.fixture(autouse=True)
def _reset_reindex_flag():
    """Guard against one test's interrupted state leaking into the next."""
    reindex.is_reindex_active = False
    yield
    reindex.is_reindex_active = False


def test_flag_is_off_before_and_after_a_normal_reindex(pg_conn, pg_table):
    with pg_conn.cursor() as cur:
        cur.execute(f'CREATE TABLE "{pg_table}" (id serial primary key, val text)')
        cur.execute(f'INSERT INTO "{pg_table}" (val) SELECT \'x\' FROM generate_series(1, 100)')

    assert reindex.is_reindex_active is False
    ok = reindex.reindex_table(pg_conn, "public", pg_table, _noop_log)
    assert ok is True
    assert reindex.is_reindex_active is False


def test_flag_is_cleared_after_a_reindex_failure(pg_conn, pg_table, monkeypatch):
    with pg_conn.cursor() as cur:
        cur.execute(f'CREATE TABLE "{pg_table}" (id serial primary key)')

    def fail(query, *args, **kwargs):
        raise psycopg.errors.InternalError_("simulated REINDEX failure")

    original_execute = psycopg.Cursor.execute

    def fake_execute(self, query, *args, **kwargs):
        text = query.as_string(self) if hasattr(query, "as_string") else str(query)
        if "REINDEX" in text:
            raise psycopg.errors.InternalError_("simulated REINDEX failure")
        return original_execute(self, query, *args, **kwargs)

    monkeypatch.setattr(psycopg.Cursor, "execute", fake_execute)
    ok = reindex.reindex_table(pg_conn, "public", pg_table, _noop_log)
    assert ok is False
    assert reindex.is_reindex_active is False


def test_flag_stays_true_when_interrupted_mid_reindex(pg_conn, pg_table, monkeypatch):
    """Regression test for the flag-reset-too-early bug.

    The original implementation cleared is_reindex_active in a try/finally
    block. Python runs `finally` during exception unwinding, before an
    outer `except KeyboardInterrupt` up the call stack ever gets a chance
    to run - so by the time the CLI's Ctrl+C handler checked the flag, it
    had already been reset to False, hiding the warning it exists to show.
    Verified directly against a live interrupt before fixing this. The
    flag must now only be cleared on the two normal-completion paths
    (success, psycopg.Error) - never while a KeyboardInterrupt is still
    propagating.
    """
    with pg_conn.cursor() as cur:
        cur.execute(f'CREATE TABLE "{pg_table}" (id serial primary key)')

    original_execute = psycopg.Cursor.execute

    def fake_execute(self, query, *args, **kwargs):
        text = query.as_string(self) if hasattr(query, "as_string") else str(query)
        if "REINDEX" in text:
            raise KeyboardInterrupt("simulated Ctrl+C")
        return original_execute(self, query, *args, **kwargs)

    monkeypatch.setattr(psycopg.Cursor, "execute", fake_execute)

    with pytest.raises(KeyboardInterrupt):
        reindex.reindex_table(pg_conn, "public", pg_table, _noop_log)

    assert reindex.is_reindex_active is True, (
        "is_reindex_active must still be True while the KeyboardInterrupt propagates, "
        "so the CLI's handler can tell this apart from an interrupt during row relocation."
    )


def test_flag_stays_false_when_interrupted_during_row_relocation(pg_conn, pg_table, monkeypatch):
    """Interrupting compact_table before REINDEX starts must leave the flag False.

    This is the CLI's basis for distinguishing "interrupted while
    rebuilding indexes" (may leave an invalid index, cleaned up
    automatically next run) from an interrupt at any other point (nothing
    unusual left behind). Simulates Ctrl+C landing during the no-op
    UPDATE loop, well before compact_table ever reaches reindex_table().
    """
    from conftest import make_bloated_table

    from pg_compact import cli
    from pg_compact.compactor import CompactionConfig, compact_table

    make_bloated_table(pg_conn, pg_table)

    def raise_interrupt(*args, **kwargs):
        raise KeyboardInterrupt("simulated Ctrl+C during row relocation")

    monkeypatch.setattr("pg_compact.compactor._compact_relation", raise_interrupt)

    config = CompactionConfig(force=True, disk_check=False)
    with pytest.raises(KeyboardInterrupt):
        compact_table(pg_conn, "public", pg_table, config, _noop_log)

    assert reindex.is_reindex_active is False

    # The CLI's interrupt handler must report the plain message, not the
    # REINDEX-specific warning, when the flag is False.
    messages = []
    cli._report_interrupt(lambda level, message: messages.append((level, message)))
    assert len(messages) == 1
    level, message = messages[0]
    assert "REINDEX" not in message and "invalid index" not in message


def test_cli_report_interrupt_warns_about_invalid_index_when_flag_is_set():
    """The CLI must show the REINDEX-specific warning only while the flag is True."""
    from pg_compact import cli

    reindex.is_reindex_active = True
    try:
        messages = []
        cli._report_interrupt(lambda level, message: messages.append((level, message)))
    finally:
        reindex.is_reindex_active = False

    assert len(messages) == 1
    level, message = messages[0]
    assert level == "warning"
    assert "invalid index" in message.lower()
