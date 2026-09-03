"""Tests for the multi-tier disk space guard (disk_guard.py).

The DF tier requires COPY FROM PROGRAM privileges (pg_execute_server_program
or superuser) - the test containers run as postgres (superuser), so the DF
tier is exercised directly. The WAL and UNAVAILABLE tiers are exercised by
monkeypatching the DF tier off, since there is no portable way to strip a
running superuser connection of that privilege mid-test.
"""

from __future__ import annotations

import pytest

from pg_compact import disk_guard
from pg_compact.disk_guard import DiskCheckMethod, DiskStatus, check_disk_space, wait_for_disk_space


def _noop_log(level, message):
    pass


def test_df_tier_reports_real_free_space(pg_conn):
    """Under a superuser connection, the DF tier should succeed and report a sane value."""
    status = check_disk_space(pg_conn, min_free_mb=1)
    assert status.method == DiskCheckMethod.DF
    assert status.ok is True  # asking for only 1 MB free should always pass on a real filesystem


def test_df_tier_flags_low_space_with_an_unreasonable_threshold(pg_conn):
    """Asking for an absurd amount of free space should make the DF tier report not-ok."""
    status = check_disk_space(pg_conn, min_free_mb=10_000_000_000)  # 10 PB, always unmet
    assert status.method == DiskCheckMethod.DF
    assert status.ok is False


def test_wal_tier_is_used_when_df_is_unavailable(pg_conn, monkeypatch):
    """Falling through to the WAL tier when the DF tier can't be used at all."""
    monkeypatch.setattr(disk_guard, "_try_df_check", lambda conn, min_free_mb: None)
    status = check_disk_space(pg_conn, min_free_mb=1)
    assert status.method == DiskCheckMethod.WAL
    # A freshly created test container should have no archiver failures and
    # a WAL directory nowhere near max_wal_size.
    assert status.ok is True


def test_unavailable_tier_when_nothing_works(pg_conn, monkeypatch):
    """When both tiers are unavailable, the check must report UNAVAILABLE and (per the
    agreed behavior) treat that as "ok to proceed" rather than blocking the run.
    """
    monkeypatch.setattr(disk_guard, "_try_df_check", lambda conn, min_free_mb: None)
    monkeypatch.setattr(disk_guard, "_try_wal_check", lambda conn: None)
    status = check_disk_space(pg_conn, min_free_mb=1)
    assert status.method == DiskCheckMethod.UNAVAILABLE
    assert status.ok is True


def test_wait_for_disk_space_returns_immediately_when_ok(pg_conn):
    # Should not raise or block at all - a low threshold is always satisfied.
    wait_for_disk_space(pg_conn, min_free_mb=1, log=_noop_log, poll_interval_s=0.05, timeout_s=5)


def test_wait_for_disk_space_raises_after_timeout_when_never_recovering(pg_conn, monkeypatch):
    """If the guarded condition never clears, wait_for_disk_space must give up, not hang forever."""
    always_low = DiskStatus(method=DiskCheckMethod.DF, ok=False, detail="simulated: 0 MB free")
    monkeypatch.setattr(disk_guard, "check_disk_space", lambda conn, min_free_mb: always_low)

    with pytest.raises(RuntimeError, match="did not recover"):
        wait_for_disk_space(pg_conn, min_free_mb=1024, log=_noop_log, poll_interval_s=0.05, timeout_s=0.2)


def test_wait_for_disk_space_polls_until_recovered(pg_conn, monkeypatch):
    """Recovery mid-wait must be picked up on the next poll, not just at the start."""
    calls = {"n": 0}

    def fake_check(conn, min_free_mb):
        calls["n"] += 1
        ok = calls["n"] >= 3  # "recovers" on the third check
        return DiskStatus(method=DiskCheckMethod.DF, ok=ok, detail=f"simulated call {calls['n']}")

    monkeypatch.setattr(disk_guard, "check_disk_space", fake_check)
    wait_for_disk_space(pg_conn, min_free_mb=1024, log=_noop_log, poll_interval_s=0.05, timeout_s=5)
    assert calls["n"] >= 3
