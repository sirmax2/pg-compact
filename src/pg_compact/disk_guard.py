"""Multi-tier disk space protection for the compaction loop.

The no-op UPDATEs this tool issues generate real WAL, like any other
write. If the server is running low on disk space for *any* reason - a
slow WAL archiver falling behind, an unrelated table growing, a stray log
file, a neighbor's backup running - continuing to write more WAL could
push the server into an outright crash when it runs out of space
entirely. Postgres does not roll back gracefully when the disk fills up;
it stops.

There is no single portable way to ask PostgreSQL "how much free disk
space is left", so this uses a tiered fallback, each tier used only if
the previous one is unavailable due to privileges:

1. ``COPY ... FROM PROGRAM`` running ``df`` on the server's data
   directory. This is the only way to get the *real* free space on the
   actual filesystem, covering every possible cause of a full disk, not
   just WAL growth. It requires the ``pg_execute_server_program`` role
   (or superuser) - verified directly: an ordinary role gets
   "permission denied to COPY to or from an external program".
2. ``pg_stat_archiver`` plus ``max_wal_size``: no special privileges
   required at all (verified directly - an ordinary role can read
   pg_stat_archiver and SHOW max_wal_size). This can't see the real disk,
   but catches the specific "archiving can't keep up" scenario this
   feature was originally requested for, by comparing failed archive
   attempts and, when available, the WAL directory's own size (see
   pg_ls_waldir, which does need pg_monitor-style privileges and is used
   opportunistically if it happens to work).
3. Neither works: log a one-time warning and proceed without this
   protection, exactly as agreed - some protection some of the time is
   better than refusing to run at all when nothing more is available.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable

import psycopg
from psycopg import sql as pg_sql

from pg_compact.db import _row
from pg_compact.logging_utils import LogFn


class DiskCheckMethod(str, Enum):
    DF = "df"  # real filesystem free space via COPY FROM PROGRAM
    WAL = "wal"  # indirect: WAL directory size vs max_wal_size, plus archiver failures
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class DiskStatus:
    method: DiskCheckMethod
    ok: bool  # True if there is no reason to pause
    detail: str  # human-readable summary for logging
    free_mb: float | None = None  # actual free space in MB when known (DF tier)


def _try_df_check(conn: psycopg.Connection, min_free_mb: int) -> DiskStatus | None:
    """Real free space on the filesystem holding the data directory, via df.

    Returns None (not DiskStatus) if this method is unavailable so the
    caller can fall through to the next tier - as opposed to returning a
    DiskStatus, which would mean the check ran successfully.
    """
    try:
        # ON COMMIT DROP only makes sense across statements sharing one
        # transaction. On an autocommit connection (which pg-compact
        # always uses - see db.connect), each statement commits on its
        # own, so without an explicit transaction block the temp table
        # would be dropped again immediately after CREATE, before the
        # subsequent SELECT ever ran (verified directly: the very next
        # statement failed with "relation ... does not exist"). Wrapping
        # the whole sequence in one transaction keeps the temp table
        # alive until we're done reading it.
        with conn.transaction(), conn.cursor() as cur:
            cur.execute("SHOW data_directory")
            (data_dir,) = _row(cur)
            cur.execute("CREATE TEMP TABLE pg_compact_df_check(info text) ON COMMIT DROP")
            # -Pk: POSIX output format, sizes in 1024-byte blocks - a
            # stable, script-friendly format across Linux and BSD/macOS df
            # implementations, unlike the default human-readable -h output.
            # COPY FROM PROGRAM requires a literal, not a bound parameter,
            # for the command string - same restriction as SET. The value
            # embedded here is the server's own data_directory as reported
            # by Postgres itself, not user input, so building it into the
            # command via sql.Literal (proper SQL-string escaping) is safe.
            cur.execute(
                pg_sql.SQL("COPY pg_compact_df_check FROM PROGRAM {}").format(
                    pg_sql.Literal(f"df -Pk {_shell_quote(data_dir)}")
                )
            )
            cur.execute("SELECT info FROM pg_compact_df_check")
            rows = cur.fetchall()
    except psycopg.Error:
        return None

    # POSIX df output is two lines: a header, then the data row we want.
    # "Filesystem 1024-blocks Used Available Capacity Mounted-on"
    data_row = rows[-1][0] if rows else None
    if data_row is None:
        return None
    available_kb = _parse_df_available_kb(data_row)
    if available_kb is None:
        return None

    available_mb = available_kb / 1024
    ok = available_mb >= min_free_mb
    detail = f"{available_mb:.0f} MB free on the data directory's filesystem (df)"
    return DiskStatus(method=DiskCheckMethod.DF, ok=ok, detail=detail, free_mb=available_mb)


def _parse_df_available_kb(data_row: str) -> int | None:
    # Available is always the 4th whitespace-separated field in POSIX -Pk
    # output: Filesystem, 1024-blocks, Used, Available, Capacity, Mounted-on.
    # (Mount points containing spaces would shift later fields, but never
    # this one, since it comes before them.)
    fields = data_row.split()
    if len(fields) >= 4 and fields[3].isdigit():
        return int(fields[3])
    return None


def _shell_quote(value: str) -> str:
    # Minimal single-quote escaping for a path that COPY FROM PROGRAM will
    # hand to /bin/sh -c. Data directory paths are server-controlled, not
    # user input, but quoting defensively costs nothing.
    return "'" + value.replace("'", "'\\''") + "'"


def _try_wal_check(conn: psycopg.Connection) -> DiskStatus | None:
    """Indirect signal: is WAL accumulating beyond max_wal_size, or is archiving failing outright?

    No special privileges are required for pg_stat_archiver or SHOW
    max_wal_size (verified directly). pg_ls_waldir() is attempted too,
    opportunistically, since it gives an exact WAL directory size when
    available (pg_monitor role or superuser) - but its absence does not
    disqualify this tier, since the archiver-failure signal alone is
    still useful.
    """
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT failed_count, last_failed_wal FROM pg_stat_archiver")
            row = cur.fetchone()
            failed_count, last_failed_wal = (row if row else (0, None))
            cur.execute("SHOW max_wal_size")
            (max_wal_size_str,) = _row(cur)
            cur.execute("SELECT pg_size_bytes(%s)", (max_wal_size_str,))
            (max_wal_bytes,) = _row(cur)
    except psycopg.Error:
        return None

    wal_bytes = None
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT coalesce(sum(size), 0) FROM pg_ls_waldir()")
            (wal_bytes,) = _row(cur)
    except psycopg.Error:
        pass  # pg_ls_waldir needs pg_monitor-style privileges; optional

    if failed_count and failed_count > 0:
        return DiskStatus(
            method=DiskCheckMethod.WAL,
            ok=False,
            detail=f"WAL archiving has failed {failed_count} time(s), most recently for {last_failed_wal}",
        )

    if wal_bytes is not None and max_wal_bytes:
        # A generous multiple of max_wal_size: some transient overshoot is
        # normal (e.g. during a checkpoint), so only treat this as a
        # problem once WAL has clearly outgrown its configured budget.
        threshold = max_wal_bytes * 2
        ok = wal_bytes < threshold
        detail = (
            f"WAL directory is {wal_bytes / (1024 * 1024):.0f} MB "
            f"(max_wal_size is {max_wal_bytes / (1024 * 1024):.0f} MB)"
        )
        return DiskStatus(method=DiskCheckMethod.WAL, ok=ok, detail=detail)

    # No pg_ls_waldir access and no archiver failures observed: nothing
    # more to go on with this tier, but not an outright failure either.
    return DiskStatus(method=DiskCheckMethod.WAL, ok=True, detail="no WAL archiving problems detected")


def check_disk_space(conn: psycopg.Connection, min_free_mb: int) -> DiskStatus:
    """Run the best available disk-space check, falling back through tiers."""
    result = _try_df_check(conn, min_free_mb)
    if result is not None:
        return result
    result = _try_wal_check(conn)
    if result is not None:
        return result
    return DiskStatus(
        method=DiskCheckMethod.UNAVAILABLE,
        ok=True,  # nothing to act on - proceed, as agreed, rather than refuse to run
        detail="disk space could not be checked (no usable privilege for df or WAL monitoring)",
    )


_unavailable_warning_shown = False


WaitCallback = Callable[[], None]


def wait_for_disk_space(
    conn: psycopg.Connection,
    min_free_mb: int,
    log: LogFn,
    poll_interval_s: float = 10.0,
    timeout_s: float = 1800.0,
    on_wait: WaitCallback | None = None,
) -> None:
    """Block until disk space looks healthy, or raise after timeout_s.

    Called once per compaction round. Cheap on the DF and WAL tiers (a
    handful of catalog/system queries), so checking every round is not a
    meaningful overhead next to the round's own UPDATE/VACUUM work.
    """
    global _unavailable_warning_shown
    status = check_disk_space(conn, min_free_mb)

    if status.method == DiskCheckMethod.UNAVAILABLE:
        if not _unavailable_warning_shown:
            _unavailable_warning_shown = True
            log(
                "warning",
                "Could not check disk space (no permission for df or WAL monitoring); "
                "proceeding without this protection. Grant pg_execute_server_program "
                "for the most reliable check.",
            )
        return

    if status.ok:
        return

    log("warning", f"Pausing before continuing: {status.detail}.")
    start = time.monotonic()
    while not status.ok:
        if time.monotonic() - start > timeout_s:
            raise RuntimeError(
                f"Disk space did not recover within {timeout_s:.0f} seconds ({status.detail}). Stopping."
            )
        if on_wait is not None:
            on_wait()
        time.sleep(poll_interval_s)
        status = check_disk_space(conn, min_free_mb)
        if status.free_mb is not None:
            pass  # caller can read updated free_mb via the next progress update
        if not status.ok:
            log("warning", f"Still waiting: {status.detail}.")
    log("info", f"Disk space recovered: {status.detail}.")
