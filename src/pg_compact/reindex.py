"""Index maintenance after compaction.

REINDEX ... CONCURRENTLY (available since PostgreSQL 12) rebuilds indexes
without holding the heavy lock that plain REINDEX takes, at the cost of
needing roughly double the index's space temporarily and not being usable
inside a transaction block.

If a previous run was interrupted (Ctrl+C, connection drop, server
restart) mid-REINDEX, PostgreSQL leaves the partially built index behind
marked invalid. Such an index is inert (never used by the planner) but
still consumes disk space and update overhead, so we detect and drop it
before starting new work.
"""

from __future__ import annotations

import psycopg
from psycopg import sql

from pg_compact.db import qualified_name
from pg_compact.logging_utils import LogFn

# Set only for the duration of the actual REINDEX statement below, so the
# CLI's Ctrl+C handler can tell "interrupted while reindexing" (where an
# invalid index may be left behind, cleaned up automatically next run)
# apart from "interrupted during the row-relocation phase" (where nothing
# unusual is left behind and no extra warning is warranted).
is_reindex_active = False


def find_invalid_indexes(conn: psycopg.Connection, schema: str, table: str) -> list[str]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.relname
            FROM pg_catalog.pg_index i
            JOIN pg_catalog.pg_class c ON c.oid = i.indexrelid
            WHERE i.indrelid = %s::regclass AND NOT i.indisvalid
            """,
            (qualified_name(conn, schema, table),),
        )
        return [row[0] for row in cur.fetchall()]


def drop_invalid_indexes(conn: psycopg.Connection, schema: str, table: str, log: LogFn) -> None:
    for index_name in find_invalid_indexes(conn, schema, table):
        log("warning", f'Found invalid index "{schema}"."{index_name}" from an interrupted run; dropping it.')
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL("DROP INDEX CONCURRENTLY IF EXISTS {}").format(sql.Identifier(schema, index_name))
            )


def reindex_table(conn: psycopg.Connection, schema: str, table: str, log: LogFn) -> bool:
    """Rebuild all of the table's indexes with REINDEX TABLE CONCURRENTLY.

    Returns True on success. On failure (e.g. a uniqueness violation
    surfaced during the concurrent rebuild) logs the error and returns
    False rather than raising, since a failed reindex should not abort an
    otherwise-successful compaction run - the caller decides how to report
    it.
    """
    global is_reindex_active
    drop_invalid_indexes(conn, schema, table, log)
    is_reindex_active = True
    # Deliberately not a try/finally: if something other than a plain
    # psycopg.Error propagates out of the REINDEX statement below - most
    # importantly a KeyboardInterrupt from Ctrl+C - is_reindex_active must
    # still read True when it reaches the CLI's interrupt handler higher
    # up the stack. A finally clause here would reset it to False while
    # the exception is still unwinding, before that handler ever runs
    # (verified directly: the flag was already False by the time the
    # KeyboardInterrupt was caught one frame up). So the flag is only
    # ever cleared on the two normal-completion paths below.
    try:
        with conn.cursor() as cur:
            cur.execute(sql.SQL("REINDEX TABLE CONCURRENTLY {}").format(sql.Identifier(schema, table)))
    except psycopg.Error as exc:
        is_reindex_active = False
        log("error", f"REINDEX TABLE CONCURRENTLY failed: {exc}")
        return False
    is_reindex_active = False
    return True
