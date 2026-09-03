"""Low level PostgreSQL helpers built on top of psycopg.

All identifier interpolation goes through ``psycopg.sql`` so table/schema/
column names are always safely quoted, never string-concatenated.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import psycopg
from psycopg import sql

# REINDEX ... CONCURRENTLY on tables/schemas requires PostgreSQL 12+.
MIN_SERVER_VERSION = 120000

# PG17 (170000) introduced pg_column_toast_chunk_id() as a built-in.
PG17_VERSION = 170000


class UnsupportedServerVersionError(RuntimeError):
    """Raised when the target server is older than MIN_SERVER_VERSION."""


class TableNotFoundError(RuntimeError):
    """Raised when the requested schema.table does not exist or is not a table."""


class MissingExtensionError(RuntimeError):
    """Raised when a required PostgreSQL extension is not installed."""


@dataclass(frozen=True)
class ConnectionParams:
    host: str | None = None
    port: int | None = None
    user: str | None = None
    password: str | None = None
    dbname: str | None = None

    @classmethod
    def from_args_and_env(
        cls,
        host: str | None,
        port: int | None,
        user: str | None,
        password: str | None,
        dbname: str | None,
    ) -> ConnectionParams:
        """Fill in gaps from the standard libpq PG* environment variables.

        This mirrors the behaviour of psql and every other libpq-based
        client, so users can rely on PGHOST/PGPORT/PGUSER/PGPASSWORD/
        PGDATABASE (and ~/.pgpass) instead of passing everything on the
        command line.
        """
        env_port = os.environ.get("PGPORT")
        return cls(
            host=host or os.environ.get("PGHOST"),
            port=port or (int(env_port) if env_port else None),
            user=user or os.environ.get("PGUSER"),
            password=password or os.environ.get("PGPASSWORD"),
            dbname=dbname or os.environ.get("PGDATABASE"),
        )

    def conninfo(self) -> str:
        kwargs: dict[str, str] = {}
        if self.host:
            kwargs["host"] = self.host
        if self.port:
            kwargs["port"] = str(self.port)
        if self.user:
            kwargs["user"] = self.user
        if self.password:
            kwargs["password"] = self.password
        if self.dbname:
            kwargs["dbname"] = self.dbname
        return psycopg.conninfo.make_conninfo(**kwargs)


def connect(params: ConnectionParams) -> psycopg.Connection:
    """Open an autocommit connection.

    Autocommit is required: VACUUM and REINDEX CONCURRENTLY cannot run
    inside a transaction block. The compaction loop opens its own short
    explicit transactions per batch where needed.
    """
    conn = psycopg.connect(params.conninfo(), autocommit=True)
    with conn.cursor() as cur:
        cur.execute("SET client_min_messages TO WARNING")
        cur.execute("SET application_name TO 'pg-compact'")
    return conn


def qualified_name(conn: psycopg.Connection, schema: str, table: str) -> str:
    """Return a safely quoted "schema"."table" string usable as a ::regclass literal."""
    return sql.Identifier(schema, table).as_string(conn)


def _row(cur: psycopg.Cursor) -> tuple:  # type: ignore[type-arg]
    """fetchone() that asserts the row exists — for queries guaranteed to return one."""
    row = cur.fetchone()
    assert row is not None, "expected exactly one row"
    return row


def get_server_version(conn: psycopg.Connection) -> int:
    with conn.cursor() as cur:
        cur.execute("SHOW server_version_num")
        (version,) = _row(cur)
        return int(version)


def check_min_version(conn: psycopg.Connection) -> None:
    version = get_server_version(conn)
    if version < MIN_SERVER_VERSION:
        raise UnsupportedServerVersionError(
            f"pg-compact requires PostgreSQL 12 or newer (found server_version_num={version}). "
            "REINDEX ... CONCURRENTLY on tables is not available on older versions."
        )


# Extensions that must be installed before pg-compact can run.
# pageinspect is only needed on PG12-16 (PG17+ has pg_column_toast_chunk_id).
_ALWAYS_REQUIRED = ("pg_freespacemap",)
_PRE_PG17_REQUIRED = ("pageinspect",)


def check_required_extensions(conn: psycopg.Connection) -> None:
    """Verify that all required PostgreSQL extensions are installed.

    Raises :class:`MissingExtensionError` listing every extension that is
    not yet present.  The caller (CLI entry-point) should run this once
    before starting any real work.
    """
    required = list(_ALWAYS_REQUIRED)
    if get_server_version(conn) < PG17_VERSION:
        required.extend(_PRE_PG17_REQUIRED)

    with conn.cursor() as cur:
        cur.execute(
            "SELECT extname FROM pg_catalog.pg_extension WHERE extname = ANY(%s)",
            (required,),
        )
        installed = {row[0] for row in cur.fetchall()}

    missing = [ext for ext in required if ext not in installed]
    if missing:
        cmds = "; ".join(f"CREATE EXTENSION {ext}" for ext in missing)
        raise MissingExtensionError(
            f"Required PostgreSQL extensions are not installed: {', '.join(missing)}. "
            f"Install them with: {cmds}"
        )


def table_exists(conn: psycopg.Connection, schema: str, table: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT EXISTS (
                SELECT 1 FROM pg_catalog.pg_class c
                JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
                WHERE n.nspname = %s AND c.relname = %s AND c.relkind IN ('r', 'p')
            )
            """,
            (schema, table),
        )
        (exists,) = _row(cur)
        return bool(exists)


def try_advisory_lock(conn: psycopg.Connection, schema: str, table: str) -> bool:
    """Best-effort mutual exclusion so two pg-compact runs don't fight over one table.

    Uses the classic two-int advisory lock keyed on pg_class's oid plus the
    target table's oid, matching the convention used by pgcompacttable.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT pg_try_advisory_lock(
                'pg_catalog.pg_class'::regclass::integer,
                %s::regclass::integer
            )
            """,
            (qualified_name(conn, schema, table),),
        )
        (locked,) = _row(cur)
        return bool(locked)


def advisory_unlock(conn: psycopg.Connection, schema: str, table: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT pg_advisory_unlock(
                'pg_catalog.pg_class'::regclass::integer,
                %s::regclass::integer
            )
            """,
            (qualified_name(conn, schema, table),),
        )


def has_blocking_triggers(conn: psycopg.Connection, schema: str, table: str) -> bool:
    """True if the table has ENABLE ALWAYS or ENABLE REPLICA triggers on UPDATE.

    Those fire even while session_replication_role = replica, so running our
    no-op UPDATE would trigger arbitrary user logic. We refuse to touch such
    tables instead of surprising the user.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT count(*) > 0
            FROM pg_catalog.pg_trigger
            WHERE tgrelid = %s::regclass
              AND tgenabled IN ('A', 'R')
              AND (tgtype & 16) != 0
            """,
            (qualified_name(conn, schema, table),),
        )
        (has_triggers,) = _row(cur)
        return bool(has_triggers)


def set_replica_role(conn: psycopg.Connection) -> None:
    """Prevent ORIGIN-role triggers (incl. FK enforcement triggers) from firing.

    Note: this also affects logical replication - changes made while this is
    set are not replicated to logical subscribers using the default filter.
    Since our UPDATEs are no-ops (column = itself), there is no data to
    replicate anyway.
    """
    with conn.cursor() as cur:
        cur.execute("SET session_replication_role TO 'replica'")


def reset_replication_role(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute("RESET session_replication_role")


def set_lock_timeout(conn: psycopg.Connection, timeout_ms: int) -> None:
    """Cap how long our own statements wait for a row/page lock held by someone else.

    Without this, a no-op UPDATE contending with a real application query
    for the same rows would queue up behind it for as long as Postgres's
    default (no timeout at all), holding up pg-compact's own progress and
    potentially making the application's transaction wait on us in turn
    once it tries to reacquire the same lock. A bounded lock_timeout turns
    that into "back off and retry the batch shortly" instead of an
    indefinite stall that could just as easily block real traffic.
    """
    # SET does not accept a bound parameter for its value; timeout_ms is
    # an int from our own config (never user-controlled SQL text), so
    # formatting it directly into the statement is safe.
    with conn.cursor() as cur:
        cur.execute(sql.SQL("SET lock_timeout = {}").format(sql.Literal(f"{int(timeout_ms)}ms")))


def vacuum(conn: psycopg.Connection, schema: str, table: str, analyze: bool = False) -> None:
    server_version = get_server_version(conn)
    options = []
    if analyze:
        options.append("ANALYZE")
    if server_version >= 120000:
        options.append("INDEX_CLEANUP ON")
    ident = sql.Identifier(schema, table)
    with conn.cursor() as cur:
        if options:
            cur.execute(sql.SQL("VACUUM ({}) {}").format(sql.SQL(", ".join(options)), ident))
        else:
            cur.execute(sql.SQL("VACUUM {}").format(ident))


def analyze_table(conn: psycopg.Connection, schema: str, table: str) -> None:
    ident = sql.Identifier(schema, table)
    with conn.cursor() as cur:
        cur.execute(sql.SQL("ANALYZE {}").format(ident))


@dataclass(frozen=True)
class SizeStats:
    table_bytes: int
    toast_bytes: int
    indexes_bytes: int
    total_bytes: int
    page_count: int
    total_page_count: int


def get_relation_size(conn: psycopg.Connection, relation: str | None) -> int:
    """Physical size of an arbitrary relation by its already-qualified name, or 0 if None."""
    if relation is None:
        return 0
    with conn.cursor() as cur:
        cur.execute("SELECT pg_catalog.pg_relation_size(%s::regclass)", (relation,))
        (size,) = _row(cur)
        return int(size)


def get_size_stats(conn: psycopg.Connection, schema: str, table: str) -> SizeStats:
    qname = qualified_name(conn, schema, table)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                heap_size,
                toast_size,
                index_size,
                total_size,
                ceil(heap_size::numeric / bs)::bigint AS page_count,
                ceil(total_size::numeric / bs)::bigint AS total_page_count
            FROM (
                SELECT
                    current_setting('block_size')::integer AS bs,
                    pg_catalog.pg_relation_size(%s::regclass) AS heap_size,
                    coalesce(pg_catalog.pg_relation_size(
                        (SELECT reltoastrelid FROM pg_catalog.pg_class WHERE oid = %s::regclass)
                    ), 0) AS toast_size,
                    pg_catalog.pg_indexes_size(%s::regclass) AS index_size,
                    pg_catalog.pg_total_relation_size(%s::regclass) AS total_size
            ) sq
            """,
            (qname, qname, qname, qname),
        )
        row = _row(cur)
        return SizeStats(
            table_bytes=row[0],
            toast_bytes=row[1],
            indexes_bytes=row[2],
            total_bytes=row[3],
            page_count=row[4],
            total_page_count=row[5],
        )


def avg_row_size(conn: psycopg.Connection, schema: str, table: str) -> int | None:
    """Estimated average row size from catalog statistics (no table scan).

    Uses pg_stats.avg_width per column (updated by ANALYZE) plus the
    fixed tuple header overhead. Returns None if stats are missing.
    """
    qname = qualified_name(conn, schema, table)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 24 + coalesce(sum(s.avg_width), 0)::int
            FROM pg_catalog.pg_attribute a
            JOIN pg_catalog.pg_stats s
              ON s.schemaname = %s AND s.tablename = %s
              AND s.attname = a.attname AND s.inherited = false
            WHERE a.attrelid = %s::regclass
              AND a.attnum > 0
              AND NOT a.attisdropped
            """,
            (schema, table, qname),
        )
        row = cur.fetchone()
        return row[0] if row and row[0] and row[0] > 24 else None


def usable_free_pages(conn: psycopg.Connection, schema: str, table: str, min_avail: int) -> int:
    """Count pages with at least min_avail bytes free, via pg_freespacemap."""
    return usable_free_pages_for_relation(conn, qualified_name(conn, schema, table), min_avail)


def usable_free_pages_for_relation(conn: psycopg.Connection, relation: str, min_avail: int) -> int:
    """Count pages with at least min_avail bytes free for an arbitrary relation name."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM pg_freespace(%s::regclass) WHERE avail >= %s",
            (relation, min_avail),
        )
        (count,) = _row(cur)
        return int(count)


def landing_capacity(conn: psycopg.Connection, schema: str, table: str, row_size: int) -> int:
    """Estimate how many rows can be absorbed by pages with enough free space.

    For every page in the FSM that has at least *row_size* bytes available,
    calculates ``floor(avail / row_size)`` and sums the results.  This tells
    us the total number of tuples that can be relocated into existing free
    space without extending the table.
    """
    return landing_capacity_for_relation(
        conn, qualified_name(conn, schema, table), row_size,
    )


def landing_capacity_for_relation(
    conn: psycopg.Connection, relation: str, row_size: int,
) -> int:
    """landing_capacity for an arbitrary relation name."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT coalesce(sum(floor(avail / %s)::bigint), 0) "
            "FROM pg_freespace(%s::regclass) WHERE avail >= %s",
            (row_size, relation, row_size),
        )
        (count,) = _row(cur)
        return int(count)


def pick_update_column(conn: psycopg.Connection, schema: str, table: str) -> str | None:
    """Pick the cheapest column to use for the no-op UPDATE col = col.

    Preference order: fixed-length (not TOAST-able) over variable-length,
    columns not covered by any index, then smallest storage size. This
    keeps the no-op update as cheap as possible and avoids churning TOAST
    storage or indexes unnecessarily.
    """
    qname = qualified_name(conn, schema, table)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT attname
            FROM pg_catalog.pg_attribute
            WHERE attnum > 0
              AND NOT attisdropped
              AND attrelid = %s::regclass
            ORDER BY
                (attlen = -1),
                (
                    attnum::text IN (
                        SELECT regexp_split_to_table(indkey::text, ' ')
                        FROM pg_catalog.pg_index
                        WHERE indrelid = %s::regclass
                    )
                ),
                attlen,
                attnum
            LIMIT 1
            """,
            (qname, qname),
        )
        row = cur.fetchone()
        return row[0] if row else None


def pick_indexed_column(conn: psycopg.Connection, schema: str, table: str) -> str | None:
    """Pick the cheapest *indexed* column for UPDATE col = col.

    When free space is fragmented so that per-page waste is below the tuple
    size, every page already has ``tuples_per_page`` live rows but not enough
    room for one more.  A non-indexed UPDATE would be a HOT update — the new
    row version stays on the same page, defeating relocation.

    Updating an indexed column forces a non-HOT update: PostgreSQL must
    allocate the new tuple on a page with enough free space (via FSM),
    effectively relocating the row.  The trade-off is extra index
    maintenance, but it is the only way to compact such tables.

    Preference: smallest fixed-length indexed column first.
    """
    qname = qualified_name(conn, schema, table)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT a.attname
            FROM pg_catalog.pg_attribute a
            WHERE a.attnum > 0
              AND NOT a.attisdropped
              AND a.attrelid = %s::regclass
              AND a.attnum::text IN (
                  SELECT regexp_split_to_table(indkey::text, ' ')
                  FROM pg_catalog.pg_index
                  WHERE indrelid = %s::regclass
              )
            ORDER BY
                (a.attlen = -1),
                a.attlen,
                a.attnum
            LIMIT 1
            """,
            (qname, qname),
        )
        row = cur.fetchone()
        return row[0] if row else None


def get_relation_page_count(conn: psycopg.Connection, relation: str | None) -> int:
    """Page count of an arbitrary relation by its already-qualified name, or 0 if None.

    Uses pg_relation_size (which reads the physical file length) rather than
    relpages from pg_class, because relpages may be stale — VACUUM updates
    the physical file via truncation but doesn't always refresh pg_class
    immediately.
    """
    if relation is None:
        return 0
    with conn.cursor() as cur:
        cur.execute(
            "SELECT ceil(pg_catalog.pg_relation_size(%s::regclass)::numeric "
            "/ current_setting('block_size')::integer)::bigint",
            (relation,),
        )
        (pages,) = _row(cur)
        return int(pages) if pages else 0


def vacuum_relation(conn: psycopg.Connection, relation: str) -> None:
    """VACUUM an arbitrary relation by its already-qualified name (e.g. a TOAST table).

    Uses ``INDEX_CLEANUP ON`` to force cleanup of dead line pointers.
    Without this, PG13+ may skip index cleanup when dead item count is
    low, leaving dead LP that prevent page truncation.
    """
    with conn.cursor() as cur:
        cur.execute(sql.SQL("VACUUM (INDEX_CLEANUP ON) {}").format(sql.SQL(relation)))


def relation_free_space_bytes(conn: psycopg.Connection, relation: str) -> int:
    """Total available free space in the relation according to the FSM."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT coalesce(sum(avail), 0) FROM pg_freespace(%s::regclass)",
            (relation,),
        )
        (total,) = _row(cur)
        return int(total)


# On-page size of a full TOAST chunk: TOAST_MAX_CHUNK_SIZE (~1996 B) plus
# the heap tuple header and item pointer.  A page whose free space is below
# this cannot accept a relocated chunk, so that free space is structurally
# unreclaimable online (analogous to heap per-page alignment waste).
TOAST_CHUNK_ONPAGE_SIZE = 2048


def relation_free_space_split(
    conn: psycopg.Connection, relation: str, chunk_size: int = TOAST_CHUNK_ONPAGE_SIZE
) -> tuple[int, int]:
    """Split a relation's FSM free space into (usable, unusable) bytes.

    *usable* is free space on pages with at least *chunk_size* bytes free —
    space that can actually hold a relocated TOAST chunk.  *unusable* is the
    sum of sub-chunk holes: free space that no whole chunk can fill, so it
    cannot be reclaimed by online relocation (only VACUUM FULL / pg_repack
    repacks it away).  One FSM scan computes both.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT "
            "coalesce(sum(avail) FILTER (WHERE avail >= %s), 0), "
            "coalesce(sum(avail) FILTER (WHERE avail > 0 AND avail < %s), 0) "
            "FROM pg_freespace(%s::regclass)",
            (chunk_size, chunk_size, relation),
        )
        usable, unusable = _row(cur)
        return int(usable), int(unusable)


def relation_free_pages_before(
    conn: psycopg.Connection, relation: str, before_page: int, min_avail: int = 2000
) -> int:
    """Count pages with >= min_avail free bytes that lie before *before_page*.

    This is the "landing zone" for rows relocated out of the tail — the
    number of earlier pages that can absorb new tuples/chunks.  Used to
    size the tail window so relocated rows land ahead of it rather than
    back inside it.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM pg_freespace(%s::regclass) "
            "WHERE blkno < %s AND avail >= %s",
            (relation, before_page, min_avail),
        )
        (count,) = _row(cur)
        return int(count)


def get_toast_chunk_ids_on_pages(
    conn: psycopg.Connection,
    toast_relation: str,
    from_page: int,
    to_page: int,
) -> list[int]:
    """Return distinct chunk_id values whose chunks live on the given TOAST pages.

    Each chunk_id uniquely identifies one TOASTed value (one column of one
    heap row).  The caller uses these to build an UPDATE that targets only
    the heap rows whose TOAST storage occupies the specified page range.
    """
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                "SELECT DISTINCT chunk_id FROM {toast} "
                "WHERE ctid >= {min}::tid AND ctid <= {max}::tid"
            ).format(
                toast=sql.SQL(toast_relation),
                min=sql.Literal(f"({from_page},0)"),
                max=sql.Literal(f"({to_page},65535)"),
            ),
        )
        return [row[0] for row in cur.fetchall()]


def has_toast_chunk_id_func(conn: psycopg.Connection) -> bool:
    """True if pg_column_toast_chunk_id() is available (PG17+)."""
    return get_server_version(conn) >= PG17_VERSION
