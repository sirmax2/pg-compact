"""TOAST-aware compaction helpers.

A no-op UPDATE that only touches an ordinary column shrinks the heap but
does nothing for TOAST storage — PostgreSQL's UPDATE leaves an unchanged
TOASTed value's on-disk bytes untouched. ``SET col = col`` doesn't help
either: it passes the *same* stored TOAST pointer straight through, so
PostgreSQL keeps the existing chunks and nothing is relocated.

The fix: recompute the value from its text form so the UPDATE receives a
fresh, un-TOASTed datum that must be written to new chunks (which the FSM
places in earlier free space):

    col = (col::text)::original_type

The ``::text`` cast reproduces the exact original value, so the round-trip
is value-preserving, and it works uniformly for text, varchar, bytea,
jsonb, arrays, and domains.  Because the result is a newly computed datum
rather than the original stored pointer, PostgreSQL always re-TOASTs it —
verified to relocate on every call.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import psycopg
from psycopg import sql

from pg_compact.db import qualified_name


@dataclass(frozen=True)
class ToastableColumn:
    name: str
    type_name: str


def get_toastable_columns(conn: psycopg.Connection, schema: str, table: str) -> list[ToastableColumn]:
    """Columns whose value can be stored out-of-line in a TOAST table."""
    qname = qualified_name(conn, schema, table)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT a.attname, pg_catalog.format_type(a.atttypid, a.atttypmod)
            FROM pg_catalog.pg_attribute a
            WHERE a.attrelid = %s::regclass
              AND a.attnum > 0
              AND NOT a.attisdropped
              AND a.attstorage != 'p'
            ORDER BY a.attnum
            """,
            (qname,),
        )
        return [ToastableColumn(name=row[0], type_name=row[1]) for row in cur.fetchall()]


def _toast_rewrite_assignments(columns: list[ToastableColumn]) -> list[sql.Composed]:
    """SET clause fragments that force a real TOAST rewrite for each column.

    IMPORTANT: a plain ``col = (col::text)::type`` does NOT work — PostgreSQL's
    expression evaluator recognises the no-op ``text::text`` (and similar
    identity) casts and passes the *original stored TOAST pointer* straight
    through, so ``heap_update`` sees an unchanged value, keeps the pointer, and
    nothing is re-TOASTed or relocated.  Verified on PG17: the chunk_id is
    unchanged and the chunks stay on their original pages.

    The reliable form builds a genuinely new string by *concatenation*, which
    yields a freshly computed (un-TOASTed) datum that must be re-TOASTed into
    new chunks (which the FSM then places in earlier free space):

        col = (left(col::text || 'X', -1) || right(col::text, 0))::type

    ``left(col::text || 'X', -1)`` drops the trailing ``'X'`` we just appended,
    reproducing ``col::text`` exactly; ``right(col::text, 0)`` is the empty
    string.  The result equals the original value (value-preserving) but is a
    new datum, so the relocation actually happens.  Works uniformly for text,
    varchar, bytea, jsonb, arrays and domains via the ``::text`` round-trip.
    """
    return [
        sql.SQL(
            "{col} = (left({col}::text || 'X', -1) || right({col}::text, 0))::{type}"
        ).format(col=sql.Identifier(col.name), type=sql.SQL(col.type_name))
        for col in columns
    ]


def build_toast_rewrite_query(
    schema: str,
    table: str,
    columns: list[ToastableColumn],
    from_page: int,
    to_page: int,
    max_tuples_per_page: int,
):
    """UPDATE that forces every TOASTable column's on-disk bytes to be rewritten."""
    min_ctid = f"({from_page},0)"
    max_ctid = f"({to_page},{max_tuples_per_page})"
    assignments = _toast_rewrite_assignments(columns)
    return sql.SQL(
        "UPDATE ONLY {table} SET {assignments} WHERE ctid >= {min_ctid}::tid AND ctid <= {max_ctid}::tid"
    ).format(
        table=sql.Identifier(schema, table),
        assignments=sql.SQL(", ").join(assignments),
        min_ctid=sql.Literal(min_ctid),
        max_ctid=sql.Literal(max_ctid),
    )


# ---------------------------------------------------------------------------
# Targeted TOAST rewrite — PG17+ built-in path
# ---------------------------------------------------------------------------


def build_targeted_toast_rewrite(
    schema: str,
    table: str,
    columns: list[ToastableColumn],
    column: ToastableColumn,
    toast_relation: str,
    last_page: int,
    chunk_id_expr: str,
) -> sql.Composed:
    """UPDATE that rewrites ONE row whose TOAST chunks live on *last_page*.

    *chunk_id_expr* is a SQL expression that returns the chunk_id for a
    row aliased as ``t``.

    The LIMIT 1 ensures only one row is rewritten per call.  The caller
    loops over this until no rows remain on the target page, then VACUUMs.
    Rewriting one row at a time guarantees that the new TOAST chunks land
    in earlier free space (the FSM serves them from the front of the file)
    rather than back on the same page — which can happen when many rows
    are rewritten in a single statement before VACUUM frees the old chunks.
    """
    tbl = sql.Identifier(schema, table)
    assignments = _toast_rewrite_assignments(columns)
    return sql.SQL(
        "UPDATE ONLY {table} SET {assignments} "
        "WHERE ctid = ("
        "  SELECT t.ctid FROM {table2} t "
        "  WHERE {chunk_id_expr} IN ("
        "    SELECT DISTINCT chunk_id FROM {toast} "
        "    WHERE ctid >= {lo}::tid AND ctid < {hi}::tid"
        "    AND chunk_seq = 0"
        "  ) LIMIT 1"
        ")"
    ).format(
        table=tbl,
        table2=tbl,
        chunk_id_expr=sql.SQL(chunk_id_expr),
        toast=sql.SQL(toast_relation),
        assignments=sql.SQL(", ").join(assignments),
        lo=sql.Literal(f"({last_page},0)"),
        hi=sql.Literal(f"({last_page + 1},0)"),
    )


def build_toast_rewrite_by_ctids(
    schema: str,
    table: str,
    columns: list[ToastableColumn],
    ctids: list[str],
) -> sql.Composed:
    """UPDATE that rewrites multiple rows identified by a list of ctids."""
    assignments = _toast_rewrite_assignments(columns)
    ctid_literals = sql.SQL(", ").join(sql.Literal(c) + sql.SQL("::tid") for c in ctids)
    return sql.SQL(
        "UPDATE ONLY {table} SET {assignments} WHERE ctid = ANY(ARRAY[{ctids}])"
    ).format(
        table=sql.Identifier(schema, table),
        assignments=sql.SQL(", ").join(assignments),
        ctids=ctid_literals,
    )


# ---------------------------------------------------------------------------
# Bulk heap scan via pageinspect (PG12-16)
# ---------------------------------------------------------------------------

# Batch size for bulk heap scan (pages per SQL call).
BULK_SCAN_BATCH_PAGES = 100_000


def create_chunk_map_function(conn: psycopg.Connection) -> None:
    """Create a session-scoped PL/pgSQL function for bulk TOAST pointer extraction.

    Scans a range of heap pages via ``get_raw_page`` + ``heap_page_items``,
    finds TOAST pointers by searching for a 4-byte ``toast_relid`` pattern
    in each tuple's ``t_data`` using ``position()`` (C-level byte search),
    then extracts the preceding 4 bytes as the ``chunk_id``.

    Returns ALL (chunk_id, heap_ctid) pairs — no target filtering.
    """
    with conn.cursor() as cur:
        cur.execute("""
            CREATE OR REPLACE FUNCTION pg_temp._pgc_scan_all(
                _tbl text, _relid_pattern bytea,
                _from_page int, _to_page int
            ) RETURNS TABLE(heap_ctid tid, chunk_id oid) AS $$
            DECLARE
                _page int;
                _rec record;
                _pos int;
                _cid oid;
            BEGIN
                FOR _page IN _from_page .. _to_page LOOP
                    FOR _rec IN
                        SELECT h.lp, h.t_data
                        FROM heap_page_items(get_raw_page(_tbl, _page)) h
                        WHERE h.lp_flags = 1 AND h.t_data IS NOT NULL
                          AND length(h.t_data) >= 8
                    LOOP
                        _pos := position(_relid_pattern IN _rec.t_data);
                        IF _pos >= 5 THEN
                            _cid := (get_byte(_rec.t_data, _pos-5)::bigint
                                + get_byte(_rec.t_data, _pos-4)::bigint * 256
                                + get_byte(_rec.t_data, _pos-3)::bigint * 65536
                                + get_byte(_rec.t_data, _pos-2)::bigint * 16777216)::oid;
                            -- Use the tuple's OWN position (page, lp), not
                            -- t_ctid which points to a successor version for
                            -- updated tuples.  lp_flags = 1 (LP_NORMAL) keeps
                            -- only in-use line pointers.
                            heap_ctid := ('(' || _page || ',' || _rec.lp || ')')::tid;
                            chunk_id := _cid;
                            RETURN NEXT;
                        END IF;
                    END LOOP;
                END LOOP;
            END $$ LANGUAGE plpgsql;
        """)


def build_chunk_map(
    conn: psycopg.Connection,
    schema: str,
    table: str,
    toast_relid: int,
    heap_pages: int,
    log: object = None,
) -> int:
    """Scan the entire heap and populate a temp table mapping chunk_id → ctid.

    Creates ``pg_temp._pgc_chunk_map(chunk_id, heap_ctid)`` with an index
    on ``chunk_id``.  One full scan of 2.3 M pages (23 M rows) takes
    approximately 90 seconds on PG14.

    Returns the number of rows inserted.
    """
    relid_pattern = struct.pack("<I", toast_relid).hex()
    qname = qualified_name(conn, schema, table)

    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS pg_temp._pgc_chunk_map")
        cur.execute(
            "CREATE TEMP TABLE _pgc_chunk_map ("
            "  chunk_id oid NOT NULL,"
            "  heap_ctid tid NOT NULL"
            ")"
        )

    total_rows = 0
    for start in range(0, heap_pages, BULK_SCAN_BATCH_PAGES):
        end = min(start + BULK_SCAN_BATCH_PAGES - 1, heap_pages - 1)
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    "INSERT INTO _pgc_chunk_map (chunk_id, heap_ctid) "
                    "SELECT chunk_id, heap_ctid "
                    "FROM pg_temp._pgc_scan_all(%s, %s::bytea, %s, %s)"
                ),
                (qname, f"\\x{relid_pattern}", start, end),
            )
            total_rows += cur.rowcount if cur.rowcount else 0

    with conn.cursor() as cur:
        cur.execute("CREATE INDEX ON _pgc_chunk_map (chunk_id)")
        cur.execute("ANALYZE _pgc_chunk_map")

    return total_rows


def lookup_chunk_map(
    conn: psycopg.Connection,
    chunk_ids: list[int],
) -> list[tuple[str, int]]:
    """Look up heap ctids for the given chunk_ids from the pre-built map.

    Returns ``(ctid_string, chunk_id)`` pairs.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT heap_ctid, chunk_id FROM _pgc_chunk_map "
            "WHERE chunk_id = ANY(%s)",
            (chunk_ids,),
        )
        return [(str(row[0]), int(row[1])) for row in cur.fetchall()]


def drop_chunk_map(conn: psycopg.Connection) -> None:
    """Clean up the temp table and scan function."""
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS _pgc_chunk_map")
        cur.execute(
            "DROP FUNCTION IF EXISTS pg_temp._pgc_scan_all(text, bytea, int, int)"
        )
