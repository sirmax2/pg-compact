"""Bloat estimation via pg_freespacemap.

The relation size and ``sum(avail)`` from ``pg_freespace()`` are enough to
split a relation the same way for the heap and its TOAST storage:

* **occupied** = ``size - free`` — data plus unavoidable per-page overhead.
* **reclaimable** = the pages above the floor a full rewrite could reach
  (``ceil(occupied / usable_per_page)``); what ``VACUUM FULL`` / ``pg_repack``
  would free.
* **alignment waste** = the free space that still would not pack away —
  structural per-page padding the FSM reports as "free" but that no tool
  can reclaim.

Fast (~100 ms plus one FSM scan) and needs no per-column statistics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import psycopg

from pg_compact.db import (
    _row,
    qualified_name,
    relation_free_space_bytes,
)
from pg_compact.fsm_predict import (
    FULL_TOAST_CHUNK_FOOTPRINT,
    admit_threshold,
    onpage_footprint,
)


def _online_reclaimable_bytes(
    conn: psycopg.Connection, relation: str, unit_footprint: int
) -> int:
    """FSM free space that online relocation can actually target, in one scan.

    The FSM only offers a page for a relocation of ``unit_footprint`` bytes
    when its free space reaches the rounded-up category threshold (see
    :func:`~pg_compact.fsm_predict.admit_threshold`).  Free space in holes
    below that is trapped (sub-tuple / sub-chunk fragmentation) and only a
    full rewrite can pack it away.  Summing ``avail`` over pages at or above
    the threshold is the space pg-compact can relocate into.

    Accuracy (measured): because ``fsm_search`` serves the earliest fitting
    page, relocation pulls chunks toward the front and truncates the tail even
    when whole-chunk-sized holes are spread uniformly through the file — there
    this estimate matches the real online result within a few percent
    (e.g. predicted 80 MB, actual 73 MB on a uniformly fragmented relation).
    It is an UPPER bound: where the free space is dominated by partial
    (sub-chunk) holes, or already lies ahead of all live data, the real online
    result can be lower.  A cheap single FSM scan, no pageinspect sampling.
    """
    threshold = admit_threshold(unit_footprint)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT coalesce(sum(avail), 0) FROM pg_freespace(%s::regclass) "
            "WHERE avail >= %s",
            (relation, threshold),
        )
        (total,) = _row(cur)
    return int(total)


@dataclass(frozen=True)
class BloatStats:
    free_percent: float
    free_bytes: int
    # reclaimable_* is the total genuine bloat a full rewrite would free
    # (the VACUUM FULL / pg_repack floor).  alignment_waste_bytes is the
    # free space that would not pack away even by a rewrite (structural
    # per-page padding).
    reclaimable_percent: float
    reclaimable_bytes: int
    alignment_waste_bytes: int
    # Split of the reclaimable bloat by WHO can return it, derived from the
    # FSM placement rule (see fsm_predict):
    #   * online_reclaimable_bytes — what pg-compact can free in place, by
    #     relocating rows/chunks into holes the FSM will actually offer
    #     (avail >= the relocation unit's rounded-up size) and truncating the
    #     tail.
    #   * vacuum_full_only_bytes — the rest of the reclaimable bloat: free
    #     space trapped in sub-tuple / sub-chunk holes the FSM never offers,
    #     so only a full rewrite (VACUUM FULL / CLUSTER / pg_repack) returns it.
    online_reclaimable_bytes: int = 0
    vacuum_full_only_bytes: int = 0


def _block_size(conn: psycopg.Connection) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT current_setting('block_size')::integer")
        (bs,) = _row(cur)
        return int(bs)


def _fillfactor(conn: psycopg.Connection, schema: str, table: str) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT coalesce(
                (regexp_match(array_to_string(reloptions, ' '), 'fillfactor=(\\d+)'))[1]::int,
                100
            )
            FROM pg_catalog.pg_class c
            JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = %s AND c.relname = %s
            """,
            (schema, table),
        )
        (fillfactor,) = _row(cur)
        return int(fillfactor)


_PAGE_HEADER_BYTES = 24


def _bloat_from_occupied(
    relation_bytes: int,
    free_bytes: int,
    block_size: int,
    fillfactor: int = 100,
    online_reclaimable_bytes: int = 0,
) -> BloatStats:
    """Split a relation's size into live data / alignment waste / reclaimable.

    Uses one cheap identity: the *occupied* bytes are ``relation_bytes -
    free_bytes`` (whatever the FSM does not report as free is holding data
    plus its unavoidable per-page overhead).  Packing that occupied volume
    as tightly as a full rewrite would — ``ceil(occupied / usable_per_page)``
    pages — gives the floor a ``VACUUM FULL`` / ``pg_repack`` could reach:

      * reclaimable   = pages above that floor  (what a rewrite frees)
      * alignment waste = the free space that still would not pack away
                          (structural per-page padding; irreducible even by
                          a full rewrite)
      * live data     = the floor itself (implicit: relation - free - waste...
                        i.e. ideal_pages worth)

    ``online_reclaimable_bytes`` (computed by the caller from the FSM
    threshold split) is clamped into the reclaimable range and the remainder
    of reclaimable becomes ``vacuum_full_only_bytes``.

    The same formula serves the heap and the TOAST relation, so both report
    the identical breakdown.  Validated against an exact sum-of-on-page-sizes
    rewrite floor on a 40 GB TOAST relation: within ~1% (and conservative —
    it never over-states reclaimable).
    """
    if relation_bytes <= 0:
        return BloatStats(0.0, 0, 0.0, 0, 0, 0, 0)

    free_percent = 100.0 * free_bytes / relation_bytes
    occupied = max(relation_bytes - free_bytes, 0)
    usable_per_page = max(int((block_size - _PAGE_HEADER_BYTES) * fillfactor / 100.0), 1)
    ideal_pages = math.ceil(occupied / usable_per_page)
    relpages = relation_bytes // block_size
    reclaimable_pages = max(relpages - ideal_pages, 0)
    reclaimable_bytes = min(reclaimable_pages * block_size, free_bytes)
    alignment_waste = max(free_bytes - reclaimable_bytes, 0)
    reclaimable_pct = 100.0 * reclaimable_bytes / relation_bytes

    # Online-reclaimable can never exceed the full-rewrite floor; the rest of
    # the reclaimable bloat is what only VACUUM FULL / pg_repack can return.
    online = max(min(online_reclaimable_bytes, reclaimable_bytes), 0)
    vacuum_full_only = max(reclaimable_bytes - online, 0)

    return BloatStats(
        free_percent=free_percent,
        free_bytes=free_bytes,
        reclaimable_percent=reclaimable_pct,
        reclaimable_bytes=reclaimable_bytes,
        alignment_waste_bytes=alignment_waste,
        online_reclaimable_bytes=online,
        vacuum_full_only_bytes=vacuum_full_only,
    )


def get_bloat_stats(
    conn: psycopg.Connection,
    schema: str,
    table: str,
    log: object = None,
) -> BloatStats:
    """Estimate heap bloat: live data / alignment waste / reclaimable.

    The raw FSM free space includes structural per-page alignment waste
    that even a full rewrite cannot reclaim; this separates it from the
    genuinely reclaimable part (see :func:`_bloat_from_occupied`).
    """
    qname = qualified_name(conn, schema, table)
    bs = _block_size(conn)
    fillfactor = _fillfactor(conn, schema, table)

    free_bytes = relation_free_space_bytes(conn, qname)
    with conn.cursor() as cur:
        cur.execute("SELECT pg_catalog.pg_relation_size(%s::regclass)", (qname,))
        (table_bytes,) = _row(cur)
    table_bytes = int(table_bytes or 0)

    # Online-reclaimable threshold for the heap = the typical live-tuple
    # on-page footprint (from catalog stats; falls back to a whole page if
    # unknown, i.e. only genuinely large holes count).
    from pg_compact.db import avg_row_size

    avg = avg_row_size(conn, schema, table)
    heap_unit = onpage_footprint(avg) if avg else (bs - _PAGE_HEADER_BYTES)
    online = _online_reclaimable_bytes(conn, qname, heap_unit)

    return _bloat_from_occupied(table_bytes, free_bytes, bs, fillfactor, online)


def get_toast_relation(conn: psycopg.Connection, schema: str, table: str) -> str | None:
    """The TOAST table's regclass name, or None if the table has none."""
    qname = qualified_name(conn, schema, table)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT CASE WHEN reltoastrelid = 0 THEN NULL
                        ELSE reltoastrelid::regclass::text
                   END
            FROM pg_catalog.pg_class WHERE oid = %s::regclass
            """,
            (qname,),
        )
        row = cur.fetchone()
        return row[0] if row and row[0] else None


def get_toast_bloat_stats(
    conn: psycopg.Connection, schema: str, table: str,
) -> BloatStats | None:
    """Bloat of the table's TOAST storage: live data / alignment waste /
    reclaimable — the same three-way breakdown as the heap.

    Returns None if the table has no TOAST relation at all.  Uses the same
    occupied-volume model as the heap (see :func:`_bloat_from_occupied`):
    the reclaimable part is what a full rewrite would free; the rest of the
    FSM free space is sub-chunk padding that survives even a repack.
    """
    toast_relation = get_toast_relation(conn, schema, table)
    if toast_relation is None:
        return None

    bs = _block_size(conn)
    with conn.cursor() as cur:
        cur.execute("SELECT pg_catalog.pg_relation_size(%s::regclass)", (toast_relation,))
        (toast_bytes,) = _row(cur)
    toast_bytes = int(toast_bytes or 0)
    if toast_bytes == 0:
        return None

    free_bytes = relation_free_space_bytes(conn, toast_relation)
    # TOAST relocates whole values, gated by a full chunk (~2036 B on-page ->
    # needs avail >= 2048).  Almost no fragmented TOAST page reaches that, so
    # online-reclaimable is usually ~0 and the bloat is VACUUM-FULL-only.
    online = _online_reclaimable_bytes(conn, toast_relation, FULL_TOAST_CHUNK_FOOTPRINT)
    return _bloat_from_occupied(toast_bytes, free_bytes, bs, 100, online)
