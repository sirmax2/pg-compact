"""FSM placement arithmetic — the rule that decides online vs VACUUM-FULL-only.

PostgreSQL stores one byte per heap page in the free space map, quantising a
page's free space into 256 categories of ``FSM_CAT_STEP = BLCKSZ / 256 = 32``
bytes each (``src/backend/storage/freespace/freespace.c``).  When an
``INSERT``/``UPDATE`` needs ``len`` bytes, ``RelationGetBufferForTuple``
(``hio.c``) asks the FSM for a page, and the FSM only offers one whose *stored*
category is ``>= ceil(len / 32)``.

Two roundings combine to trap free space:

* the stored category rounds a page's free space **down** to a multiple of 32;
* the request rounds the needed size **up** to a multiple of 32.

So a hole a little smaller than one whole tuple/chunk (after rounding up) is
invisible to the allocator — it can never be filled by relocating a row/chunk
into it.  That trapped space is reclaimable only by a full rewrite
(``VACUUM FULL`` / ``CLUSTER`` / ``pg_repack``), which lays rows down
sequentially and controls the layout.  Space in holes at or above the
threshold is what an in-place tool like pg-compact can relocate and truncate.

This module provides that placement predicate and the derived constants;
``stats.py`` uses them to split a relation's reclaimable bloat into the part
pg-compact can free online and the part that needs a full rewrite.

The arithmetic is a direct transcription of ``fsm_space_avail_to_cat`` /
``fsm_space_needed_to_cat`` / ``fsm_space_cat_to_avail`` from ``freespace.c``.
"""

from __future__ import annotations

# --- freespace.c constants (default BLCKSZ = 8192) -------------------------

BLCKSZ = 8192
FSM_CATEGORIES = 256
FSM_CAT_STEP = BLCKSZ // FSM_CATEGORIES  # 32 bytes per category
# MaxFSMRequestSize == MaxHeapTupleSize: the largest request the FSM tracks.
# Anything >= this maps to the top category (255).
MAX_FSM_REQUEST_SIZE = 8164

# Per-tuple line pointer (ItemIdData) added when a new tuple lands on a page.
LINE_POINTER_BYTES = 4

# On-page footprint of a full TOAST chunk: TOAST_MAX_CHUNK_SIZE (~1996 B) plus
# the heap tuple header, MAXALIGN'd (~2032 B measured), plus the line pointer.
# A page needs FSM category ceil(2036/32)=64 to accept one, i.e. avail >= 2048.
FULL_TOAST_CHUNK_FOOTPRINT = 2036


# --- FSM category arithmetic (verbatim from freespace.c) -------------------


def fsm_space_avail_to_cat(avail: int) -> int:
    """Map available bytes on a page to its stored FSM category (rounds DOWN).

    This is what the FSM records per page, so it *understates* the true free
    space by up to ``FSM_CAT_STEP - 1`` bytes — the root cause of "a hole the
    same size as a tuple still won't be offered".
    """
    if avail >= MAX_FSM_REQUEST_SIZE:
        return 255
    cat = avail // FSM_CAT_STEP
    if cat > 255:
        cat = 255
    return cat


def fsm_space_needed_to_cat(needed: int) -> int:
    """Map a requested allocation size to the minimum satisfying category (rounds UP)."""
    if needed <= 0:
        needed = 1
    cat = (needed + FSM_CAT_STEP - 1) // FSM_CAT_STEP
    if cat > 255:
        cat = 255
    return cat


def fsm_space_cat_to_avail(cat: int) -> int:
    """Bytes the FSM reports for a category — what ``pg_freespace().avail`` returns."""
    if cat >= 255:
        return MAX_FSM_REQUEST_SIZE
    return cat * FSM_CAT_STEP


def fsm_admits(page_avail: int, needed_len: int) -> bool:
    """True iff the FSM would offer a page with ``page_avail`` free bytes for an
    allocation of ``needed_len`` bytes.

    The single predicate everything rests on: placement succeeds iff
    ``floor(avail / 32) >= ceil(needed / 32)``.  ``page_avail`` is the
    FSM-reported value (already rounded down), matching ``pg_freespace()``.
    """
    return fsm_space_avail_to_cat(page_avail) >= fsm_space_needed_to_cat(needed_len)


def admit_threshold(needed_len: int) -> int:
    """Minimum ``pg_freespace().avail`` a page must report to accept ``needed_len``.

    Free space below this is trapped (sub-tuple / sub-chunk fragmentation) and
    only a full rewrite can pack it away.
    """
    return fsm_space_needed_to_cat(needed_len) * FSM_CAT_STEP


def maxalign(n: int) -> int:
    """MAXALIGN — round up to an 8-byte boundary (typical MAXIMUM_ALIGNOF)."""
    return (n + 7) & ~7


def onpage_footprint(tuple_len: int) -> int:
    """Total page space a relocated tuple of ``tuple_len`` data bytes consumes.

    ``RelationGetBufferForTuple`` asks the FSM for ``MAXALIGN(tuple_len)`` and
    the page must additionally have room for a new line pointer.
    """
    return maxalign(tuple_len) + LINE_POINTER_BYTES


# --- live fragmentation map (for the interactive display) ------------------

# Single-character state codes for one map cell (a range of blocks).
CELL_LIVE = "L"        # mostly occupied (live data / packed)
CELL_VF_ONLY = "V"     # free, but in sub-unit holes: only VACUUM FULL frees it
CELL_RECLAIMABLE = "R"  # free in holes big enough for online relocation
CELL_EMPTY = "E"       # (near-)empty pages — truncatable tail candidates


def build_fragmentation_map(
    conn,
    relation: str,
    page_count: int,
    unit_footprint: int,
    ncells: int = 60,
) -> list[str]:
    """Classify a relation's block range into ``ncells`` map cells (front->tail).

    One ``pg_freespace()`` scan bucketed by block number.  Each cell aggregates
    its blocks' free space and is labelled by the dominant state, using the FSM
    admit threshold to tell reclaimable-now holes from sub-unit (VACUUM-FULL-
    only) ones:

      * ``CELL_EMPTY``       — free space >= ~90% of the cell's capacity
      * ``CELL_RECLAIMABLE`` — most free space is in holes the FSM would offer
        a relocation (avail >= admit_threshold)
      * ``CELL_VF_ONLY``     — most free space is in sub-unit holes
      * ``CELL_LIVE``        — little free space (packed with data)

    Cheap enough to refresh periodically during compaction (reads only the
    small FSM fork), which is what makes the map "live".
    """
    if page_count <= 0:
        return []
    ncells = max(1, min(ncells, page_count))
    threshold = admit_threshold(unit_footprint)
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT width_bucket(blkno, 0, %s, %s) AS cell, "
            "coalesce(sum(avail), 0) AS total_free, "
            "coalesce(sum(avail) FILTER (WHERE avail >= %s), 0) AS reclaimable_free, "
            "count(*) AS pages "
            "FROM pg_freespace(%s::regclass) GROUP BY cell ORDER BY cell",
            (page_count, ncells, threshold, relation),
        )
        rows = cur.fetchall()
    finally:
        cur.close()

    cell_capacity = (page_count // ncells + 1) * (BLCKSZ - PAGE_HEADER_BYTES)
    cells = [CELL_LIVE] * (ncells + 1)
    for cell_idx, total_free, reclaimable_free, _pages in rows:
        i = min(int(cell_idx), ncells) - 1
        if i < 0:
            i = 0
        total_free = int(total_free or 0)
        reclaimable_free = int(reclaimable_free or 0)
        if cell_capacity and total_free >= 0.9 * cell_capacity:
            cells[i] = CELL_EMPTY
        elif total_free <= 0.10 * cell_capacity:
            cells[i] = CELL_LIVE
        elif reclaimable_free >= 0.5 * total_free:
            cells[i] = CELL_RECLAIMABLE
        else:
            cells[i] = CELL_VF_ONLY
    return cells[:ncells]


# Page-header constant reused by the map (mirrors freespace.c usable space).
PAGE_HEADER_BYTES = 24
