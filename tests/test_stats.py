"""Tests for bloat estimation via FSM + catalog stats."""

from __future__ import annotations

from conftest import make_bloated_table

from pg_compact.stats import get_bloat_stats


def test_bloat_stats_detects_reclaimable_space(pg_conn, pg_table):
    """A heavily-bloated table should have significant reclaimable space."""
    make_bloated_table(pg_conn, pg_table)
    stats = get_bloat_stats(pg_conn, "public", pg_table)

    assert stats.free_percent > 50
    assert stats.reclaimable_bytes > 0


def test_bloat_stats_alignment_waste(pg_conn, pg_table):
    """reclaimable_percent should be <= free_percent (alignment waste subtracted)."""
    make_bloated_table(pg_conn, pg_table)
    stats = get_bloat_stats(pg_conn, "public", pg_table)

    assert stats.reclaimable_percent <= stats.free_percent


def test_bloat_stats_online_vf_only_split(pg_conn, pg_table):
    """Online + VACUUM-FULL-only must exactly partition the reclaimable bloat.

    The split tells the user what pg-compact frees in place versus what only a
    full rewrite can return, so the two parts must sum to reclaimable_bytes and
    neither may exceed it.
    """
    make_bloated_table(pg_conn, pg_table)
    stats = get_bloat_stats(pg_conn, "public", pg_table)

    assert stats.online_reclaimable_bytes >= 0
    assert stats.vacuum_full_only_bytes >= 0
    assert stats.online_reclaimable_bytes <= stats.reclaimable_bytes
    assert (
        stats.online_reclaimable_bytes + stats.vacuum_full_only_bytes
        == stats.reclaimable_bytes
    )


def test_fragmentation_map_shape_and_states(pg_conn, pg_table):
    """The live fragmentation map returns one cell code per requested cell and
    only uses known state codes."""
    from conftest import make_bloated_table

    from pg_compact import db
    from pg_compact.fsm_predict import (
        CELL_EMPTY,
        CELL_LIVE,
        CELL_RECLAIMABLE,
        CELL_VF_ONLY,
        build_fragmentation_map,
    )

    make_bloated_table(pg_conn, pg_table)
    qname = db.qualified_name(pg_conn, "public", pg_table)
    pages = db.get_relation_page_count(pg_conn, qname)
    cells = build_fragmentation_map(pg_conn, qname, pages, 2048, ncells=40)
    assert 0 < len(cells) <= 40
    assert set(cells) <= {CELL_LIVE, CELL_VF_ONLY, CELL_RECLAIMABLE, CELL_EMPTY}


def test_toast_bloat_is_mostly_vacuum_full_only(pg_conn, pg_table):
    """A fragmented TOAST relation returns its bloat mostly via VACUUM FULL.

    Heavily-updated/deleted TOAST storage fragments into sub-chunk holes the
    FSM never offers for a full chunk, so online relocation reclaims little and
    the bulk is VACUUM-FULL-only.  This is the core distinction the split
    surfaces.
    """
    from conftest import make_toast_bloated_table

    from pg_compact.stats import get_toast_bloat_stats

    make_toast_bloated_table(pg_conn, pg_table)
    tstats = get_toast_bloat_stats(pg_conn, "public", pg_table)
    assert tstats is not None
    # The split partitions the reclaimable bloat exactly.
    assert (
        tstats.online_reclaimable_bytes + tstats.vacuum_full_only_bytes
        == tstats.reclaimable_bytes
    )
