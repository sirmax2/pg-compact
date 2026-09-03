"""Unit tests for the FSM placement arithmetic.

These functions are a direct transcription of freespace.c and are fully
deterministic, so the tests pin the exact constants and the placement
predicate that the online-vs-VACUUM-FULL-only split in stats.py rests on.
"""

from __future__ import annotations

from pg_compact.fsm_predict import (
    BLCKSZ,
    FSM_CAT_STEP,
    FULL_TOAST_CHUNK_FOOTPRINT,
    MAX_FSM_REQUEST_SIZE,
    admit_threshold,
    fsm_admits,
    fsm_space_avail_to_cat,
    fsm_space_cat_to_avail,
    fsm_space_needed_to_cat,
    onpage_footprint,
)


def test_cat_step_is_32_bytes():
    assert FSM_CAT_STEP == 32
    assert BLCKSZ == 8192


def test_avail_to_cat_rounds_down():
    # 2047 bytes is one short of category 64 -> stays at 63.
    assert fsm_space_avail_to_cat(2047) == 63
    assert fsm_space_avail_to_cat(2048) == 64
    assert fsm_space_avail_to_cat(0) == 0
    # At or above the max request maps to the top category.
    assert fsm_space_avail_to_cat(MAX_FSM_REQUEST_SIZE) == 255
    assert fsm_space_avail_to_cat(BLCKSZ) == 255


def test_needed_to_cat_rounds_up():
    # A full TOAST chunk footprint (2036 B) needs category 64 -> avail >= 2048.
    assert fsm_space_needed_to_cat(2036) == 64
    assert fsm_space_needed_to_cat(2048) == 64
    assert fsm_space_needed_to_cat(2049) == 65
    assert fsm_space_needed_to_cat(1) == 1
    assert fsm_space_needed_to_cat(0) == 1


def test_cat_to_avail_matches_pg_freespace():
    for cat in (0, 1, 63, 64, 128, 254):
        assert fsm_space_cat_to_avail(cat) == cat * FSM_CAT_STEP
    assert fsm_space_cat_to_avail(255) == MAX_FSM_REQUEST_SIZE


def test_fsm_admits_is_the_placement_predicate():
    # A page with 2016 free bytes (cat 63) cannot accept a 2036-byte chunk,
    # even though 2036 < 2048 numerically — the FSM rounds the page down to
    # cat 63 and the request up to cat 64.  This is the sub-chunk trap.
    assert not fsm_admits(2016, 2036)
    assert fsm_admits(2048, 2036)
    # A small tuple fits a small hole.
    assert fsm_admits(256, 200)
    assert not fsm_admits(160, 200)


def test_admit_threshold_full_toast_chunk_is_2048():
    # The crux of the TOAST split: full chunks (~2036 B on-page) require a
    # page reporting avail >= 2048, so pages with 1920-2016 free are one FSM
    # step short and never offered — that free space is VACUUM-FULL-only.
    assert admit_threshold(FULL_TOAST_CHUNK_FOOTPRINT) == 2048
    footprint = onpage_footprint(2032)
    assert footprint >= FULL_TOAST_CHUNK_FOOTPRINT
    assert not fsm_admits(2016, footprint)
    assert fsm_admits(2048, footprint)


def test_onpage_footprint_maxaligns_and_adds_line_pointer():
    # 2029 -> MAXALIGN 2032 + 4-byte line pointer = 2036.
    assert onpage_footprint(2029) == 2036
    # 100 -> MAXALIGN 104 + 4-byte line pointer = 108.
    assert onpage_footprint(100) == 108
