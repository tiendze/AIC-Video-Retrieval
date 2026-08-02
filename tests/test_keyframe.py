"""RED reproducer for pipelines/keyframe.py — PRD §F1 4-keyframe/shot formula.

k_extract = { K_(a + floor(i * (b - a) / 3)) , for all i in {0,1,2,3} }
"""

from __future__ import annotations

from pipelines.keyframe import keyframe_frame_indices, keyframe_indices_for_shots
from pipelines.shot_detect import Shot


def test_keyframe_indices_match_paper_formula():
    # a=10, b=100, span=90: 10, 10+30, 10+60, 10+90
    shot = Shot(start_frame=10, end_frame=100)
    assert keyframe_frame_indices(shot) == [10, 40, 70, 100]


def test_keyframe_indices_match_paper_formula_with_non_divisible_span():
    # a=0, b=10, span=10: 0, floor(10/3)=3, floor(20/3)=6, floor(30/3)=10
    shot = Shot(start_frame=0, end_frame=10)
    assert keyframe_frame_indices(shot) == [0, 3, 6, 10]


def test_keyframe_frame_indices_always_includes_start_and_end():
    shot = Shot(start_frame=200, end_frame=257)
    indices = keyframe_frame_indices(shot)
    assert indices[0] == shot.start_frame
    assert indices[-1] == shot.end_frame


def test_keyframe_frame_indices_deduplicates_short_shot_per_set_notation():
    # a=5, b=7, span=2: 5, 5+floor(2/3)=5, 5+floor(4/3)=6, 5+floor(6/3)=7
    # -> {5,5,6,7} deduplicated, not 4 repeated entries
    shot = Shot(start_frame=5, end_frame=7)
    assert keyframe_frame_indices(shot) == [5, 6, 7]


def test_keyframe_frame_indices_single_frame_shot_yields_one_index():
    shot = Shot(start_frame=5, end_frame=5)
    assert keyframe_frame_indices(shot) == [5]


def test_keyframe_frame_indices_two_frame_shot_yields_two_indices():
    # a=0, b=1, span=1: 0, floor(1/3)=0, floor(2/3)=0, floor(3/3)=1 -> {0,1}
    shot = Shot(start_frame=0, end_frame=1)
    assert keyframe_frame_indices(shot) == [0, 1]


def test_keyframe_indices_for_shots_flattens_in_shot_order():
    shots = [Shot(start_frame=0, end_frame=3), Shot(start_frame=10, end_frame=13)]
    assert keyframe_indices_for_shots(shots) == [0, 1, 2, 3, 10, 11, 12, 13]


def test_keyframe_indices_for_shots_empty_list_returns_empty():
    assert keyframe_indices_for_shots([]) == []
