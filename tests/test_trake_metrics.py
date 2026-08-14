"""RED reproducer for eval/metrics/trake.py (plan §P2.3, PRD §6.4) -- pure
scoring functions over TRAKE query results. Mirrors
tests/test_retrieval_metrics.py's structure but a different formula: TRAKE
scores an N-keyframe SEQUENCE per query, not a single point-in-segment hit
(module docstring of eval/metrics/retrieval.py explicitly says not to reuse
its functions for this).

Kendall's tau design (chủ dự án 2026-08-14, "Cách B"): each retrieved
keyframe is reassigned to whichever ground-truth event's segment center it
is temporally CLOSEST to (not the event at the same position i) -- because
DANTE's τ<t invariant already forces the retrieved sequence to be strictly
time-increasing, and the N ground-truth segments are already time-ordered by
construction (N consecutive shots), so a naive position-i-vs-position-i
comparison would always read 1.0 regardless of retrieval quality. Nearest-
segment reassignment is the only version of this metric that can actually
vary with retrieval quality (and therefore with λ, per PRD §6.4).
"""

from __future__ import annotations

import pytest

from eval.metrics.trake import (
    TrakeGroundTruthEvent,
    TrakeGroundTruthQuery,
    TrakeQueryResult,
    TrakeRetrievedVideo,
    assign_nearest_segment_order,
    kendall_tau,
    mean_kendall_tau,
    mean_offset_sec,
    mean_temporal_offset_sec,
    per_event_hits,
    per_event_recall_at_k,
    score_query,
    sequence_exact_match_rate,
    sequence_kendall_tau,
    video_level_rank,
    video_level_recall_at_k,
)


def _events(*windows: tuple[float, float]) -> tuple[TrakeGroundTruthEvent, ...]:
    return tuple(TrakeGroundTruthEvent(start_sec=s, end_sec=e) for s, e in windows)


def _gt(video_id: str, *windows: tuple[float, float]) -> TrakeGroundTruthQuery:
    return TrakeGroundTruthQuery(video_id=video_id, events=_events(*windows))


def _video(video_id: str, *timestamps: float) -> TrakeRetrievedVideo:
    return TrakeRetrievedVideo(video_id=video_id, keyframe_timestamps=tuple(timestamps))


# --------------------------------------------------------------------------
# video_level_rank
# --------------------------------------------------------------------------


def test_video_level_rank_returns_1_indexed_position_of_matching_video():
    retrieved = [_video("L21_V999", 0.0), _video("L21_V001", 5.0)]

    assert video_level_rank(retrieved, "L21_V001") == 2


def test_video_level_rank_returns_none_when_video_never_retrieved():
    retrieved = [_video("L21_V999", 0.0)]

    assert video_level_rank(retrieved, "L21_V001") is None


# --------------------------------------------------------------------------
# per_event_hits -- position-aligned point-in-segment (PRD §6.2 rule reused
# at each fixed position i, not a matching problem)
# --------------------------------------------------------------------------


def test_per_event_hits_is_true_when_timestamp_falls_inside_its_own_window():
    events = _events((0.0, 2.0), (10.0, 12.0), (20.0, 22.0))

    hits = per_event_hits([1.0, 11.0, 21.0], events)

    assert hits == (True, True, True)


def test_per_event_hits_is_false_when_timestamp_falls_outside_its_own_window():
    events = _events((0.0, 2.0), (10.0, 12.0))

    hits = per_event_hits([1.0, 999.0], events)

    assert hits == (True, False)


def test_per_event_hits_window_bounds_are_inclusive():
    events = _events(
        (10.0, 12.0),
    )

    assert per_event_hits([10.0], events) == (True,)
    assert per_event_hits([12.0], events) == (True,)


def test_per_event_hits_raises_on_length_mismatch():
    events = _events((0.0, 2.0), (10.0, 12.0))

    with pytest.raises(ValueError, match="length"):
        per_event_hits([1.0], events)


# --------------------------------------------------------------------------
# assign_nearest_segment_order -- Kendall's tau "Cách B" building block
# --------------------------------------------------------------------------


def test_assign_nearest_segment_order_maps_each_timestamp_to_its_closest_center():
    # centers: 1.0, 11.0, 21.0
    events = _events((0.0, 2.0), (10.0, 12.0), (20.0, 22.0))

    assert assign_nearest_segment_order([1.0, 11.0, 21.0], events) == (0, 1, 2)


def test_assign_nearest_segment_order_can_produce_an_out_of_order_assignment():
    # retrieved position 0's timestamp (21.0) is actually closest to event 2's
    # center, and position 2's timestamp (1.0) is closest to event 0's --
    # a real misassignment, not a position echo.
    events = _events((0.0, 2.0), (10.0, 12.0), (20.0, 22.0))

    assert assign_nearest_segment_order([21.0, 11.0, 1.0], events) == (2, 1, 0)


def test_assign_nearest_segment_order_breaks_ties_toward_the_earlier_segment():
    events = _events((0.0, 0.0), (10.0, 10.0))  # centers 0.0 and 10.0

    assert assign_nearest_segment_order([5.0], events) == (0,)


# --------------------------------------------------------------------------
# kendall_tau -- generic pairwise tau-b
# --------------------------------------------------------------------------


def test_kendall_tau_is_1_for_perfectly_concordant_sequences():
    assert kendall_tau([0, 1, 2], [0, 1, 2]) == pytest.approx(1.0)


def test_kendall_tau_is_negative_1_for_perfectly_reversed_sequences():
    assert kendall_tau([0, 1, 2], [2, 1, 0]) == pytest.approx(-1.0)


def test_kendall_tau_is_between_bounds_for_a_single_swapped_pair():
    tau = kendall_tau([0, 1, 2], [0, 2, 1])

    assert -1.0 < tau < 1.0


def test_kendall_tau_raises_on_fewer_than_two_points():
    with pytest.raises(ValueError, match="at least 2"):
        kendall_tau([0], [0])


def test_kendall_tau_raises_on_mismatched_lengths():
    with pytest.raises(ValueError, match="length"):
        kendall_tau([0, 1], [0, 1, 2])


def test_kendall_tau_handles_a_pair_tied_only_in_x():
    # pair (0,1): dx=0 (tied in x), dy=1 -- excluded from concordant/discordant
    # counts but still counted in the tau-b denominator via tied_x.
    assert kendall_tau([0, 0, 1], [0, 1, 2]) == pytest.approx(2 / 6**0.5)


def test_kendall_tau_handles_a_pair_tied_only_in_y():
    assert kendall_tau([0, 1, 2], [0, 0, 1]) == pytest.approx(2 / 6**0.5)


def test_kendall_tau_is_0_when_every_pair_is_tied_in_both_x_and_y():
    assert kendall_tau([0, 0], [0, 0]) == pytest.approx(0.0)


# --------------------------------------------------------------------------
# sequence_kendall_tau / mean_temporal_offset_sec
# --------------------------------------------------------------------------


def test_sequence_kendall_tau_is_1_when_every_keyframe_lands_in_its_own_segment():
    events = _events((0.0, 2.0), (10.0, 12.0), (20.0, 22.0))

    assert sequence_kendall_tau([1.0, 11.0, 21.0], events) == pytest.approx(1.0)


def test_sequence_kendall_tau_drops_when_frames_are_misassigned():
    events = _events((0.0, 2.0), (10.0, 12.0), (20.0, 22.0))

    assert sequence_kendall_tau([21.0, 11.0, 1.0], events) == pytest.approx(-1.0)


def test_mean_temporal_offset_sec_is_zero_when_every_timestamp_is_exactly_centered():
    events = _events((0.0, 2.0), (10.0, 14.0))  # centers 1.0, 12.0

    assert mean_temporal_offset_sec([1.0, 12.0], events) == pytest.approx(0.0)


def test_mean_temporal_offset_sec_averages_absolute_distance_to_each_center():
    events = _events((0.0, 2.0), (10.0, 14.0))  # centers 1.0, 12.0

    # |3 - 1| = 2, |20 - 12| = 8 -> mean 5.0
    assert mean_temporal_offset_sec([3.0, 20.0], events) == pytest.approx(5.0)


# --------------------------------------------------------------------------
# score_query -- one query's full outcome
# --------------------------------------------------------------------------


def test_score_query_when_video_not_found_leaves_every_other_field_none():
    gt = _gt("L21_V001", (0.0, 2.0), (10.0, 12.0))
    retrieved = [_video("L21_V999", 0.0, 10.0)]

    result = score_query(retrieved, gt)

    assert result == TrakeQueryResult(
        video_rank=None, per_event_hit=None, kendall_tau=None, mean_temporal_offset_sec=None
    )


def test_score_query_when_video_found_populates_every_field():
    gt = _gt("L21_V001", (0.0, 2.0), (10.0, 12.0))
    retrieved = [_video("L21_V001", 1.0, 11.0)]

    result = score_query(retrieved, gt)

    assert result.video_rank == 1
    assert result.per_event_hit == (True, True)
    assert result.kendall_tau == pytest.approx(1.0)
    assert result.mean_temporal_offset_sec == pytest.approx(0.0)


def test_score_query_raises_when_matched_videos_keyframe_count_does_not_match_n():
    # TrakeService's own contract guarantees len(keyframes) == len(events);
    # a caller violating that is a wiring bug, not a scoreable miss.
    gt = _gt("L21_V001", (0.0, 2.0), (10.0, 12.0))
    retrieved = [_video("L21_V001", 1.0)]  # only 1 timestamp, gt has 2 events

    with pytest.raises(ValueError, match="keyframe"):
        score_query(retrieved, gt)


# --------------------------------------------------------------------------
# Aggregation over many queries
# --------------------------------------------------------------------------


def _found(video_rank: int, per_event_hit: tuple[bool, ...], tau: float, offset: float):
    return TrakeQueryResult(
        video_rank=video_rank,
        per_event_hit=per_event_hit,
        kendall_tau=tau,
        mean_temporal_offset_sec=offset,
    )


_NOT_FOUND = TrakeQueryResult(
    video_rank=None, per_event_hit=None, kendall_tau=None, mean_temporal_offset_sec=None
)


def test_sequence_exact_match_rate_counts_only_all_events_hit():
    results = [
        _found(1, (True, True), 1.0, 0.0),
        _found(1, (True, False), 1.0, 0.0),
        _NOT_FOUND,
    ]

    assert sequence_exact_match_rate(results) == pytest.approx(1 / 3)


def test_sequence_exact_match_rate_raises_on_empty_input():
    with pytest.raises(ValueError, match="empty"):
        sequence_exact_match_rate([])


def test_video_level_recall_at_k_counts_videos_found_within_k():
    results = [_found(1, (True,), 1.0, 0.0), _found(5, (True,), 1.0, 0.0), _NOT_FOUND]

    assert video_level_recall_at_k(results, k=1) == pytest.approx(1 / 3)
    assert video_level_recall_at_k(results, k=5) == pytest.approx(2 / 3)


def test_per_event_recall_at_k_is_a_miss_when_video_rank_exceeds_k():
    results = [_found(5, (True, True), 1.0, 0.0)]

    assert per_event_recall_at_k(results, event_index=0, k=1) == pytest.approx(0.0)
    assert per_event_recall_at_k(results, event_index=0, k=5) == pytest.approx(1.0)


def test_per_event_recall_at_k_is_a_miss_when_that_specific_event_was_not_hit():
    results = [_found(1, (True, False), 1.0, 0.0)]

    assert per_event_recall_at_k(results, event_index=0, k=1) == pytest.approx(1.0)
    assert per_event_recall_at_k(results, event_index=1, k=1) == pytest.approx(0.0)


def test_mean_kendall_tau_averages_over_found_queries_only():
    results = [_found(1, (True,), 1.0, 0.0), _found(1, (True,), 0.5, 0.0), _NOT_FOUND]

    assert mean_kendall_tau(results) == pytest.approx(0.75)


def test_mean_kendall_tau_raises_when_no_query_found_the_video():
    with pytest.raises(ValueError, match="found"):
        mean_kendall_tau([_NOT_FOUND])


def test_mean_offset_sec_averages_over_found_queries_only():
    results = [_found(1, (True,), 1.0, 2.0), _found(1, (True,), 1.0, 4.0), _NOT_FOUND]

    assert mean_offset_sec(results) == pytest.approx(3.0)


def test_mean_offset_sec_raises_when_no_query_found_the_video():
    with pytest.raises(ValueError, match="found"):
        mean_offset_sec([_NOT_FOUND])
