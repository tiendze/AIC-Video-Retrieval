"""RED reproducer for eval/metrics/retrieval.py (plan §P0.9) -- pure scoring
functions over a list of per-query ranks. No store/encoder involved: the
`SearchService` -> `RetrievedHit` conversion happens in eval/run.py, this
module only knows about ranks and the point-in-segment hit rule.
"""

from __future__ import annotations

import math

import pytest

from eval.metrics.retrieval import (
    GroundTruthQuery,
    RetrievedHit,
    aic_final_score,
    first_hit_rank,
    mean_average_precision,
    mean_reciprocal_rank,
    median_rank,
    ndcg_at_k,
    recall_at_k,
    recall_at_k_by_group,
)


def _hits(*pairs: tuple[str, float]) -> list[RetrievedHit]:
    return [RetrievedHit(video_id=video_id, timestamp=timestamp) for video_id, timestamp in pairs]


def _gt(video_id: str = "L21_V001", start_sec: float = 10.0, end_sec: float = 12.0) -> GroundTruthQuery:
    return GroundTruthQuery(video_id=video_id, start_sec=start_sec, end_sec=end_sec)


# --------------------------------------------------------------------------
# first_hit_rank -- point-in-segment rule (PRD §6.2)
# --------------------------------------------------------------------------


def test_first_hit_rank_video_level_matches_any_timestamp_in_correct_video():
    hits = _hits(("L21_V999", 0.0), ("L21_V001", 500.0))  # timestamp way outside gt window

    rank = first_hit_rank(hits, _gt(), level="video")

    assert rank == 2


def test_first_hit_rank_video_level_returns_none_when_video_never_hit():
    hits = _hits(("L21_V999", 0.0), ("L21_V888", 1.0))

    assert first_hit_rank(hits, _gt(), level="video") is None


def test_first_hit_rank_segment_level_requires_timestamp_inside_window():
    hits = _hits(("L21_V001", 11.0))

    assert first_hit_rank(hits, _gt(start_sec=10.0, end_sec=12.0), level="segment") == 1


def test_first_hit_rank_segment_level_returns_none_when_timestamp_outside_window_even_if_video_matches():
    # Same video, but the keyframe timestamp falls outside [start_sec, end_sec]
    # -- this is exactly the video-level-high/segment-level-low split the plan
    # calls out as diagnostic of a keyframe-sampling bug, not an embedding bug.
    hits = _hits(("L21_V001", 500.0))

    assert first_hit_rank(hits, _gt(start_sec=10.0, end_sec=12.0), level="segment") is None


def test_first_hit_rank_segment_level_boundaries_are_inclusive():
    gt = _gt(start_sec=10.0, end_sec=12.0)

    assert first_hit_rank(_hits(("L21_V001", 10.0)), gt, level="segment") == 1
    assert first_hit_rank(_hits(("L21_V001", 12.0)), gt, level="segment") == 1


def test_first_hit_rank_is_1_indexed_and_returns_first_matching_position():
    hits = _hits(("L21_V999", 0.0), ("L21_V999", 0.0), ("L21_V001", 11.0))

    assert first_hit_rank(hits, _gt(), level="segment") == 3


def test_first_hit_rank_returns_none_for_empty_hits():
    assert first_hit_rank([], _gt(), level="video") is None


# --------------------------------------------------------------------------
# recall_at_k
# --------------------------------------------------------------------------


def test_recall_at_k_counts_only_ranks_within_k():
    ranks = [1, 3, None, 7]

    assert recall_at_k(ranks, k=1) == pytest.approx(0.25)
    assert recall_at_k(ranks, k=3) == pytest.approx(0.5)
    assert recall_at_k(ranks, k=7) == pytest.approx(0.75)


def test_recall_at_k_raises_on_empty_ranks():
    # CLAUDE.md §7: missing data must fail loudly, not report a misleading 0.0
    with pytest.raises(ValueError):
        recall_at_k([], k=5)


# --------------------------------------------------------------------------
# mean_reciprocal_rank / mean_average_precision
# --------------------------------------------------------------------------


def test_mean_reciprocal_rank_treats_not_found_as_zero():
    ranks = [1, 4, None]

    assert mean_reciprocal_rank(ranks) == pytest.approx((1 / 1 + 1 / 4 + 0) / 3)


def test_mean_reciprocal_rank_raises_on_empty_ranks():
    with pytest.raises(ValueError):
        mean_reciprocal_rank([])


def test_mean_average_precision_equals_reciprocal_rank_for_single_relevant_item():
    # Ground truth here has exactly one relevant item per query (a single
    # caption -> a single shot), so AP collapses to 1/rank -- documented
    # assumption, not a generic multi-relevant mAP implementation.
    ranks = [1, 4, None]

    assert mean_average_precision(ranks) == pytest.approx(mean_reciprocal_rank(ranks))


# --------------------------------------------------------------------------
# ndcg_at_k
# --------------------------------------------------------------------------


def test_ndcg_at_k_gives_full_score_for_rank_1():
    assert ndcg_at_k([1], k=10) == pytest.approx(1.0)


def test_ndcg_at_k_is_zero_beyond_k():
    assert ndcg_at_k([11], k=10) == pytest.approx(0.0)


def test_ndcg_at_k_discounts_by_log2_rank_plus_1():
    # single relevant item per query => IDCG = 1 / log2(1+1) = 1, so nDCG
    # reduces to DCG itself.
    assert ndcg_at_k([3], k=10) == pytest.approx(1 / math.log2(3 + 1))


def test_ndcg_at_k_raises_on_empty_ranks():
    with pytest.raises(ValueError):
        ndcg_at_k([], k=10)


# --------------------------------------------------------------------------
# median_rank
# --------------------------------------------------------------------------


def test_median_rank_of_found_ranks_ignores_not_found():
    assert median_rank([1, 3, None, 5]) == pytest.approx(3.0)


def test_median_rank_averages_the_two_middle_values_for_even_count():
    assert median_rank([1, 2, 3, 4]) == pytest.approx(2.5)


def test_median_rank_returns_none_when_nothing_found():
    assert median_rank([None, None]) is None


def test_median_rank_raises_on_empty_ranks():
    with pytest.raises(ValueError):
        median_rank([])


# --------------------------------------------------------------------------
# aic_final_score -- mean(Recall@k) over the official k in {1,5,20,50,100}
# --------------------------------------------------------------------------


def test_aic_final_score_is_mean_of_recall_at_the_five_official_ks():
    ranks = [1] * 10  # every query found at rank 1 -> every Recall@k is 1.0

    assert aic_final_score(ranks) == pytest.approx(1.0)


def test_aic_final_score_matches_manual_mean_of_recall_at_k():
    ranks = [1, 6, 30, 60, None]
    expected = sum(recall_at_k(ranks, k) for k in (1, 5, 20, 50, 100)) / 5

    assert aic_final_score(ranks) == pytest.approx(expected)


def test_aic_final_score_raises_on_empty_ranks():
    with pytest.raises(ValueError):
        aic_final_score([])


# --------------------------------------------------------------------------
# recall_at_k_by_group -- author/batch breakdown
# --------------------------------------------------------------------------


def test_recall_at_k_by_group_partitions_correctly():
    ranks_with_group = [(1, "L21"), (None, "L21"), (2, "L22")]

    result = recall_at_k_by_group(ranks_with_group, k=1)

    assert result == {"L21": pytest.approx(0.5), "L22": pytest.approx(0.0)}


def test_recall_at_k_by_group_raises_on_empty_input():
    with pytest.raises(ValueError):
        recall_at_k_by_group([], k=1)
