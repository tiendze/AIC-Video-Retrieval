from __future__ import annotations

import numpy as np
import pytest

from app.services.kis_scoring import (
    KisConfig,
    bm25_reference,
    candidate_depth,
    geometric_mean_score,
    normalize_bm25,
    normalize_ocr_text,
    ocr_coverage,
    rank_percentiles,
)

CONFIG = KisConfig(candidate_multiplier=2, candidate_min=10)


def test_candidate_depth_uses_larger_retrieval_pool_than_display_top_k() -> None:
    assert candidate_depth(20, CONFIG) == 40
    assert candidate_depth(2, CONFIG) == 10


def test_config_rejects_invalid_geomean_epsilon() -> None:
    with pytest.raises(ValueError, match="kis_geomean_epsilon"):
        KisConfig(geomean_epsilon=0.0)
    with pytest.raises(ValueError, match="kis_geomean_epsilon"):
        KisConfig(geomean_epsilon=1.0)


def test_rank_percentiles_map_best_to_one_and_worst_to_zero() -> None:
    ranked = rank_percentiles({"A": 0.9, "B": 0.4, "C": 0.1}, ["A", "B", "C"])

    assert ranked["A"].rank == 1
    assert ranked["A"].rank_percentile == pytest.approx(1.0)
    assert ranked["B"].rank == 2
    assert ranked["B"].rank_percentile == pytest.approx(0.5)
    assert ranked["C"].rank == 3
    assert ranked["C"].rank_percentile == pytest.approx(0.0)


def test_rank_percentiles_handle_single_candidate_pool() -> None:
    ranked = rank_percentiles({"A": 0.12}, ["A"])

    assert ranked["A"].rank == 1
    assert ranked["A"].rank_percentile == pytest.approx(1.0)


def test_rank_percentiles_give_equal_scores_equal_worst_group_rank() -> None:
    ranked = rank_percentiles({"A": 0.9, "B": 0.5, "C": 0.5, "D": 0.1}, ["A", "B", "C", "D"])

    assert ranked["A"].rank == 1
    assert ranked["A"].rank_percentile == pytest.approx(1.0)
    assert ranked["B"].rank == 3
    assert ranked["B"].rank_percentile == pytest.approx(1 / 3)
    assert ranked["C"].rank == 3
    assert ranked["C"].rank_percentile == pytest.approx(1 / 3)
    assert ranked["D"].rank == 4
    assert ranked["D"].rank_percentile == pytest.approx(0.0)


def test_rank_percentiles_give_every_worst_tied_candidate_zero() -> None:
    ranked = rank_percentiles({"A": 0.9, "B": 0.0, "C": 0.0}, ["A", "B", "C"])

    assert ranked["B"].rank == 3
    assert ranked["B"].rank_percentile == pytest.approx(0.0)
    assert ranked["C"].rank == 3
    assert ranked["C"].rank_percentile == pytest.approx(0.0)


def test_rank_percentiles_give_all_zero_scores_zero_percentile() -> None:
    ranked = rank_percentiles({"A": 0.0, "B": 0.0}, ["A", "B"])

    assert ranked["A"].rank == 2
    assert ranked["A"].rank_percentile == pytest.approx(0.0)
    assert ranked["B"].rank == 2
    assert ranked["B"].rank_percentile == pytest.approx(0.0)


def test_geometric_mean_prefers_balanced_candidate_over_one_weak_query() -> None:
    candidate_a = geometric_mean_score([1.0, 1.0, 0.2], CONFIG)
    candidate_b = geometric_mean_score([0.7, 0.7, 0.7], CONFIG)

    assert candidate_a == pytest.approx(0.5848035)
    assert candidate_b == pytest.approx(0.7)
    assert candidate_b > candidate_a


def test_one_active_score_geometric_mean_is_that_score() -> None:
    assert geometric_mean_score([0.73], CONFIG) == pytest.approx(0.73)
    assert geometric_mean_score([0.0], CONFIG) == pytest.approx(0.0)


def test_geometric_mean_rejects_out_of_range_scores() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        geometric_mean_score([1.01], CONFIG)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        geometric_mean_score([-0.01], CONFIG)


def test_bm25_reference_uses_max_when_too_few_positive_scores() -> None:
    config = KisConfig(bm25_percentile_min_samples=5)

    assert bm25_reference([1.0, 3.0, 2.0], config) == pytest.approx(3.0)
    assert normalize_bm25(1.5, 3.0) == pytest.approx(0.5)


def test_bm25_reference_uses_percentile_per_query_when_enough_scores() -> None:
    config = KisConfig(bm25_percentile=95, bm25_percentile_min_samples=2)

    ref_a = bm25_reference([1.0, 100.0], config)
    ref_b = bm25_reference([1.0, 2.0], config)

    assert ref_a != ref_b
    assert ref_a == pytest.approx(np.percentile(np.asarray([1.0, 100.0]), 95))
    assert ref_b == pytest.approx(np.percentile(np.asarray([1.0, 2.0]), 95))


def test_bm25_normalization_returns_zero_without_positive_reference() -> None:
    assert bm25_reference([], CONFIG) == 0.0
    assert normalize_bm25(10.0, 0.0) == 0.0


def test_numeric_ocr_tokens_are_strict() -> None:
    assert ocr_coverage("2026", "2026", CONFIG) == pytest.approx(1.0)
    assert ocr_coverage("2026", "2025", CONFIG) == pytest.approx(0.0)


def test_vietnamese_ocr_matches_without_accents() -> None:
    assert ocr_coverage("Việt Nam", "Viet Nam", CONFIG) >= 0.99


def test_ocr_split_word_matches_through_bigram_unit() -> None:
    assert ocr_coverage("Vinamilk", "Vina milk", CONFIG) >= 0.99


def test_short_tokens_do_not_overmatch_by_edit_distance() -> None:
    assert ocr_coverage("AI", "AN", CONFIG) == 0.0


def test_ocr_normalization_preserves_and_strips_vietnamese_forms() -> None:
    normalized = normalize_ocr_text("  Việt, Nam!  ")

    assert normalized.text == "việt nam"
    assert normalized.stripped_text == "viet nam"
    assert normalized.tokens == ["việt", "nam"]
    assert normalized.stripped_tokens == ["viet", "nam"]


def test_scoring_invariants_stay_in_range() -> None:
    ranked = rank_percentiles({"A": 0.9, "B": 0.4, "C": 0.0}, ["A", "B", "C"])
    percentiles = [score.rank_percentile for score in ranked.values()]
    modality_score = geometric_mean_score([1.0, 0.8, 0.2], CONFIG)
    final_score = geometric_mean_score([modality_score, 0.58], CONFIG)

    assert all(0 <= percentile <= 1 for percentile in percentiles)
    assert 0 <= modality_score <= 1
    assert 0 <= final_score <= 1
