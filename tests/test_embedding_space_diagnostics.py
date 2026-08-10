"""Behaviour lock for eval/diagnostics/embedding_space.py.

The diagnostic exists to answer one question -- "is the semantic channel
losing to random because the embedding space is degenerate?" -- so these
tests pin the two answers that matter: a healthy space must NOT be flagged
(no false alarm that sends the team re-embedding for nothing), and a
collapsed space MUST be flagged with evidence naming which signal fired.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from eval.diagnostics.embedding_space import (
    CollapseThresholds,
    assess_collapse,
    pairwise_cosine_stats,
    render_report_md,
    result_overlap_stats,
    score_spread_stats,
)

_THRESHOLDS = CollapseThresholds(
    query_cosine_mean=0.80,
    result_jaccard_mean=0.30,
    union_ratio=0.25,
    score_gap=0.05,
    min_signals=2,
)


def _unit_rows(rows) -> np.ndarray:
    array = np.asarray(rows, dtype=np.float32)
    return array / np.linalg.norm(array, axis=1, keepdims=True)


class TestPairwiseCosineStats:
    def test_identical_vectors_have_cosine_one(self) -> None:
        vectors = _unit_rows([[1.0, 0.0, 0.0]] * 4)

        stats = pairwise_cosine_stats(vectors)

        assert stats.mean == pytest.approx(1.0)
        assert stats.minimum == pytest.approx(1.0)

    def test_orthogonal_vectors_have_cosine_zero(self) -> None:
        vectors = _unit_rows([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        stats = pairwise_cosine_stats(vectors)

        assert stats.mean == pytest.approx(0.0, abs=1e-6)
        assert stats.maximum == pytest.approx(0.0, abs=1e-6)

    def test_counts_unordered_pairs_only(self) -> None:
        vectors = _unit_rows([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, -1.0]])

        stats = pairwise_cosine_stats(vectors)

        assert stats.num_vectors == 4
        assert stats.num_pairs == 6  # C(4,2), not 4*4 and not 4*3

    def test_percentiles_bracket_the_median(self) -> None:
        rng = np.random.default_rng(20260810)
        vectors = _unit_rows(rng.normal(size=(20, 8)))

        stats = pairwise_cosine_stats(vectors)

        assert stats.minimum <= stats.p05 <= stats.median <= stats.p95 <= stats.maximum

    def test_rejects_a_single_vector_instead_of_returning_nan(self) -> None:
        with pytest.raises(ValueError, match="at least 2 vectors"):
            pairwise_cosine_stats(_unit_rows([[1.0, 0.0]]))

    def test_rejects_non_2d_input(self) -> None:
        with pytest.raises(ValueError, match="2-D"):
            pairwise_cosine_stats(np.zeros((2, 3, 4), dtype=np.float32))


class TestResultOverlapStats:
    def test_identical_result_sets_have_jaccard_one(self) -> None:
        stats = result_overlap_stats([["a", "b", "c"], ["a", "b", "c"], ["a", "b", "c"]])

        assert stats.mean_jaccard == pytest.approx(1.0)
        assert stats.union_size == 3

    def test_disjoint_result_sets_have_jaccard_zero(self) -> None:
        stats = result_overlap_stats([["a", "b"], ["c", "d"], ["e", "f"]])

        assert stats.mean_jaccard == pytest.approx(0.0)
        assert stats.union_size == 6

    def test_union_ratio_is_distinct_over_max_possible(self) -> None:
        # 3 queries x top_k 2 = 6 possible, but only 4 distinct ids reached.
        stats = result_overlap_stats([["a", "b"], ["b", "c"], ["c", "d"]])

        assert stats.max_possible_union == 6
        assert stats.union_size == 4
        assert stats.union_ratio == pytest.approx(4 / 6)

    def test_dedupes_repeated_ids_inside_one_result_set(self) -> None:
        stats = result_overlap_stats([["a", "a", "b"], ["a", "b"]])

        assert stats.mean_jaccard == pytest.approx(1.0)
        assert stats.union_size == 2

    def test_two_empty_result_sets_score_zero_not_one(self) -> None:
        # Empty-vs-empty is trivially "identical"; scoring it 1.0 would invent
        # collapse evidence out of a store that simply returned nothing.
        stats = result_overlap_stats([[], []])

        assert stats.mean_jaccard == pytest.approx(0.0)
        assert stats.union_ratio == pytest.approx(0.0)

    def test_rejects_a_single_result_set(self) -> None:
        with pytest.raises(ValueError, match="at least 2 result sets"):
            result_overlap_stats([["a"]])


class TestScoreSpreadStats:
    def test_gap_is_first_minus_last_score(self) -> None:
        stats = score_spread_stats([[0.9, 0.5, 0.1], [0.8, 0.4, 0.2]])

        assert stats.mean_top1 == pytest.approx(0.85)
        assert stats.mean_last == pytest.approx(0.15)
        assert stats.mean_gap == pytest.approx(0.70)

    def test_skips_empty_result_lists(self) -> None:
        stats = score_spread_stats([[0.9, 0.1], [], [0.7, 0.3]])

        assert stats.num_queries == 2
        assert stats.mean_gap == pytest.approx(0.6)

    def test_raises_when_no_query_returned_anything(self) -> None:
        with pytest.raises(ValueError, match="nothing to measure"):
            score_spread_stats([[], []])


class TestAssessCollapse:
    def _healthy(self):
        cosine = pairwise_cosine_stats(
            _unit_rows([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        )
        overlap = result_overlap_stats([["a", "b"], ["c", "d"], ["e", "f"]])
        spread = score_spread_stats([[0.9, 0.1], [0.8, 0.2]])
        return cosine, overlap, spread

    def _collapsed(self):
        cosine = pairwise_cosine_stats(_unit_rows([[1.0, 0.01], [1.0, 0.02], [1.0, 0.03]]))
        overlap = result_overlap_stats([["a", "b"], ["a", "b"], ["a", "b"]])
        spread = score_spread_stats([[0.71, 0.70], [0.72, 0.71]])
        return cosine, overlap, spread

    def test_healthy_space_is_not_flagged(self) -> None:
        cosine, overlap, spread = self._healthy()

        verdict = assess_collapse(cosine, overlap, spread, _THRESHOLDS)

        assert verdict.collapsed is False
        assert verdict.signals == ()

    def test_collapsed_space_is_flagged_with_evidence(self) -> None:
        cosine, overlap, spread = self._collapsed()

        verdict = assess_collapse(cosine, overlap, spread, _THRESHOLDS)

        assert verdict.collapsed is True
        assert len(verdict.signals) >= _THRESHOLDS.min_signals
        assert any("query_cosine_mean" in s for s in verdict.signals)
        assert any("result_jaccard_mean" in s for s in verdict.signals)

    def test_one_signal_alone_does_not_flag_collapse(self) -> None:
        # High result overlap alone is expected of a near-duplicate news
        # corpus -- it must not be enough on its own.
        cosine, _, spread = self._healthy()
        overlap = result_overlap_stats([["a", "b"], ["a", "b"], ["a", "b"]])

        verdict = assess_collapse(cosine, overlap, spread, _THRESHOLDS)

        assert verdict.collapsed is False
        assert len(verdict.signals) == 1

    def test_min_signals_is_configurable_not_hardcoded(self) -> None:
        cosine, _, spread = self._healthy()
        overlap = result_overlap_stats([["a", "b"], ["a", "b"], ["a", "b"]])
        strict = CollapseThresholds(
            query_cosine_mean=_THRESHOLDS.query_cosine_mean,
            result_jaccard_mean=_THRESHOLDS.result_jaccard_mean,
            union_ratio=_THRESHOLDS.union_ratio,
            score_gap=_THRESHOLDS.score_gap,
            min_signals=1,
        )

        assert assess_collapse(cosine, overlap, spread, strict).collapsed is True


class TestRenderReportMd:
    def _render(self, cosine, overlap, spread, verdict) -> str:
        return render_report_md(
            label="vi",
            git_sha="deadbee",
            model_id="beit3_base_itc_patch16_224",
            collection_name="keyframes_beit3_base_itc_patch16_224",
            num_queries=3,
            cosine=cosine,
            overlap=overlap,
            spread=spread,
            verdict=verdict,
        )

    def test_report_states_verdict_and_names_each_signal(self) -> None:
        cosine = pairwise_cosine_stats(_unit_rows([[1.0, 0.01], [1.0, 0.02], [1.0, 0.03]]))
        overlap = result_overlap_stats([["a", "b"], ["a", "b"], ["a", "b"]])
        spread = score_spread_stats([[0.71, 0.70]])
        verdict = assess_collapse(cosine, overlap, spread, _THRESHOLDS)

        report = self._render(cosine, overlap, spread, verdict)

        assert "collapsed: **True**" in report
        assert "query_cosine_mean" in report
        assert "keyframes_beit3_base_itc_patch16_224" in report

    def test_report_says_so_explicitly_when_no_signal_fired(self) -> None:
        cosine = pairwise_cosine_stats(
            _unit_rows([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        )
        overlap = result_overlap_stats([["a", "b"], ["c", "d"], ["e", "f"]])
        spread = score_spread_stats([[0.9, 0.1], [0.8, 0.2]])
        verdict = assess_collapse(cosine, overlap, spread, _THRESHOLDS)

        report = self._render(cosine, overlap, spread, verdict)

        assert "collapsed: **False**" in report
        assert "no signal fired" in report

    def test_every_rendered_number_is_finite(self) -> None:
        cosine = pairwise_cosine_stats(_unit_rows([[1.0, 0.0], [0.0, 1.0]]))
        overlap = result_overlap_stats([["a"], ["b"]])
        spread = score_spread_stats([[0.5]])

        assert all(
            math.isfinite(v)
            for v in (cosine.mean, overlap.mean_jaccard, spread.mean_gap, overlap.union_ratio)
        )
