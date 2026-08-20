"""Behaviour lock for app/services/dante_service.py (plan §P2.1).

The DP recurrence (Eq 1-5, PRD §F6) has two invariants that are easy to
silently break while "optimizing": the running-max trick must still enforce
`tau < t` (CLAUDE.md §3 -- "DANTE giu rang buoc thu tu tau < t"), and the
O(N*T^2) naive DP collapsed into O(N*T) must actually behave that way, not
just look that way on paper (plan: "kiem chung thuc nghiem tuyen bo O(NT) ma
paper khong chung minh bang so").
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from app.services.dante_service import (
    DEFAULT_LAMBDA_PENALTY,
    LAMBDA_TIME,
    DanteResult,
    DanteServiceError,
    compute_similarity_matrix,
    dante_search,
    dante_search_sequences,
    temporal_gap_seconds,
)


def _similarity(rows: list[list[float]]) -> np.ndarray:
    return np.array(rows, dtype=np.float64)


def _timestamps_for(similarity: np.ndarray) -> np.ndarray:
    return np.arange(similarity.shape[1], dtype=np.float64)


class TestDanteNeverReturnsOutOfOrderSequence:
    def test_ignores_a_similarity_matrix_that_baits_a_reversed_pick(self) -> None:
        # Event 0's best match is late in the video (t=5), event 2's best
        # match is early (t=0) -- a naive per-event argmax would pick
        # t=(5, ?, 0), which is not a valid ordered sequence at all.
        similarity = _similarity(
            [
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.9],  # event 0: peak at t=5
                [0.1, 0.1, 0.9, 0.1, 0.1, 0.1],  # event 1: peak at t=2
                [0.9, 0.1, 0.1, 0.1, 0.1, 0.1],  # event 2: peak at t=0
            ]
        )

        result = dante_search(
            similarity,
            s_v=100,
            timestamps=_timestamps_for(similarity),
            lambda_penalty=0.0,
        )

        indices = result.global_indices
        assert list(indices) == sorted(indices), "sequence must be strictly increasing"
        assert len(set(indices)) == len(indices), "no event may reuse another's frame"

    def test_never_returns_out_of_order_across_many_random_matrices(self) -> None:
        rng = np.random.default_rng(42)
        for _seed in range(20):
            similarity = rng.standard_normal((4, 12))
            result = dante_search(
                similarity,
                s_v=0,
                timestamps=_timestamps_for(similarity),
                lambda_penalty=0.005,
            )
            indices = result.global_indices
            assert list(indices) == sorted(indices)
            assert len(set(indices)) == len(indices)


class TestDanteReturnsExactlyNKeyframes:
    @pytest.mark.parametrize("n_events", [1, 2, 4, 7])
    def test_backtracking_yields_exactly_n_indices(self, n_events: int) -> None:
        rng = np.random.default_rng(n_events)
        similarity = rng.standard_normal((n_events, 15))

        result = dante_search(
            similarity,
            s_v=0,
            timestamps=_timestamps_for(similarity),
            lambda_penalty=0.001,
        )

        assert len(result.global_indices) == n_events

    def test_indices_are_offset_by_s_v(self) -> None:
        similarity = _similarity([[0.5, 0.1, 0.2], [0.1, 0.5, 0.9]])

        result = dante_search(
            similarity,
            s_v=1000,
            timestamps=_timestamps_for(similarity),
            lambda_penalty=0.0,
        )

        assert all(1000 <= idx <= 1002 for idx in result.global_indices)


class TestDanteRuntimeIsLinearInT:
    def test_doubling_t_roughly_doubles_runtime_not_quadruples(self) -> None:
        n_events = 5
        base_t = 3000

        # Warm up so import/interpreter overhead on the first call doesn't
        # skew the smaller of the two timed samples below.
        _time_dante_search(n_events, 200, lambda_penalty=0.001, seed=0)

        time_1x = min(_time_dante_search(n_events, base_t, 0.001, seed) for seed in range(3))
        time_2x = min(_time_dante_search(n_events, base_t * 2, 0.001, seed) for seed in range(3))

        ratio = time_2x / time_1x
        # O(NT) predicts ~2x; O(NT^2) would predict ~4x. Floor is low (not
        # ~2x) because under full-suite contention this measured as low as
        # 1.15 with no code change -- the floor only needs to rule out
        # "runtime independent of T", the ceiling is what actually catches a
        # quadratic regression.
        assert 1.05 < ratio < 3.0, f"expected ~2x scaling for 2x frames, got ratio={ratio:.2f}"


def _time_dante_search(n_events: int, n_frames: int, lambda_penalty: float, seed: int) -> float:
    rng = np.random.default_rng(seed)
    similarity = rng.standard_normal((n_events, n_frames))
    start = time.perf_counter()
    dante_search(
        similarity,
        s_v=0,
        timestamps=_timestamps_for(similarity),
        lambda_penalty=lambda_penalty,
    )
    return time.perf_counter() - start


class TestLambdaZeroReducesToPureOrderConstraint:
    def test_matches_brute_force_best_ordered_pair_when_lambda_is_zero(self) -> None:
        similarity = _similarity(
            [
                [0.1, 0.9, 0.2, 0.05],
                [0.05, 0.1, 0.8, 0.95],
            ]
        )

        result = dante_search(
            similarity,
            s_v=0,
            timestamps=_timestamps_for(similarity),
            lambda_penalty=0.0,
        )

        expected_score, expected_pair = _brute_force_best_ordered_pair(similarity)
        assert result.score == pytest.approx(expected_score)
        assert tuple(result.global_indices) == expected_pair

    def test_lambda_zero_ignores_distance_between_chosen_frames(self) -> None:
        # Event 0 peaks far from event 1's peak; a lambda > 0 would prefer a
        # closer, lower-similarity pair instead. At lambda=0 the algorithm
        # must take the raw best pair regardless of how far apart it is.
        similarity = _similarity(
            [
                [0.9, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.9],
            ]
        )

        result = dante_search(
            similarity,
            s_v=0,
            timestamps=_timestamps_for(similarity),
            lambda_penalty=0.0,
        )

        assert tuple(result.global_indices) == (0, 5)
        assert result.score == pytest.approx(1.8)


class TestTimestampGapPenalty:
    def test_uses_timestamp_gap_not_keyframe_index_gap(self) -> None:
        similarity = _similarity(
            [
                [1.0, float("-inf")],
                [float("-inf"), 1.0],
            ]
        )
        timestamps = np.array([10.0, 20.0], dtype=np.float64)

        result = dante_search(
            similarity,
            s_v=100,
            timestamps=timestamps,
            lambda_penalty=LAMBDA_TIME,
        )

        assert result.global_indices == (100, 101)
        assert result.score == pytest.approx(1.9)
        assert temporal_gap_seconds(10.0, 20.0) == pytest.approx(10.0)
        assert result.debug_steps[1].gap_seconds == pytest.approx(10.0)
        assert result.debug_steps[1].temporal_penalty == pytest.approx(0.1)

    def test_rejects_reverse_timestamp_order(self) -> None:
        similarity = _similarity(
            [
                [float("-inf"), 1.0],
                [1.0, float("-inf")],
            ]
        )
        timestamps = np.array([10.0, 20.0], dtype=np.float64)

        with pytest.raises(DanteServiceError, match="no valid strictly-ordered sequence"):
            dante_search(
                similarity,
                s_v=0,
                timestamps=timestamps,
                lambda_penalty=0.0,
            )

    def test_semantic_scores_are_preserved_in_debug_steps(self) -> None:
        similarity = _similarity(
            [
                [0.42, float("-inf")],
                [float("-inf"), 0.45],
            ]
        )
        timestamps = np.array([10.0, 13.0], dtype=np.float64)

        result = dante_search(
            similarity,
            s_v=0,
            timestamps=timestamps,
            lambda_penalty=LAMBDA_TIME,
        )

        assert [step.semantic_score for step in result.debug_steps] == pytest.approx([0.42, 0.45])
        assert result.score == pytest.approx(0.42 + 0.45 - 0.01 * 3.0)


def _brute_force_best_ordered_pair(similarity: np.ndarray) -> tuple[float, tuple[int, int]]:
    n_frames = similarity.shape[1]
    best_score = float("-inf")
    best_pair = (-1, -1)
    for tau in range(n_frames):
        for t in range(tau + 1, n_frames):
            score = similarity[0, tau] + similarity[1, t]
            if score > best_score:
                best_score = score
                best_pair = (tau, t)
    return best_score, best_pair


class TestDanteSearchSequences:
    def test_returns_multiple_ordered_sequences_at_or_above_threshold(self) -> None:
        similarity = _similarity(
            [
                [0.9, 0.8, 0.1],
                [0.1, 0.9, 0.7],
            ]
        )

        results = dante_search_sequences(
            similarity,
            s_v=100,
            threshold=1.55,
            top_k=10,
            timestamps=_timestamps_for(similarity),
            lambda_penalty=0.0,
        )

        assert [(result.score, result.global_indices) for result in results] == [
            (pytest.approx(1.8), (100, 101)),
            (pytest.approx(1.6), (100, 102)),
        ]

    def test_limits_sequence_candidates_to_top_k_after_sorting_by_score(self) -> None:
        similarity = _similarity(
            [
                [0.9, 0.8, 0.7, 0.1],
                [0.1, 0.9, 0.8, 0.7],
            ]
        )

        results = dante_search_sequences(
            similarity,
            s_v=0,
            threshold=0.0,
            top_k=2,
            timestamps=_timestamps_for(similarity),
            lambda_penalty=0.0,
        )

        assert [result.global_indices for result in results] == [(0, 1), (0, 2)]
        assert [result.score for result in results] == pytest.approx([1.8, 1.7])

    def test_returns_an_empty_list_when_sequences_are_finite_but_below_threshold(self) -> None:
        similarity = _similarity([[0.4, 0.3], [0.2, 0.4]])

        results = dante_search_sequences(
            similarity,
            s_v=0,
            threshold=10.0,
            top_k=10,
            timestamps=_timestamps_for(similarity),
            lambda_penalty=0.0,
        )

        assert results == []


class TestDanteSearchErrors:
    def test_raises_when_fewer_frames_than_events_are_available(self) -> None:
        similarity = _similarity([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])  # 3 events, 2 frames

        with pytest.raises(DanteServiceError, match="not enough frames") as exc_info:
            dante_search(
                similarity,
                s_v=0,
                timestamps=_timestamps_for(similarity),
                lambda_penalty=0.001,
            )

        # error_code/status_code mirror SearchServiceError (app/services/
        # search_service.py) so a future POST /api/trake can map this to an
        # HTTP response without parsing the message text (CLAUDE.md §7).
        assert exc_info.value.error_code == "insufficient_frames"
        assert exc_info.value.status_code == 422

    def test_raises_on_a_1d_similarity_array(self) -> None:
        similarity = np.array([0.1, 0.2, 0.3])

        with pytest.raises(DanteServiceError, match="2D") as exc_info:
            dante_search(
                similarity,
                s_v=0,
                timestamps=np.array([0.0, 1.0, 2.0]),
                lambda_penalty=0.001,
            )

        assert exc_info.value.error_code == "invalid_similarity_shape"
        assert exc_info.value.status_code == 400

    def test_raises_on_zero_events_instead_of_crashing(self) -> None:
        # Regression: an empty (0, T) similarity matrix used to index dp[0, :]
        # on a 0-row array and crash with a raw IndexError instead of the
        # documented DanteServiceError (CLAUDE.md §7 -- missing/malformed
        # input must fail loudly through the module's own error type, not an
        # unrelated exception a caller has no reason to expect or catch).
        similarity = np.empty((0, 5))

        with pytest.raises(DanteServiceError, match="at least one event") as exc_info:
            dante_search(
                similarity,
                s_v=0,
                timestamps=_timestamps_for(similarity),
                lambda_penalty=0.001,
            )

        assert exc_info.value.error_code == "invalid_similarity_shape"
        assert exc_info.value.status_code == 400

    def test_raises_when_every_forced_path_is_non_finite(self) -> None:
        # T == N leaves exactly one possible index assignment (0, 1, ..., N-1).
        # If a caller's similarity matrix has -inf along that forced
        # diagonal (e.g. an upstream NaN/-inf leaking from a buggy encoder),
        # no finite ordered sequence exists at all -- this must fail loudly
        # rather than return a DanteResult with a -inf score.
        similarity = _similarity([[float("-inf"), 0.5], [0.5, float("-inf")]])

        with pytest.raises(
            DanteServiceError, match="no valid strictly-ordered sequence"
        ) as exc_info:
            dante_search(
                similarity,
                s_v=0,
                timestamps=_timestamps_for(similarity),
                lambda_penalty=0.001,
            )

        assert exc_info.value.error_code == "no_valid_sequence"
        assert exc_info.value.status_code == 422


class TestComputeSimilarityMatrix:
    def test_returns_cosine_similarity_for_l2_normalized_vectors(self) -> None:
        event_vectors = np.array([[1.0, 0.0], [0.0, 1.0]])
        frame_vectors = np.array([[1.0, 0.0], [0.0, 1.0], [0.70710678, 0.70710678]])

        similarity = compute_similarity_matrix(event_vectors, frame_vectors)

        assert similarity.shape == (2, 3)
        assert similarity[0, 0] == pytest.approx(1.0)
        assert similarity[0, 1] == pytest.approx(0.0)
        assert similarity[1, 2] == pytest.approx(0.70710678, rel=1e-4)


class TestDefaultLambda:
    def test_lambda_time_is_the_fixed_timestamp_penalty(self) -> None:
        assert LAMBDA_TIME == pytest.approx(0.01)
        assert DEFAULT_LAMBDA_PENALTY == pytest.approx(LAMBDA_TIME)

    def test_dante_search_uses_default_lambda_when_not_passed(self) -> None:
        similarity = _similarity([[0.5, 0.1, 0.2], [0.1, 0.5, 0.9]])

        with_default = dante_search(
            similarity,
            s_v=0,
            timestamps=_timestamps_for(similarity),
        )
        with_explicit = dante_search(
            similarity,
            s_v=0,
            timestamps=_timestamps_for(similarity),
            lambda_penalty=DEFAULT_LAMBDA_PENALTY,
        )

        assert with_default == with_explicit


class TestDanteResultIsImmutable:
    def test_dante_result_is_a_frozen_dataclass(self) -> None:
        result = DanteResult(score=1.0, global_indices=(0, 1))

        with pytest.raises(AttributeError):
            result.score = 2.0  # type: ignore[misc]
