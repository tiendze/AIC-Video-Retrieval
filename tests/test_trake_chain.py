"""Unit tests for the TRAKE candidate-chaining algorithm.

Each test pins one invariant that, if broken, costs points silently rather than
raising (CLAUDE.md §3) -- ordering, the minimum gap between two events, the
weak-link objective, near-duplicate suppression, video length bias, and the
submission slot layout.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from app.services.trake_chain import (
    CandidateFrame,
    ChainCandidate,
    TrakeChainError,
    allocate_submission_slots,
    chain_sequences,
    kis_evidence_score,
    score_videos,
    sequence_nms,
    temporal_nms,
    zscore_rows,
)

CHAIN_DEFAULTS = dict(
    min_gap_seconds=1.0,
    max_gap_seconds=120.0,
    beam_width=10,
    weak_link_beta=0.3,
    coverage_threshold=0.0,
    coverage_weight=0.0,
)


def frame(
    timestamp: float,
    score: float,
    *,
    video_id: str = "L01_V001",
    frame_idx: int | None = None,
    kis_rank: int | None = None,
) -> CandidateFrame:
    index = int(timestamp * 25) if frame_idx is None else frame_idx
    return CandidateFrame(
        corpus_index=index,
        keyframe_id=f"{video_id}_{index}",
        video_id=video_id,
        frame_idx=index,
        timestamp=timestamp,
        score=score,
        semantic_z=score,
        kis_rank=kis_rank,
    )


def chain(video_id: str, score: float, frame_indices: tuple[int, ...]) -> ChainCandidate:
    frames = tuple(
        CandidateFrame(
            corpus_index=index,
            keyframe_id=f"{video_id}_{index}",
            video_id=video_id,
            frame_idx=index,
            timestamp=index / 25.0,
            score=score,
        )
        for index in frame_indices
    )
    return ChainCandidate(
        video_id=video_id,
        frames=frames,
        min_score=score,
        mean_score=score,
        coverage=1.0,
        score=score,
    )


class TestZscoreRows:
    def test_two_events_on_different_cosine_scales_become_comparable(self) -> None:
        # Event 0's wording scores high against everything; event 1's scores
        # low. Raw, event 0 dominates any cross-event sum for no real reason.
        scores = np.array([[0.30, 0.32, 0.34], [0.02, 0.04, 0.06]])
        normalized = zscore_rows(scores)
        # Same internal ordering, same relative spread -- only the offset and
        # scale differ, which is exactly what must be removed.
        assert np.allclose(normalized[0], normalized[1])

    def test_constant_row_maps_to_zeros_instead_of_dividing_by_noise(self) -> None:
        normalized = zscore_rows(np.array([[0.25, 0.25, 0.25]]))
        assert np.allclose(normalized, 0.0)

    def test_rejects_non_2d_input(self) -> None:
        with pytest.raises(TrakeChainError) as excinfo:
            zscore_rows(np.array([0.1, 0.2]))
        assert excinfo.value.error_code == "invalid_score_shape"


class TestKisEvidenceScore:
    def test_rank_zero_is_close_to_one_and_decays_with_rank(self) -> None:
        assert kis_evidence_score(0, 60) == pytest.approx(1.0)
        assert kis_evidence_score(60, 60) == pytest.approx(0.5)
        assert kis_evidence_score(0, 60) > kis_evidence_score(10, 60)

    def test_a_frame_kis_never_returned_gets_no_bonus(self) -> None:
        assert kis_evidence_score(None, 60) == 0.0


class TestTemporalNms:
    def test_suppresses_a_cluster_from_one_shot(self) -> None:
        # Five keyframes of the same shot plus one distinct later moment.
        frames = [frame(t, 1.0 - t / 100) for t in (10.0, 10.3, 10.6, 10.9, 11.2)]
        frames.append(frame(40.0, 0.5))
        kept = temporal_nms(frames, min_seconds=1.5, keep=10)
        assert [item.timestamp for item in kept] == [10.0, 40.0]

    def test_keeps_the_highest_scoring_member_of_a_cluster(self) -> None:
        frames = [frame(10.0, 0.2), frame(10.4, 0.9)]
        kept = temporal_nms(frames, min_seconds=1.5, keep=10)
        assert [item.timestamp for item in kept] == [10.4]

    def test_respects_the_keep_bound(self) -> None:
        frames = [frame(float(t * 10), 1.0) for t in range(10)]
        assert len(temporal_nms(frames, min_seconds=1.0, keep=3)) == 3


class TestChainSequences:
    def test_returned_sequences_are_strictly_time_ordered(self) -> None:
        buckets = [
            [frame(5.0, 0.9), frame(50.0, 0.95)],
            [frame(10.0, 0.9), frame(60.0, 0.4)],
        ]
        results = chain_sequences(buckets, **CHAIN_DEFAULTS)
        assert results
        for candidate in results:
            timestamps = candidate.timestamps
            assert all(a < b for a, b in zip(timestamps, timestamps[1:]))

    def test_min_gap_rejects_two_events_landing_in_the_same_shot(self) -> None:
        # Both frames score well, but 0.4s apart they are the same moment --
        # DANTE's telescoping span penalty could not express this constraint.
        buckets = [[frame(10.0, 0.9)], [frame(10.4, 0.9)]]
        assert chain_sequences(buckets, **CHAIN_DEFAULTS) == []

        relaxed = dict(CHAIN_DEFAULTS, min_gap_seconds=0.1)
        assert chain_sequences(buckets, **relaxed)

    def test_a_frame_can_never_answer_two_events_even_with_zero_min_gap(self) -> None:
        shared = frame(10.0, 0.9)
        buckets = [[shared], [shared]]
        assert chain_sequences(buckets, **dict(CHAIN_DEFAULTS, min_gap_seconds=0.0)) == []

    def test_max_gap_rejects_a_sequence_spread_across_the_whole_video(self) -> None:
        buckets = [[frame(5.0, 0.9)], [frame(900.0, 0.9)]]
        assert chain_sequences(buckets, **dict(CHAIN_DEFAULTS, max_gap_seconds=120.0)) == []
        assert chain_sequences(buckets, **dict(CHAIN_DEFAULTS, max_gap_seconds=1000.0))

    def test_weak_link_beats_a_higher_sum_with_one_hopeless_event(self) -> None:
        # The gap window leaves exactly two paths:
        #   10s -> 40s   scores (0.60, 0.62)  sum 1.22, weakest event 0.60
        #   150s -> 170s scores (1.90, 0.05)  sum 1.95, weakest event 0.05
        # A sum-maximising DP (what DANTE is forced into) picks the second.
        # TRAKE grades every event, so the second sequence is almost certainly
        # wrong on event 2 -- the first is the one worth submitting.
        buckets = [
            [frame(10.0, 0.60), frame(150.0, 1.90)],
            [frame(40.0, 0.62), frame(170.0, 0.05)],
        ]
        options = dict(CHAIN_DEFAULTS, max_gap_seconds=60.0)
        results = chain_sequences(buckets, **options)
        assert len(results) == 2

        by_sum = max(results, key=lambda item: item.mean_score)
        assert by_sum.timestamps == (150.0, 170.0)

        assert results[0].timestamps == (10.0, 40.0)
        assert results[0].min_score == pytest.approx(0.60)

    def test_empty_bucket_returns_no_sequence_rather_than_a_short_one(self) -> None:
        buckets = [[frame(10.0, 0.9)], []]
        assert chain_sequences(buckets, **CHAIN_DEFAULTS) == []

    def test_every_returned_sequence_has_one_frame_per_event(self) -> None:
        buckets = [
            [frame(5.0, 0.5), frame(6.0, 0.6)],
            [frame(20.0, 0.5), frame(21.0, 0.6)],
            [frame(40.0, 0.5), frame(41.0, 0.6)],
        ]
        for candidate in chain_sequences(buckets, **CHAIN_DEFAULTS):
            assert len(candidate.frames) == 3

    def test_coverage_counts_events_clearing_the_threshold(self) -> None:
        buckets = [[frame(10.0, 1.0)], [frame(30.0, -1.0)]]
        weighted = dict(CHAIN_DEFAULTS, coverage_threshold=0.0, coverage_weight=1.0)
        candidate = chain_sequences(buckets, **weighted)[0]
        assert candidate.coverage == pytest.approx(0.5)

    def test_rejects_an_empty_event_list(self) -> None:
        with pytest.raises(TrakeChainError) as excinfo:
            chain_sequences([], **CHAIN_DEFAULTS)
        assert excinfo.value.error_code == "no_events"

    def test_rejects_a_zero_beam(self) -> None:
        buckets = [[frame(10.0, 0.5)], [frame(20.0, 0.5)]]
        with pytest.raises(TrakeChainError) as excinfo:
            chain_sequences(buckets, **dict(CHAIN_DEFAULTS, beam_width=0))
        assert excinfo.value.error_code == "invalid_beam_width"


class TestSequenceNms:
    def test_drops_a_sequence_shifted_by_one_keyframe(self) -> None:
        candidates = [
            chain("L01_V001", 0.9, (100, 300)),
            chain("L01_V001", 0.8, (103, 302)),
        ]
        kept = sequence_nms(candidates, frame_tolerance=10)
        assert [item.score for item in kept] == [0.9]

    def test_keeps_a_genuinely_different_moment_in_the_same_video(self) -> None:
        candidates = [
            chain("L01_V001", 0.9, (100, 300)),
            chain("L01_V001", 0.8, (900, 1400)),
        ]
        assert len(sequence_nms(candidates, frame_tolerance=10)) == 2

    def test_never_suppresses_across_videos(self) -> None:
        candidates = [
            chain("L01_V001", 0.9, (100, 300)),
            chain("L02_V007", 0.8, (100, 300)),
        ]
        assert len(sequence_nms(candidates, frame_tolerance=10)) == 2


class TestScoreVideos:
    def _score(self, candidates, keyframe_counts, **overrides):
        kwargs = dict(
            keyframe_counts=keyframe_counts,
            rank_evidence={},
            top_chains=3,
            logsumexp_weight=1.0,
            coverage_weight=0.0,
            rank_evidence_weight=0.0,
            length_penalty=0.0,
            # Large enough to be a no-op for tests that aren't about this
            # mechanism -- see TestScoreVideos.test_several_good_sequences_*
            # and test_capped_bonus_* below for tests that set a realistic cap.
            smooth_max_bonus_cap=100.0,
        )
        kwargs.update(overrides)
        return score_videos(candidates, **kwargs)

    def test_length_penalty_stops_a_long_video_winning_on_size_alone(self) -> None:
        candidates = [chain("SHORT", 1.0, (10, 20)), chain("LONG", 1.02, (10, 20))]
        counts = {"SHORT": 40, "LONG": 4000}
        without = self._score(candidates, counts)
        assert without["LONG"].score > without["SHORT"].score

        with_penalty = self._score(candidates, counts, length_penalty=0.15)
        assert with_penalty["SHORT"].score > with_penalty["LONG"].score

    def test_several_good_sequences_outrank_one_lucky_one(self) -> None:
        # Same fixture as the fix's regression test below, but here the extra
        # sequences are close enough in quality to the best one that even a
        # realistic cap (configs/search/trake.yaml: video_smooth_max_bonus_cap)
        # leaves them ahead -- capping bounds the bonus, it doesn't zero it.
        candidates = [
            chain("MANY", 1.0, (10, 20)),
            chain("MANY", 0.99, (200, 400)),
            chain("MANY", 0.98, (900, 1100)),
            chain("ONE", 1.0, (10, 20)),
        ]
        scores = self._score(candidates, {"MANY": 100, "ONE": 100}, smooth_max_bonus_cap=0.3)
        assert scores["MANY"].score > scores["ONE"].score

    def test_capped_bonus_stops_many_mediocre_sequences_beating_one_good_one(self) -> None:
        # 5 mediocre-but-distinct sequences in one video vs. one clean, better
        # sequence in another -- measured 2026-08-28 from a real report of the
        # correct video ranking low. Uncapped, logsumexp of 5 tied scores adds
        # log(5) ~= 1.61 for free, enough to let MEDIOCRE win outright; capped
        # at a realistic value it can't.
        candidates = [
            chain("MEDIOCRE", 1.0, (0, 10)),
            chain("MEDIOCRE", 1.0, (100, 110)),
            chain("MEDIOCRE", 1.0, (300, 310)),
            chain("MEDIOCRE", 1.0, (500, 510)),
            chain("MEDIOCRE", 1.0, (700, 710)),
            chain("GOOD", 2.0, (10, 20)),
        ]
        counts = {"MEDIOCRE": 100, "GOOD": 100}
        uncapped = self._score(candidates, counts)
        assert uncapped["MEDIOCRE"].score > uncapped["GOOD"].score

        capped = self._score(candidates, counts, smooth_max_bonus_cap=0.3)
        assert capped["GOOD"].score > capped["MEDIOCRE"].score

    def test_rank_evidence_is_carried_through(self) -> None:
        candidates = [chain("A", 1.0, (10, 20)), chain("B", 1.0, (10, 20))]
        scores = self._score(
            candidates,
            {"A": 100, "B": 100},
            rank_evidence={"A": 2.0},
            rank_evidence_weight=1.0,
        )
        assert scores["A"].score == pytest.approx(scores["B"].score + 2.0)
        assert scores["A"].rank_evidence == pytest.approx(2.0)

    def test_reports_a_finite_smooth_max(self) -> None:
        scores = self._score([chain("A", 1.0, (10, 20))], {"A": 100})
        assert math.isfinite(scores["A"].score)


class TestAllocateSubmissionSlots:
    def _video_scores(self, candidates, **overrides):
        kwargs = dict(
            keyframe_counts={c.video_id: 100 for c in candidates},
            rank_evidence={},
            top_chains=3,
            logsumexp_weight=1.0,
            coverage_weight=0.0,
            rank_evidence_weight=0.0,
            length_penalty=0.0,
            smooth_max_bonus_cap=100.0,
        )
        kwargs.update(overrides)
        return score_videos(candidates, **kwargs)

    def test_first_slots_go_to_distinct_videos(self) -> None:
        # One video with four strong variants would otherwise take every early
        # slot -- and if that video is wrong, so is the whole top of the list.
        candidates = [
            chain("A", 1.00, (10, 20)),
            chain("A", 0.99, (500, 900)),
            chain("A", 0.98, (1500, 1900)),
            chain("B", 0.50, (10, 20)),
            chain("C", 0.40, (10, 20)),
        ]
        allocated = allocate_submission_slots(
            candidates,
            self._video_scores(candidates),
            max_sequences_per_video=5,
            limit=3,
        )
        assert [item.video_id for item in allocated] == ["A", "B", "C"]

    def test_later_rounds_add_frame_variants(self) -> None:
        candidates = [
            chain("A", 1.00, (10, 20)),
            chain("A", 0.99, (500, 900)),
            chain("B", 0.50, (10, 20)),
        ]
        allocated = allocate_submission_slots(
            candidates,
            self._video_scores(candidates),
            max_sequences_per_video=5,
            limit=10,
        )
        assert [item.video_id for item in allocated] == ["A", "B", "A"]

    def test_caps_sequences_per_video(self) -> None:
        candidates = [chain("A", 1.0 - i / 100, (i * 100, i * 100 + 50)) for i in range(10)]
        allocated = allocate_submission_slots(
            candidates,
            self._video_scores(candidates),
            max_sequences_per_video=3,
            limit=100,
        )
        assert len(allocated) == 3

    def test_respects_the_submission_limit(self) -> None:
        candidates = [chain(f"V{i:03d}", 1.0 - i / 1000, (10, 20)) for i in range(200)]
        allocated = allocate_submission_slots(
            candidates,
            self._video_scores(candidates),
            max_sequences_per_video=5,
            limit=100,
        )
        assert len(allocated) == 100

    def test_zero_limit_returns_nothing(self) -> None:
        candidates = [chain("A", 1.0, (10, 20))]
        assert (
            allocate_submission_slots(
                candidates,
                self._video_scores(candidates),
                max_sequences_per_video=5,
                limit=0,
            )
            == []
        )
