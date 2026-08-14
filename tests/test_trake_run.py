"""RED reproducer for eval/trake_run.py (plan §P2.3) -- TRAKE evaluation
orchestration kept as its own module (eval/run.py is already past CLAUDE.md
§6's 800-line ceiling; chủ dự án 2026-08-14 confirmed splitting rather than
growing eval/run.py further). Fakes stand in for TrakeService/ground truth --
no Docker/GPU here, mirroring tests/test_eval_run.py's own pattern; the real
`python -m eval.run --config configs/eval/trake_val.yaml` run is a manual
step (CLAUDE.md §10).
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from datasets.base import GroundTruthSegment
from eval.trake_run import (
    TrakeGroupMetrics,
    aggregate_trake_group,
    render_trake_report_md,
    run_trake_eval,
    run_trake_queries,
)


def _segment(
    video_id: str, order_index: int, start: float, end: float, caption: str
) -> GroundTruthSegment:
    return GroundTruthSegment(
        video_id=video_id, start_sec=start, end_sec=end, order_index=order_index, caption=caption
    )


# Three consecutive shots from the same video -- one valid N=3 TRAKE sequence.
_SEQUENCE_A = [
    _segment("L21_V001", 0, 0.0, 2.0, "su kien 0"),
    _segment("L21_V001", 1, 10.0, 12.0, "su kien 1"),
    _segment("L21_V001", 2, 20.0, 22.0, "su kien 2"),
]


@dataclass(frozen=True)
class _FakeKeyframe:
    timestamp: float


@dataclass(frozen=True)
class _FakeVideoResult:
    video_id: str
    keyframes: tuple[_FakeKeyframe, ...]


class _FakeSearcher:
    """Stands in for TrakeService -- same `search(events, top_k,
    lambda_penalty)` shape."""

    def __init__(self, results_by_call: list[list[_FakeVideoResult]]) -> None:
        self._results_by_call = results_by_call
        self.calls: list[tuple[list[str], int, float | None]] = []

    async def search(self, events, top_k, lambda_penalty=None):
        self.calls.append((list(events), top_k, lambda_penalty))
        return self._results_by_call[len(self.calls) - 1]


def _perfect_hit_result() -> list[_FakeVideoResult]:
    # Exactly matches _SEQUENCE_A's own segment centers -- a perfect hit.
    return [
        _FakeVideoResult(
            video_id="L21_V001",
            keyframes=(_FakeKeyframe(1.0), _FakeKeyframe(11.0), _FakeKeyframe(21.0)),
        )
    ]


def _miss_result() -> list[_FakeVideoResult]:
    return [_FakeVideoResult(video_id="L21_V999", keyframes=(_FakeKeyframe(0.0),) * 3)]


# --------------------------------------------------------------------------
# run_trake_queries
# --------------------------------------------------------------------------


async def test_run_trake_queries_sends_captions_in_order_as_the_events_list():
    searcher = _FakeSearcher([_perfect_hit_result()])

    await run_trake_queries(searcher, [_SEQUENCE_A], top_k=10, lambda_penalty=None)

    assert searcher.calls == [(["su kien 0", "su kien 1", "su kien 2"], 10, None)]


async def test_run_trake_queries_scores_a_perfect_hit_as_found_with_all_events_true():
    searcher = _FakeSearcher([_perfect_hit_result()])

    results = await run_trake_queries(searcher, [_SEQUENCE_A], top_k=10, lambda_penalty=None)

    assert len(results) == 1
    assert results[0].video_rank == 1
    assert results[0].per_event_hit == (True, True, True)


async def test_run_trake_queries_scores_a_miss_as_not_found():
    searcher = _FakeSearcher([_miss_result()])

    results = await run_trake_queries(searcher, [_SEQUENCE_A], top_k=10, lambda_penalty=None)

    assert results[0].video_rank is None


async def test_run_trake_queries_passes_an_explicit_lambda_penalty_through():
    searcher = _FakeSearcher([_perfect_hit_result()])

    await run_trake_queries(searcher, [_SEQUENCE_A], top_k=10, lambda_penalty=0.01)

    assert searcher.calls[0][2] == 0.01


# --------------------------------------------------------------------------
# aggregate_trake_group
# --------------------------------------------------------------------------


async def test_aggregate_trake_group_reports_sequence_exact_match_and_wall_clock():
    searcher = _FakeSearcher([_perfect_hit_result()])
    results = await run_trake_queries(searcher, [_SEQUENCE_A], top_k=10, lambda_penalty=None)

    group = aggregate_trake_group(sequence_length=3, results=results, top_k=10, wall_clock_sec=1.5)

    assert group.sequence_length == 3
    assert group.num_queries == 1
    assert group.num_found == 1
    assert group.sequence_exact_match == pytest.approx(1.0)
    assert group.video_level_recall_at_k == pytest.approx(1.0)
    assert group.per_event_recall_at_k == (1.0, 1.0, 1.0)
    assert group.mean_kendall_tau == pytest.approx(1.0)
    assert group.mean_offset_sec == pytest.approx(0.0)
    assert group.wall_clock_sec == 1.5


async def test_aggregate_trake_group_reports_none_for_tau_and_offset_when_nothing_found():
    searcher = _FakeSearcher([_miss_result()])
    results = await run_trake_queries(searcher, [_SEQUENCE_A], top_k=10, lambda_penalty=None)

    group = aggregate_trake_group(sequence_length=3, results=results, top_k=10, wall_clock_sec=0.1)

    assert group.num_found == 0
    assert group.sequence_exact_match == pytest.approx(0.0)
    assert group.mean_kendall_tau is None
    assert group.mean_offset_sec is None


# --------------------------------------------------------------------------
# run_trake_eval -- extracts sequences per N, runs + aggregates each
# --------------------------------------------------------------------------


async def test_run_trake_eval_produces_one_group_per_sequence_length():
    # ground_truth has exactly 3 consecutive shots -- only N=3 is extractable.
    searcher = _FakeSearcher([_perfect_hit_result()])

    groups = await run_trake_eval(
        searcher,
        ground_truth=_SEQUENCE_A,
        sequence_lengths=[3],
        top_k=10,
        lambda_penalty=None,
    )

    assert [g.sequence_length for g in groups] == [3]
    assert groups[0].num_queries == 1


async def test_run_trake_eval_raises_when_a_sequence_length_has_no_extractable_sequence():
    searcher = _FakeSearcher([_perfect_hit_result()])

    with pytest.raises(ValueError, match="N=5"):
        await run_trake_eval(
            searcher,
            ground_truth=_SEQUENCE_A,  # only 3 consecutive shots available
            sequence_lengths=[5],
            top_k=10,
            lambda_penalty=None,
        )


async def test_run_trake_eval_fails_fast_before_running_any_query_when_a_later_n_is_unextractable():
    # Regression (chủ dự án 2026-08-14, real run): N=3 and N=5 succeeded
    # (256s + 76s of real DANTE search) before N=7 crashed at the END of the
    # loop with nothing written to disk -- all of that work was wasted.
    # Every sequence_length must be validated BEFORE any searcher.search()
    # call runs, so a data gap is caught in milliseconds, not minutes.
    searcher = _FakeSearcher([_perfect_hit_result(), _perfect_hit_result()])

    with pytest.raises(ValueError, match="N=5"):
        await run_trake_eval(
            searcher,
            ground_truth=_SEQUENCE_A,  # only 3 consecutive shots -- N=5 unextractable
            sequence_lengths=[3, 5],
            top_k=10,
            lambda_penalty=None,
        )

    assert searcher.calls == []


# --------------------------------------------------------------------------
# render_trake_report_md
# --------------------------------------------------------------------------


def test_render_trake_report_md_includes_every_groups_headline_numbers():
    groups = [
        TrakeGroupMetrics(
            sequence_length=3,
            num_queries=10,
            num_found=8,
            sequence_exact_match=0.25,
            video_level_recall_at_k=0.8,
            per_event_recall_at_k=(0.9, 0.7, 0.6),
            mean_kendall_tau=0.75,
            mean_offset_sec=1.2,
            wall_clock_sec=3.4,
        )
    ]

    report = render_trake_report_md(groups)

    assert "0.2500" in report  # sequence_exact_match
    assert "0.7500" in report  # mean_kendall_tau
    assert "3" in report  # sequence_length column


def test_render_trake_report_md_shows_na_when_nothing_was_found():
    groups = [
        TrakeGroupMetrics(
            sequence_length=3,
            num_queries=5,
            num_found=0,
            sequence_exact_match=0.0,
            video_level_recall_at_k=0.0,
            per_event_recall_at_k=(0.0, 0.0, 0.0),
            mean_kendall_tau=None,
            mean_offset_sec=None,
            wall_clock_sec=0.5,
        )
    ]

    report = render_trake_report_md(groups)

    assert "n/a" in report
