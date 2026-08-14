"""TRAKE evaluation orchestration for `eval/run.py` (plan §P2.3, PRD §6.4).

Kept as its own module -- eval/run.py is already past CLAUDE.md §6's
800-line file ceiling (chủ dự án 2026-08-14: split rather than grow it
further). `eval/run.py` stays the single CLI entry point (CLAUDE.md §6);
this module supplies the functions it calls when `score_trake: true`, the
same separation `eval/baselines/*.py` and `eval/metrics/*.py` already use.

One TRAKE query = one N-consecutive-shot sequence
(`eval.build_eval_set.extract_trake_sequences`, already built in P0.4). The
sequence's captions become the ordered `events` sent to `TrakeService`
(or a Fake with the same shape, in tests); the sequence's own
`start_sec`/`end_sec`/`order_index` become the `eval.metrics.trake` ground
truth `score_query` scores against.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from datasets.base import GroundTruthSegment
from eval.build_eval_set import extract_trake_sequences
from eval.metrics.trake import (
    TrakeGroundTruthEvent,
    TrakeGroundTruthQuery,
    TrakeQueryResult,
    TrakeRetrievedVideo,
    mean_kendall_tau,
    mean_offset_sec,
    per_event_recall_at_k,
    score_query,
    sequence_exact_match_rate,
    video_level_recall_at_k,
)

logger = logging.getLogger(__name__)


class TrakeSearcher(Protocol):
    """`TrakeService.search`'s shape -- a Fake in tests, the real service in
    `eval/run.py`."""

    async def search(
        self, events: Sequence[str], top_k: int, lambda_penalty: float | None = None
    ) -> Sequence[Any]: ...


@dataclass(frozen=True)
class TrakeGroupMetrics:
    """One N-value's aggregated TRAKE metrics -- PRD §6.4's required table,
    one row per N in the mandatory N ∈ {3, 5, 7} breakdown."""

    sequence_length: int
    num_queries: int
    num_found: int
    sequence_exact_match: float
    video_level_recall_at_k: float
    per_event_recall_at_k: tuple[float, ...]  # length == sequence_length
    mean_kendall_tau: float | None  # None when nothing was found this group
    mean_offset_sec: float | None
    wall_clock_sec: float


def _to_ground_truth_query(sequence: Sequence[GroundTruthSegment]) -> TrakeGroundTruthQuery:
    return TrakeGroundTruthQuery(
        video_id=sequence[0].video_id,
        events=tuple(
            TrakeGroundTruthEvent(start_sec=s.start_sec, end_sec=s.end_sec) for s in sequence
        ),
    )


def _to_retrieved(raw_results: Sequence[Any]) -> list[TrakeRetrievedVideo]:
    return [
        TrakeRetrievedVideo(
            video_id=r.video_id, keyframe_timestamps=tuple(kf.timestamp for kf in r.keyframes)
        )
        for r in raw_results
    ]


async def run_trake_queries(
    searcher: TrakeSearcher,
    sequences: Sequence[Sequence[GroundTruthSegment]],
    top_k: int,
    lambda_penalty: float | None,
) -> list[TrakeQueryResult]:
    """Runs every TRAKE query sequence through `searcher.search()` (one
    sequence's captions -> its `events` list, in `order_index` order) and
    scores each with `eval.metrics.trake.score_query`."""
    results: list[TrakeQueryResult] = []
    for sequence in sequences:
        gt_query = _to_ground_truth_query(sequence)
        events = [segment.caption for segment in sequence]
        raw_results = await searcher.search(events, top_k=top_k, lambda_penalty=lambda_penalty)
        results.append(score_query(_to_retrieved(raw_results), gt_query))
    return results


def aggregate_trake_group(
    sequence_length: int, results: Sequence[TrakeQueryResult], top_k: int, wall_clock_sec: float
) -> TrakeGroupMetrics:
    num_found = sum(1 for r in results if r.video_rank is not None)
    per_event = tuple(
        per_event_recall_at_k(results, event_index=i, k=top_k) for i in range(sequence_length)
    )
    return TrakeGroupMetrics(
        sequence_length=sequence_length,
        num_queries=len(results),
        num_found=num_found,
        sequence_exact_match=sequence_exact_match_rate(results),
        video_level_recall_at_k=video_level_recall_at_k(results, top_k),
        per_event_recall_at_k=per_event,
        mean_kendall_tau=mean_kendall_tau(results) if num_found else None,
        mean_offset_sec=mean_offset_sec(results) if num_found else None,
        wall_clock_sec=wall_clock_sec,
    )


async def run_trake_eval(
    searcher: TrakeSearcher,
    ground_truth: Sequence[GroundTruthSegment],
    sequence_lengths: Sequence[int],
    top_k: int,
    lambda_penalty: float | None,
) -> list[TrakeGroupMetrics]:
    """Full TRAKE eval pass: for each N in `sequence_lengths`, extract every
    valid N-consecutive-shot sequence from `ground_truth`, run + score them,
    and aggregate.

    Every N is validated as extractable BEFORE any `searcher.search()` call
    runs (regression, chủ dự án 2026-08-14: a real run hit N=7's
    ValueError only AFTER N=3 and N=5 had already spent 5+ minutes of real
    DANTE search, with nothing written to disk since `write_report` only
    runs after this whole function returns -- all of that compute was
    wasted). Raises `ValueError` (not a silently empty group) when an N has
    zero extractable sequences -- CLAUDE.md §7: missing data must fail
    loudly, not print a metrics table that reads all-zero as if the system
    itself were broken."""
    sequences_by_n: dict[int, list[Sequence[GroundTruthSegment]]] = {}
    for n in sequence_lengths:
        sequences = extract_trake_sequences(ground_truth, n)
        if not sequences:
            raise ValueError(
                f"run_trake_eval: N={n} has no extractable TRAKE sequence in ground_truth -- "
                "nothing to evaluate for this sequence length"
            )
        sequences_by_n[n] = sequences

    groups: list[TrakeGroupMetrics] = []
    for n in sequence_lengths:
        sequences = sequences_by_n[n]
        start = time.perf_counter()
        results = await run_trake_queries(searcher, sequences, top_k, lambda_penalty)
        wall_clock_sec = time.perf_counter() - start
        group = aggregate_trake_group(n, results, top_k, wall_clock_sec)
        groups.append(group)
        logger.info(
            "trake: N=%d, %d sequence(s), Sequence Exact-Match=%.4f, wall_clock=%.2fs",
            n,
            len(sequences),
            group.sequence_exact_match,
            wall_clock_sec,
        )
    return groups


def render_trake_report_md(groups: Sequence[TrakeGroupMetrics]) -> str:
    lines = [
        "## TRAKE metrics (DANTE)",
        "",
        "| N | queries | found | Sequence Exact-Match | Video Recall@K | "
        "Kendall's tau | Mean Offset (s) | wall-clock (s) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for g in groups:
        tau_str = "n/a" if g.mean_kendall_tau is None else f"{g.mean_kendall_tau:.4f}"
        offset_str = "n/a" if g.mean_offset_sec is None else f"{g.mean_offset_sec:.4f}"
        lines.append(
            f"| {g.sequence_length} | {g.num_queries} | {g.num_found} | "
            f"{g.sequence_exact_match:.4f} | {g.video_level_recall_at_k:.4f} | "
            f"{tau_str} | {offset_str} | {g.wall_clock_sec:.2f} |"
        )

    lines += ["", "Per-event Recall@K (segment position within sequence):", ""]
    for g in groups:
        formatted = [f"{r:.4f}" for r in g.per_event_recall_at_k]
        lines.append(f"- N={g.sequence_length}: {formatted}")

    return "\n".join(lines) + "\n"
