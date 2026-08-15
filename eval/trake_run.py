"""TRAKE evaluation orchestration for `eval/run.py` (plan §P2.3, PRD §6.4).

Kept as its own module -- eval/run.py is already past the 800-line file
ceiling in .claude/rules/ecc/common/coding-style.md (chủ dự án 2026-08-14:
split rather than grow it further). `eval/run.py` stays the single CLI
entry point (CLAUDE.md §6);
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


@dataclass(frozen=True)
class TrakeAblationCell:
    """One (λ, N) cell of the P2.4 ablation grid (PRD §6.4, plan §P2.4) --
    the same `TrakeGroupMetrics` a single-λ run already produces, tagged
    with which λ produced it."""

    lambda_penalty: float
    metrics: TrakeGroupMetrics


async def run_trake_lambda_ablation(
    searcher: TrakeSearcher,
    ground_truth: Sequence[GroundTruthSegment],
    sequence_lengths: Sequence[int],
    lambda_values: Sequence[float],
    top_k: int,
) -> list[TrakeAblationCell]:
    """Full λ × N ablation sweep (plan §P2.4, PRD §6.4's mandatory ablation
    table) -- paper proposes two λ values with no ablation ever run to
    justify either.

    Each N's ground-truth sequences are extracted ONCE (`extract_trake_
    sequences` is independent of λ -- only DANTE's DP inside `searcher.
    search()` changes across λ), then re-run against every λ in
    `lambda_values`. Same fail-fast contract as `run_trake_eval` (P2.3
    regression): every N is validated extractable BEFORE any
    `searcher.search()` call runs for ANY λ, so a data gap (e.g. N=7 not
    yet in the frozen eval set) is caught in milliseconds, not after
    several λ×N cells' worth of real DANTE search.
    """
    sequences_by_n: dict[int, list[Sequence[GroundTruthSegment]]] = {}
    for n in sequence_lengths:
        sequences = extract_trake_sequences(ground_truth, n)
        if not sequences:
            raise ValueError(
                f"run_trake_lambda_ablation: N={n} has no extractable TRAKE sequence in "
                "ground_truth -- nothing to evaluate for this sequence length"
            )
        sequences_by_n[n] = sequences

    cells: list[TrakeAblationCell] = []
    for lam in lambda_values:
        for n in sequence_lengths:
            sequences = sequences_by_n[n]
            start = time.perf_counter()
            results = await run_trake_queries(searcher, sequences, top_k, lam)
            wall_clock_sec = time.perf_counter() - start
            group = aggregate_trake_group(n, results, top_k, wall_clock_sec)
            cells.append(TrakeAblationCell(lambda_penalty=lam, metrics=group))
            logger.info(
                "trake_ablation: lambda=%s, N=%d, Sequence Exact-Match=%.4f, "
                "Kendall's tau=%s, wall_clock=%.2fs",
                lam,
                n,
                group.sequence_exact_match,
                "n/a" if group.mean_kendall_tau is None else f"{group.mean_kendall_tau:.4f}",
                wall_clock_sec,
            )
    return cells


def render_trake_ablation_report_md(cells: Sequence[TrakeAblationCell]) -> str:
    """λ × N grid (plan §P2.4, PRD §6.4) -- two tables (Sequence Exact-Match,
    Kendall's tau) since a single cell number can't show both headline
    metrics PRD §6.4 requires at once."""
    sequence_lengths = sorted({c.metrics.sequence_length for c in cells})
    lambda_values = sorted({c.lambda_penalty for c in cells})
    by_key = {(c.lambda_penalty, c.metrics.sequence_length): c.metrics for c in cells}

    header = "| λ | " + " | ".join(f"N={n}" for n in sequence_lengths) + " |"
    sep = "|---|" + "---|" * len(sequence_lengths)

    def _grid(value_of) -> list[str]:
        rows = [header, sep]
        for lam in lambda_values:
            row = [f"{lam:g}"]
            for n in sequence_lengths:
                metrics = by_key.get((lam, n))
                value = None if metrics is None else value_of(metrics)
                row.append("n/a" if value is None else f"{value:.4f}")
            rows.append("| " + " | ".join(row) + " |")
        return rows

    lines = ["## DANTE λ ablation (PRD §6.4, plan §P2.4)", ""]
    lines += ["Sequence Exact-Match", ""] + _grid(lambda m: m.sequence_exact_match)
    lines += ["", "Kendall's tau", ""] + _grid(lambda m: m.mean_kendall_tau)
    lines += [
        "",
        "λ=0 is the pure order-constraint case (no time-distance penalty) -- compare its "
        "row to λ>0 rows to see whether DANTE's distance penalty contributes anything "
        "beyond the ordering constraint alone. Paper-proposed values: λ=0.001 (index "
        "distance 3-15), λ=0.01 (distance 1-3).",
    ]
    return "\n".join(lines) + "\n"


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
