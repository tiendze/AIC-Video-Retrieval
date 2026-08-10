"""Embedding-space collapse diagnostic — the pure-measurement half.

Motivation (2026-08-10 report a40aac8): the `semantic` channel scored BELOW
the random baseline (video-level AIC 0.0451 vs 0.0809). "A bit weak on
Vietnamese" does not produce that ordering — a retriever ranking by a real
signal, however noisy, does not lose to uniform sampling. One mechanism does
produce it: if the text tower never learned Vietnamese->image alignment, every
Vietnamese query lands in the same off-distribution pocket of the space, so
top-100 is nearly the SAME keyframe set for every query. A fixed narrow set
covers fewer distinct videos than a genuinely uniform sample over the whole
corpus, so it can score worse than random.

This module turns that mechanism into four measurable signals:

  1. query-query cosine   -- do different questions produce the same vector?
  2. result-set Jaccard   -- do different questions retrieve the same keyframes?
  3. union coverage ratio -- how much of the corpus is reachable at all?
  4. score spread         -- does the ranking discriminate within top-k?

None of these needs ground-truth labels, so the verdict is independent of the
self-generated eval set (CLAUDE.md §1.5) and cannot be gamed by it.

Every function here is pure and takes plain arrays/sequences: the encoder,
Milvus and the filesystem live in `scripts/diagnose_embedding_space.py`.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

_MIN_VECTORS_FOR_PAIRWISE = 2


@dataclass(frozen=True)
class CosineStats:
    """Distribution of pairwise cosine similarity among query vectors.

    On an L2-normalized space, unrelated sentences sit well below 1.0; a
    `mean` close to 1.0 means the encoder maps different questions to nearly
    the same point, which makes ANN search return nearly the same neighbours
    regardless of what was asked.
    """

    num_vectors: int
    num_pairs: int
    mean: float
    median: float
    p05: float
    p95: float
    minimum: float
    maximum: float


@dataclass(frozen=True)
class ResultOverlapStats:
    """How much the retrieved keyframe sets repeat across different queries.

    `union_ratio` is the share of the theoretical maximum distinct results
    (`num_queries * top_k`) actually reached. A healthy retriever over a
    873-video corpus reaches a large fraction; a collapsed one keeps handing
    back the same page.
    """

    num_queries: int
    num_pairs: int
    top_k: int
    mean_jaccard: float
    median_jaccard: float
    union_size: int
    max_possible_union: int
    union_ratio: float


@dataclass(frozen=True)
class ScoreSpreadStats:
    """Within-query discrimination: the similarity gap between the best and
    the worst hit returned. A near-zero gap means the ranking inside top-k is
    essentially arbitrary even if the set itself varies."""

    num_queries: int
    mean_top1: float
    mean_last: float
    mean_gap: float


@dataclass(frozen=True)
class CollapseThresholds:
    """Cutoffs for `assess_collapse`, loaded from `configs/eval/diagnostics.yaml`
    (CLAUDE.md §6: no magic numbers). Deliberately generous — this diagnostic
    separates "degenerate" from "merely weak", it does not grade quality."""

    query_cosine_mean: float
    result_jaccard_mean: float
    union_ratio: float
    score_gap: float
    min_signals: int


@dataclass(frozen=True)
class CollapseVerdict:
    """`signals` names every threshold that fired, in fixed order, so a report
    reader sees WHICH evidence supports the verdict rather than a bare bool."""

    collapsed: bool
    signals: tuple[str, ...]
    num_signals_required: int


def pairwise_cosine_stats(vectors: np.ndarray) -> CosineStats:
    """Cosine stats over every unordered pair of rows in `vectors`.

    `vectors` must be L2-normalized (the `MultimodalEncoder` contract in
    `app/encoders/base.py` guarantees this), so the Gram matrix IS the cosine
    matrix. No re-normalization happens here on purpose: re-normalizing would
    silently paper over an encoder that violated the contract.
    """
    if vectors.ndim != 2:
        raise ValueError(f"vectors must be 2-D (N, dim), got shape {vectors.shape}")
    if vectors.shape[0] < _MIN_VECTORS_FOR_PAIRWISE:
        raise ValueError(
            f"need at least {_MIN_VECTORS_FOR_PAIRWISE} vectors to form a pair, "
            f"got {vectors.shape[0]}"
        )

    as_float64 = vectors.astype(np.float64)
    gram = as_float64 @ as_float64.T
    upper_i, upper_j = np.triu_indices(gram.shape[0], k=1)
    pairs = gram[upper_i, upper_j]
    return CosineStats(
        num_vectors=int(vectors.shape[0]),
        num_pairs=int(pairs.size),
        mean=float(np.mean(pairs)),
        median=float(np.median(pairs)),
        p05=float(np.percentile(pairs, 5)),
        p95=float(np.percentile(pairs, 95)),
        minimum=float(np.min(pairs)),
        maximum=float(np.max(pairs)),
    )


def result_overlap_stats(result_sets: Sequence[Sequence[str]]) -> ResultOverlapStats:
    """Jaccard overlap between every pair of retrieved id lists, plus how many
    distinct ids the whole query batch reached.

    Duplicate ids inside one result list are collapsed (a set is the right
    model of "which keyframes did this query reach"). `top_k` is reported as
    the largest list length seen, so `max_possible_union` stays honest when
    Milvus returns fewer rows than requested for some query.
    """
    if len(result_sets) < _MIN_VECTORS_FOR_PAIRWISE:
        raise ValueError(
            f"need at least {_MIN_VECTORS_FOR_PAIRWISE} result sets to form a pair, "
            f"got {len(result_sets)}"
        )

    sets = [frozenset(ids) for ids in result_sets]
    jaccards: list[float] = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            union = sets[i] | sets[j]
            # Two empty result lists are trivially "identical" but carry no
            # evidence either way -- score them 0.0 rather than dividing by 0.
            jaccards.append(len(sets[i] & sets[j]) / len(union) if union else 0.0)

    union_all: set[str] = set()
    for one_set in sets:
        union_all |= one_set
    top_k = max((len(one_set) for one_set in sets), default=0)
    max_possible = len(sets) * top_k

    return ResultOverlapStats(
        num_queries=len(sets),
        num_pairs=len(jaccards),
        top_k=top_k,
        mean_jaccard=float(statistics.fmean(jaccards)),
        median_jaccard=float(statistics.median(jaccards)),
        union_size=len(union_all),
        max_possible_union=max_possible,
        union_ratio=(len(union_all) / max_possible) if max_possible else 0.0,
    )


def score_spread_stats(scores_per_query: Sequence[Sequence[float]]) -> ScoreSpreadStats:
    """Mean best score, mean worst score and mean gap across queries.

    Queries that returned nothing are skipped rather than counted as a zero
    gap -- an empty result list means "Milvus had nothing to return", which is
    a different failure from "everything scored the same".
    """
    non_empty = [list(scores) for scores in scores_per_query if len(scores) > 0]
    if not non_empty:
        raise ValueError("no query returned any score -- nothing to measure")

    tops = [scores[0] for scores in non_empty]
    lasts = [scores[-1] for scores in non_empty]
    return ScoreSpreadStats(
        num_queries=len(non_empty),
        mean_top1=float(statistics.fmean(tops)),
        mean_last=float(statistics.fmean(lasts)),
        mean_gap=float(
            statistics.fmean([top - last for top, last in zip(tops, lasts, strict=True)])
        ),
    )


def assess_collapse(
    cosine: CosineStats,
    overlap: ResultOverlapStats,
    spread: ScoreSpreadStats,
    thresholds: CollapseThresholds,
) -> CollapseVerdict:
    """Fires a named signal per breached threshold; `collapsed` is True once at
    least `thresholds.min_signals` fire.

    Requiring more than one signal on purpose: any single measure has an
    innocent explanation (a corpus of near-duplicate news frames genuinely
    produces high result overlap, for instance). Several firing together do
    not.
    """
    signals: list[str] = []
    if cosine.mean >= thresholds.query_cosine_mean:
        signals.append(
            f"query_cosine_mean={cosine.mean:.4f} >= {thresholds.query_cosine_mean} "
            "(different questions encode to nearly the same vector)"
        )
    if overlap.mean_jaccard >= thresholds.result_jaccard_mean:
        signals.append(
            f"result_jaccard_mean={overlap.mean_jaccard:.4f} >= {thresholds.result_jaccard_mean} "
            "(different questions retrieve the same keyframes)"
        )
    if overlap.union_ratio <= thresholds.union_ratio:
        signals.append(
            f"union_ratio={overlap.union_ratio:.4f} <= {thresholds.union_ratio} "
            "(the query batch reaches only a narrow slice of the corpus)"
        )
    if spread.mean_gap <= thresholds.score_gap:
        signals.append(
            f"score_gap={spread.mean_gap:.4f} <= {thresholds.score_gap} "
            "(rank 1 and rank k are scored almost identically)"
        )
    return CollapseVerdict(
        collapsed=len(signals) >= thresholds.min_signals,
        signals=tuple(signals),
        num_signals_required=thresholds.min_signals,
    )


def render_report_md(
    *,
    label: str,
    git_sha: str,
    model_id: str,
    collection_name: str,
    num_queries: int,
    cosine: CosineStats,
    overlap: ResultOverlapStats,
    spread: ScoreSpreadStats,
    verdict: CollapseVerdict,
) -> str:
    """Markdown for one query-language run. `label` distinguishes runs over the
    same collection (e.g. "vi" vs "en") so the two can be diffed side by side."""
    lines = [
        f"# Embedding-space diagnostic -- {label}",
        "",
        f"- git_sha: {git_sha}",
        f"- model_id: {model_id}",
        f"- collection: {collection_name}",
        f"- num_queries: {num_queries}",
        "",
        "## Verdict",
        "",
        f"- collapsed: **{verdict.collapsed}** "
        f"({len(verdict.signals)} signal(s) fired, "
        f"{verdict.num_signals_required} required)",
    ]
    lines += [f"  - {signal}" for signal in verdict.signals] or ["  - no signal fired"]

    lines += [
        "",
        "## Query-query cosine (text tower)",
        "",
        f"- pairs: {cosine.num_pairs}",
        f"- mean: {cosine.mean:.4f} | median: {cosine.median:.4f}",
        f"- p05: {cosine.p05:.4f} | p95: {cosine.p95:.4f}",
        f"- min: {cosine.minimum:.4f} | max: {cosine.maximum:.4f}",
        "",
        "## Result-set overlap (top-k from Milvus)",
        "",
        f"- top_k: {overlap.top_k} | pairs: {overlap.num_pairs}",
        f"- mean Jaccard: {overlap.mean_jaccard:.4f} | median: {overlap.median_jaccard:.4f}",
        f"- distinct keyframes reached: {overlap.union_size}/{overlap.max_possible_union} "
        f"(ratio {overlap.union_ratio:.4f})",
        "",
        "## Score spread (within query)",
        "",
        f"- mean rank-1 score: {spread.mean_top1:.4f}",
        f"- mean rank-k score: {spread.mean_last:.4f}",
        f"- mean gap: {spread.mean_gap:.4f}",
        "",
    ]
    return "\n".join(lines) + "\n"
