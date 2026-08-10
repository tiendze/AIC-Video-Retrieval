"""Pure retrieval-scoring functions for `eval/run.py` (plan §P0.9).

The ground truth has exactly one relevant item per query -- one caption maps
to one shot (`GroundTruthSegment`, `datasets/base.py`). That single fact
shapes every formula here: `mean_average_precision` collapses to `1/rank`
(no need for a full ranked-relevance-list AP), and `ndcg_at_k`'s ideal DCG is
always 1.0 (a single relevant item at rank 1). These are documented
assumptions, not a generic multi-relevant IR library -- do not reuse these
functions for TRAKE sequence scoring (that needs a different formula, see
PRD §6.3 Sequence Exact-Match / Kendall's tau, tracked separately).

`recall_at_k`/`mean_reciprocal_rank`/`ndcg_at_k`/`median_rank`/
`aic_final_score`/`recall_at_k_by_group` all raise `ValueError` on an empty
`ranks` input rather than returning a misleading 0.0 (CLAUDE.md §7: missing
data must fail loudly, not print a metrics table that looks like "the system
is broken" when really "there was nothing to evaluate").
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

Rank = int | None
HitLevel = Literal["video", "segment"]

# The AIC 2026 submission's own scoring mesh (references.md §1.2) -- the one
# set of K's that is directly comparable to the contest scoreboard.
AIC_OFFICIAL_KS: tuple[int, ...] = (1, 5, 20, 50, 100)


@dataclass(frozen=True)
class RetrievedHit:
    """One ranked search result, reduced to only what scoring needs."""

    video_id: str
    timestamp: float


@dataclass(frozen=True)
class GroundTruthQuery:
    """One evaluation query's answer key -- a `GroundTruthSegment` narrowed
    to the fields the point-in-segment hit rule (PRD §6.2) needs."""

    video_id: str
    start_sec: float
    end_sec: float


def first_hit_rank(
    hits: Sequence[RetrievedHit], ground_truth: GroundTruthQuery, level: HitLevel
) -> Rank:
    """1-indexed rank of the first hit that matches `ground_truth`, or `None`
    if no hit matches. `level="video"` only requires the same `video_id`;
    `level="segment"` additionally requires the point-in-segment rule
    (PRD §6.2): `start_sec <= hit.timestamp <= end_sec`, inclusive on both
    ends. The two levels diverging (video-level high, segment-level low)
    is the plan's diagnostic for a keyframe-sampling bug rather than an
    embedding bug -- both must be computed, not just one."""
    for rank, hit in enumerate(hits, start=1):
        if hit.video_id != ground_truth.video_id:
            continue
        if level == "video":
            return rank
        if ground_truth.start_sec <= hit.timestamp <= ground_truth.end_sec:
            return rank
    return None


def _require_nonempty(ranks: Sequence[Rank], fn_name: str) -> None:
    if not ranks:
        raise ValueError(f"{fn_name}: ranks is empty -- nothing was evaluated")


def recall_at_k(ranks: Sequence[Rank], k: int) -> float:
    _require_nonempty(ranks, "recall_at_k")
    hits = sum(1 for r in ranks if r is not None and r <= k)
    return hits / len(ranks)


def mean_reciprocal_rank(ranks: Sequence[Rank]) -> float:
    _require_nonempty(ranks, "mean_reciprocal_rank")
    return sum((1.0 / r) if r is not None else 0.0 for r in ranks) / len(ranks)


def mean_average_precision(ranks: Sequence[Rank]) -> float:
    """AP per query is `1/rank` (or 0 if not found) since there is exactly
    one relevant item per query -- see module docstring. Numerically
    identical to `mean_reciprocal_rank`; kept as a separate named function
    because PRD §6.3 reports mAP and MRR as distinct rows."""
    return mean_reciprocal_rank(ranks)


def ndcg_at_k(ranks: Sequence[Rank], k: int) -> float:
    """Ideal DCG is always `1 / log2(1+1) = 1` (single relevant item, best
    case rank 1), so nDCG@k reduces to plain DCG@k here."""
    _require_nonempty(ranks, "ndcg_at_k")

    def _dcg(rank: Rank) -> float:
        if rank is None or rank > k:
            return 0.0
        return 1.0 / math.log2(rank + 1)

    return sum(_dcg(r) for r in ranks) / len(ranks)


def median_rank(ranks: Sequence[Rank]) -> float | None:
    """Median over the *found* ranks only; `None` if nothing was found.
    Reported alongside the found/not-found count in eval/run.py's report --
    this function alone cannot distinguish "median rank 3 out of 600 found"
    from "median rank 3 out of 2 found"."""
    _require_nonempty(ranks, "median_rank")
    found = sorted(r for r in ranks if r is not None)
    if not found:
        return None
    n = len(found)
    mid = n // 2
    if n % 2 == 1:
        return float(found[mid])
    return (found[mid - 1] + found[mid]) / 2.0


def aic_final_score(ranks: Sequence[Rank], ks: tuple[int, ...] = AIC_OFFICIAL_KS) -> float:
    """`mean(Recall@k for k in {1,5,20,50,100})` -- the one number directly
    comparable to the AIC 2026 scoreboard (references.md §1.2, PRD §6.3)."""
    _require_nonempty(ranks, "aic_final_score")
    return sum(recall_at_k(ranks, k) for k in ks) / len(ks)


def recall_at_k_by_group(ranks_with_group: Sequence[tuple[Rank, str]], k: int) -> dict[str, float]:
    """Recall@k computed independently within each group -- used for the
    per-`author`/per-`batch` breakdown (PRD §6.1.4, replacing YouCook2's
    `recipe_type`)."""
    if not ranks_with_group:
        raise ValueError("recall_at_k_by_group: ranks_with_group is empty")
    by_group: dict[str, list[Rank]] = defaultdict(list)
    for rank, group in ranks_with_group:
        by_group[group].append(rank)
    return {group: recall_at_k(ranks, k) for group, ranks in by_group.items()}
