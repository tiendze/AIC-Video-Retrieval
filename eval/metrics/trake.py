"""Pure TRAKE scoring functions for `eval/run.py` (plan §P2.3, PRD §6.4).

Deliberately NOT built on eval/metrics/retrieval.py's functions (that
module's own docstring says not to): TRAKE scores an N-keyframe SEQUENCE per
query against N ordered ground-truth segments, not a single point-in-segment
hit against one segment.

**Kendall's tau design ("Cách B", confirmed with chủ dự án 2026-08-14):**
each retrieved keyframe is reassigned to whichever ground-truth event's
segment CENTER it is temporally closest to -- not the event at the same
sequence position. DANTE's `τ < t` invariant already forces the retrieved
sequence to be strictly time-increasing (app/services/dante_service.py), and
the N ground-truth segments are already time-ordered by construction (N
consecutive shots, `datasets/base.py`'s `order_index`). Comparing position i
to position i directly would therefore ALWAYS read tau=1.0 regardless of
retrieval quality -- degenerate, and unresponsive to `λ` even though PRD
§6.4 explicitly says tau is "the thing λ controls". Nearest-center
reassignment is the version of this metric that can actually vary: if the
DP picks a frame that is semantically/temporally closer to a DIFFERENT
event's window than the one it was assigned to find, that shows up as a
rank inversion here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

Rank = int | None


@dataclass(frozen=True)
class TrakeGroundTruthEvent:
    """One event's answer key within a TRAKE query's N-event sequence -- a
    `GroundTruthSegment` (`datasets/base.py`) narrowed to what TRAKE scoring
    needs, mirroring `eval/metrics/retrieval.py`'s `GroundTruthQuery`."""

    start_sec: float
    end_sec: float

    @property
    def center_sec(self) -> float:
        return (self.start_sec + self.end_sec) / 2.0


@dataclass(frozen=True)
class TrakeGroundTruthQuery:
    """`events` in `order_index` order, length N."""

    video_id: str
    events: tuple[TrakeGroundTruthEvent, ...]


@dataclass(frozen=True)
class TrakeRetrievedVideo:
    """One ranked TRAKE result, reduced to only what scoring needs --
    mirrors `RetrievedHit`. `keyframe_timestamps` is
    `TrakeVideoResult.keyframes[i].timestamp` in retrieval order
    (`app/services/trake_service.py`)."""

    video_id: str
    keyframe_timestamps: tuple[float, ...]


@dataclass(frozen=True)
class TrakeQueryResult:
    """One query's full scoring outcome. Every field beyond `video_rank` is
    `None` when the ground-truth video was never retrieved -- there is
    nothing to score a keyframe sequence against."""

    video_rank: Rank
    per_event_hit: tuple[bool, ...] | None
    kendall_tau: float | None
    mean_temporal_offset_sec: float | None


def video_level_rank(retrieved: Sequence[TrakeRetrievedVideo], ground_truth_video_id: str) -> Rank:
    """1-indexed rank of the first result whose `video_id` matches, or
    `None` if never retrieved."""
    for rank, video in enumerate(retrieved, start=1):
        if video.video_id == ground_truth_video_id:
            return rank
    return None


def _require_same_length(a: Sequence, b: Sequence, fn_name: str) -> None:
    if len(a) != len(b):
        raise ValueError(f"{fn_name}: length mismatch -- {len(a)} keyframe(s) vs {len(b)} event(s)")


def per_event_hits(
    keyframe_timestamps: Sequence[float], events: Sequence[TrakeGroundTruthEvent]
) -> tuple[bool, ...]:
    """Position-aligned point-in-segment rule (PRD §6.2's rule, applied at
    each fixed position i): does `keyframe_timestamps[i]` fall inside
    `events[i]`'s own `[start_sec, end_sec]` window, inclusive on both ends.
    """
    _require_same_length(keyframe_timestamps, events, "per_event_hits")
    return tuple(
        event.start_sec <= ts <= event.end_sec
        for ts, event in zip(keyframe_timestamps, events, strict=True)
    )


def assign_nearest_segment_order(
    keyframe_timestamps: Sequence[float], events: Sequence[TrakeGroundTruthEvent]
) -> tuple[int, ...]:
    """For each retrieved timestamp, the index of the event whose segment
    center it is closest to (ties broken toward the earlier/lower index,
    since `min` keeps the first-seen minimum). See module docstring for why
    this -- not a position-i echo -- is the input Kendall's tau needs."""
    centers = [event.center_sec for event in events]
    return tuple(
        min(range(len(centers)), key=lambda i: abs(ts - centers[i])) for ts in keyframe_timestamps
    )


def kendall_tau(x: Sequence[float], y: Sequence[float]) -> float:
    """Generic pairwise Kendall's tau-b: `(C - D) / sqrt((C+D+Tx)(C+D+Ty))`
    over all `n*(n-1)/2` pairs, where C/D are concordant/discordant pairs and
    Tx/Ty are pairs tied only in x/only in y (a pair tied in both is excluded
    from every term, standard tau-b treatment)."""
    _require_same_length(x, y, "kendall_tau")
    n = len(x)
    if n < 2:
        raise ValueError(f"kendall_tau: need at least 2 points, got {n}")

    concordant = discordant = tied_x = tied_y = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[j] - x[i]
            dy = y[j] - y[i]
            if dx == 0 and dy == 0:
                continue  # tied in both -- excluded from every term
            if dx == 0:
                tied_x += 1
            elif dy == 0:
                tied_y += 1
            elif (dx > 0) == (dy > 0):
                concordant += 1
            else:
                discordant += 1

    denom = ((concordant + discordant + tied_x) * (concordant + discordant + tied_y)) ** 0.5
    if denom == 0:
        return 0.0
    return (concordant - discordant) / denom


def sequence_kendall_tau(
    keyframe_timestamps: Sequence[float], events: Sequence[TrakeGroundTruthEvent]
) -> float:
    """`kendall_tau` between retrieval position `[0, 1, ..., N-1]` and each
    keyframe's nearest-segment assignment (`assign_nearest_segment_order`)."""
    assigned = assign_nearest_segment_order(keyframe_timestamps, events)
    return kendall_tau(list(range(len(keyframe_timestamps))), list(assigned))


def mean_temporal_offset_sec(
    keyframe_timestamps: Sequence[float], events: Sequence[TrakeGroundTruthEvent]
) -> float:
    """PRD §6.4: mean absolute distance between `timestamp(k_i)` and
    `events[i]`'s segment center, position-aligned (this metric answers "how
    far off was the position-i pick", independent of the tau reassignment
    question above)."""
    _require_same_length(keyframe_timestamps, events, "mean_temporal_offset_sec")
    offsets = [
        abs(ts - event.center_sec) for ts, event in zip(keyframe_timestamps, events, strict=True)
    ]
    return sum(offsets) / len(offsets)


def score_query(
    retrieved: Sequence[TrakeRetrievedVideo], ground_truth: TrakeGroundTruthQuery
) -> TrakeQueryResult:
    """One query's full outcome: find the ground-truth video in `retrieved`,
    then score its keyframe sequence against `ground_truth.events`. Every
    field beyond `video_rank` is `None` if the video was never retrieved."""
    rank = video_level_rank(retrieved, ground_truth.video_id)
    if rank is None:
        return TrakeQueryResult(
            video_rank=None, per_event_hit=None, kendall_tau=None, mean_temporal_offset_sec=None
        )

    matched = retrieved[rank - 1]
    if len(matched.keyframe_timestamps) != len(ground_truth.events):
        raise ValueError(
            f"score_query: matched video_id={matched.video_id!r} has "
            f"{len(matched.keyframe_timestamps)} keyframe(s), expected "
            f"{len(ground_truth.events)} to match ground_truth.events -- TrakeService's own "
            "contract guarantees these are equal, so this is a wiring bug, not a scoreable miss"
        )

    return TrakeQueryResult(
        video_rank=rank,
        per_event_hit=per_event_hits(matched.keyframe_timestamps, ground_truth.events),
        kendall_tau=sequence_kendall_tau(matched.keyframe_timestamps, ground_truth.events),
        mean_temporal_offset_sec=mean_temporal_offset_sec(
            matched.keyframe_timestamps, ground_truth.events
        ),
    )


def _require_nonempty(results: Sequence[TrakeQueryResult], fn_name: str) -> None:
    if not results:
        raise ValueError(f"{fn_name}: results is empty -- nothing was evaluated")


def sequence_exact_match_rate(results: Sequence[TrakeQueryResult]) -> float:
    """PRD §6.4's headline metric: fraction of queries where the video was
    found AND every one of its N keyframes hit its own window."""
    _require_nonempty(results, "sequence_exact_match_rate")
    matches = sum(1 for r in results if r.per_event_hit is not None and all(r.per_event_hit))
    return matches / len(results)


def video_level_recall_at_k(results: Sequence[TrakeQueryResult], k: int) -> float:
    """Fraction of queries where the ground-truth video was retrieved within
    the top-k -- isolates "wrong video" errors from "wrong frame" errors."""
    _require_nonempty(results, "video_level_recall_at_k")
    hits = sum(1 for r in results if r.video_rank is not None and r.video_rank <= k)
    return hits / len(results)


def per_event_recall_at_k(results: Sequence[TrakeQueryResult], event_index: int, k: int) -> float:
    """Recall@k for one specific event position in the sequence: a hit
    requires BOTH the video ranked within k AND that specific event's
    keyframe landing in its own window -- pinpoints which position in the
    chain (usually the first or last) tends to fail."""
    _require_nonempty(results, "per_event_recall_at_k")
    hits = sum(
        1
        for r in results
        if r.video_rank is not None
        and r.video_rank <= k
        and r.per_event_hit is not None
        and r.per_event_hit[event_index]
    )
    return hits / len(results)


def mean_kendall_tau(results: Sequence[TrakeQueryResult]) -> float:
    """Mean `kendall_tau` over queries where the video was found -- a query
    where the video was never retrieved has no sequence to score for
    ordering quality."""
    found = [r.kendall_tau for r in results if r.kendall_tau is not None]
    if not found:
        raise ValueError("mean_kendall_tau: no query found its ground-truth video")
    return sum(found) / len(found)


def mean_offset_sec(results: Sequence[TrakeQueryResult]) -> float:
    """Mean `mean_temporal_offset_sec` over queries where the video was
    found."""
    found = [r.mean_temporal_offset_sec for r in results if r.mean_temporal_offset_sec is not None]
    if not found:
        raise ValueError("mean_offset_sec: no query found its ground-truth video")
    return sum(found) / len(found)
