"""TRAKE candidate-set chaining -- the pure algorithm behind `POST /api/trake`.

This module replaces the corpus-wide DANTE dynamic program
(`app/services/dante_service.py`, now unused by the API) with the candidate-set
design: KIS retrieves per-event candidates, the surviving videos each get a
small layered DAG, and the ordered sequence is the highest-scoring path through
it.

Why a different algorithm and not just a faster DANTE:

* DANTE has to run over every video's full keyframe slice, so its objective is
  forced to be additive -- only a sum telescopes into the O(N*T) running-max
  trick. TRAKE is graded per event, so a sequence is worth what its WORST event
  is worth. Here each video's graph has ~`bucket_top_k` nodes per event, so the
  bottleneck ("weak link") objective is affordable and is what actually gets
  optimized.
* DANTE's temporal term telescopes to `lambda * (t_last - t_first)`: it charges
  for the total span and says nothing about the gaps in between, so two
  different events can be answered by two frames 0.4s apart in the same shot.
  Here `min_gap_seconds` / `max_gap_seconds` are edge constraints, checked per
  gap.

Everything in here is pure: numpy arrays and frozen dataclasses in, ranked
sequences out. No Milvus, no Mongo, no encoder -- that orchestration lives in
`app/services/trake_service.py`, which is what makes these invariants unit
testable (CLAUDE.md §8: turn the task into a verifiable criterion).

Score scale. Every `CandidateFrame.score` is expected to already be comparable
ACROSS events -- see `zscore_rows`. Raw cosine is not: different event wordings
have different score distributions against the same corpus, so summing raw
cosines across events lets the "easy" event dominate the chain.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

# Strict temporal ordering is enforced independently of `min_gap_seconds`: even
# with min_gap 0, two events can never land on the same instant.
_STRICT_ORDER_EPSILON = 0.0


class TrakeChainError(Exception):
    """Raised for a caller-fixable input problem (malformed matrix, empty
    bucket set). Carries `error_code`/`status_code` so `app/api/trake.py` can
    map each distinct cause to an HTTP response without parsing message text
    (CLAUDE.md §7), mirroring `SearchServiceError`."""

    def __init__(self, message: str, error_code: str, status_code: int) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.status_code = status_code


@dataclass(frozen=True)
class CandidateFrame:
    """One keyframe offered as the answer to ONE event.

    `frame_idx` is the absolute frame index and is what gets submitted to the
    organisers -- it is carried through from the store, never recomputed as
    `timestamp * fps` (CLAUDE.md §3: one conversion mistake silently misses the
    TRAKE window).
    """

    corpus_index: int
    keyframe_id: str
    video_id: str
    frame_idx: int
    timestamp: float
    score: float
    semantic_z: float = 0.0
    kis_evidence: float = 0.0
    kis_rank: int | None = None


@dataclass(frozen=True)
class ChainCandidate:
    """One complete ordered sequence: exactly one frame per event, strictly
    increasing in time. `frames[i]` answers event `i`."""

    video_id: str
    frames: tuple[CandidateFrame, ...]
    min_score: float
    mean_score: float
    coverage: float
    score: float

    @property
    def frame_indices(self) -> tuple[int, ...]:
        return tuple(frame.frame_idx for frame in self.frames)

    @property
    def timestamps(self) -> tuple[float, ...]:
        return tuple(frame.timestamp for frame in self.frames)


@dataclass(frozen=True)
class VideoScore:
    video_id: str
    score: float
    coverage: float
    best_chain_score: float
    rank_evidence: float
    num_keyframes: int
    num_chains: int


@dataclass(frozen=True)
class _Partial:
    """A partial path through the layered DAG. `min_s` and `sum_s` are carried
    together because the ranking objective needs both, and neither alone is
    decomposable into the other."""

    min_s: float
    sum_s: float
    indices: tuple[int, ...]


def zscore_rows(scores: np.ndarray) -> np.ndarray:
    """Make each event row comparable to every other event row.

    `scores[i, t]` is event `i`'s cosine against corpus frame `t`. Two rows
    computed from two different event texts live on different scales: against
    the same corpus, "a man holding a microphone" has a visibly higher mean
    cosine than "a referee raises a red card". Any cross-event comparison --
    the chain sum, the weak-link min, a shared threshold -- is meaningless
    until that is removed.

    Returns a float64 array of the same shape. A constant row (zero variance)
    maps to all zeros rather than dividing by ~0.
    """
    matrix = np.asarray(scores, dtype=np.float64)
    if matrix.ndim != 2:
        raise TrakeChainError(
            f"scores must be a 2D (N events x T frames) array, got shape {matrix.shape}",
            error_code="invalid_score_shape",
            status_code=400,
        )
    mean = matrix.mean(axis=1, keepdims=True)
    std = matrix.std(axis=1, keepdims=True)
    # Rows whose std underflows are constant: every frame is equally (un)like
    # the event, so every z is 0. Dividing instead would amplify float noise
    # into a fake ranking.
    safe_std = np.where(std > 1e-12, std, 1.0)
    normalized = (matrix - mean) / safe_std
    return np.where(std > 1e-12, normalized, 0.0)


def kis_evidence_score(rank: int | None, rrf_k: int) -> float:
    """Rank-based bonus for a frame KIS actually returned, in (0, 1].

    Rank is used rather than the KIS fused score because the fused score mixes
    BM25, title matching and cosine on a scale that has nothing to do with the
    semantic z-scores the chain runs on. Rank is comparable by construction.
    """
    if rank is None or rank < 0:
        return 0.0
    return float(rrf_k) / float(rrf_k + rank)


def temporal_nms(
    frames: Sequence[CandidateFrame], *, min_seconds: float, keep: int
) -> list[CandidateFrame]:
    """Greedy non-maximum suppression on the time axis.

    KIS's top-k inside one video is a cluster of adjacent keyframes from a
    single shot: without this, a bucket of 30 describes one moment 30 times
    instead of 30 distinct moments, and the chaining step below then spends its
    whole beam on near-duplicates.

    Returns at most `keep` frames, highest score first, each at least
    `min_seconds` away from every frame already kept.
    """
    if keep <= 0:
        return []
    ordered = sorted(frames, key=lambda frame: (-frame.score, frame.timestamp))
    kept: list[CandidateFrame] = []
    for frame in ordered:
        if len(kept) >= keep:
            break
        if any(abs(frame.timestamp - other.timestamp) < min_seconds for other in kept):
            continue
        kept.append(frame)
    return kept


def _partial_objective(partial: _Partial, length: int, weak_link_beta: float) -> float:
    return partial.min_s + weak_link_beta * (partial.sum_s / max(length, 1))


def _edge_is_valid(gap: float, min_gap: float, max_gap: float) -> bool:
    # Strict ordering first: `gap > 0` is the invariant a TRAKE sequence can
    # never violate, independent of the tunable min/max gap window.
    if gap <= _STRICT_ORDER_EPSILON:
        return False
    return min_gap <= gap <= max_gap


def chain_sequences(
    buckets: Sequence[Sequence[CandidateFrame]],
    *,
    min_gap_seconds: float,
    max_gap_seconds: float,
    beam_width: int,
    weak_link_beta: float,
    coverage_threshold: float,
    coverage_weight: float,
) -> list[ChainCandidate]:
    """Exact-ish top paths through the layered DAG for ONE video.

    Layer `i` holds event `i`'s candidate frames, sorted by timestamp. An edge
    `(i, t) -> (i+1, t')` exists when `t'` is strictly later than `t` and their
    gap falls inside `[min_gap_seconds, max_gap_seconds]`.

    At each node only the best `beam_width` partial paths are kept, ranked by
    `min + beta * mean`. With `beam_width` in the tens this is practically
    exact for the small buckets this pipeline produces, and it is the reason
    the bottleneck objective is affordable here at all.

    Returns every surviving complete path, best first. Returns `[]` (not an
    error) when no ordered path exists -- the caller decides whether to relax
    `max_gap_seconds` and retry, or drop the video with a log line.
    """
    if not buckets:
        raise TrakeChainError(
            "chain_sequences requires at least one event bucket",
            error_code="no_events",
            status_code=400,
        )
    if any(len(bucket) == 0 for bucket in buckets):
        return []
    if beam_width < 1:
        raise TrakeChainError(
            "beam_width must be at least 1",
            error_code="invalid_beam_width",
            status_code=400,
        )

    n_events = len(buckets)
    layer: list[list[_Partial]] = [
        [_Partial(min_s=frame.score, sum_s=frame.score, indices=(index,))]
        for index, frame in enumerate(buckets[0])
    ]

    for event_index in range(1, n_events):
        previous_frames = buckets[event_index - 1]
        current_frames = buckets[event_index]
        next_layer: list[list[_Partial]] = []
        length = event_index + 1
        for current_index, frame in enumerate(current_frames):
            extensions: list[_Partial] = []
            for previous_index, previous_frame in enumerate(previous_frames):
                if not _edge_is_valid(
                    frame.timestamp - previous_frame.timestamp,
                    min_gap_seconds,
                    max_gap_seconds,
                ):
                    continue
                for partial in layer[previous_index]:
                    extensions.append(
                        _Partial(
                            min_s=min(partial.min_s, frame.score),
                            sum_s=partial.sum_s + frame.score,
                            indices=partial.indices + (current_index,),
                        )
                    )
            extensions.sort(
                key=lambda item: (
                    -_partial_objective(item, length, weak_link_beta),
                    item.indices,
                )
            )
            next_layer.append(extensions[:beam_width])
        layer = next_layer

    completed = [partial for node_partials in layer for partial in node_partials]
    if not completed:
        return []

    candidates = [
        _to_chain_candidate(
            partial,
            buckets,
            weak_link_beta=weak_link_beta,
            coverage_threshold=coverage_threshold,
            coverage_weight=coverage_weight,
        )
        for partial in completed
    ]
    candidates.sort(key=lambda candidate: (-candidate.score, candidate.frame_indices))
    return candidates


def _to_chain_candidate(
    partial: _Partial,
    buckets: Sequence[Sequence[CandidateFrame]],
    *,
    weak_link_beta: float,
    coverage_threshold: float,
    coverage_weight: float,
) -> ChainCandidate:
    frames = tuple(
        buckets[event_index][frame_index] for event_index, frame_index in enumerate(partial.indices)
    )
    n_events = len(frames)
    mean_score = partial.sum_s / n_events
    covered = sum(1 for frame in frames if frame.score >= coverage_threshold)
    coverage = covered / n_events
    score = partial.min_s + weak_link_beta * mean_score + coverage_weight * coverage
    return ChainCandidate(
        video_id=frames[0].video_id,
        frames=frames,
        min_score=partial.min_s,
        mean_score=mean_score,
        coverage=coverage,
        score=score,
    )


def sequence_nms(
    candidates: Sequence[ChainCandidate], *, frame_tolerance: int
) -> list[ChainCandidate]:
    """Drop sequences that are the same answer as a better one already kept.

    Two sequences are the same answer when EVERY event's frame is within
    `frame_tolerance` absolute frames of the other's -- the graded window is
    under 10 frames (CLAUDE.md §1.2), so two sequences differing by one keyframe
    on one event will both be right or both be wrong. Keeping both burns a
    submission slot for nothing.
    """
    if frame_tolerance <= 0:
        return list(candidates)
    kept: list[ChainCandidate] = []
    by_video: dict[str, list[ChainCandidate]] = {}
    for candidate in sorted(candidates, key=lambda item: (-item.score, item.frame_indices)):
        neighbours = by_video.setdefault(candidate.video_id, [])
        if any(_is_duplicate_sequence(candidate, other, frame_tolerance) for other in neighbours):
            continue
        neighbours.append(candidate)
        kept.append(candidate)
    return kept


def _is_duplicate_sequence(
    left: ChainCandidate, right: ChainCandidate, frame_tolerance: int
) -> bool:
    if len(left.frames) != len(right.frames):
        return False
    return all(
        abs(a - b) <= frame_tolerance for a, b in zip(left.frame_indices, right.frame_indices)
    )


def score_videos(
    candidates: Sequence[ChainCandidate],
    *,
    keyframe_counts: dict[str, int],
    rank_evidence: dict[str, float],
    top_chains: int,
    logsumexp_weight: float,
    coverage_weight: float,
    rank_evidence_weight: float,
    length_penalty: float,
) -> dict[str, VideoScore]:
    """Rank videos, not sequences.

    A wrong video makes every one of its sequences wrong, so which video is on
    top matters more than which sequence inside it is. Three signals, plus one
    correction:

    * `logsumexp` of the video's best few sequence scores -- several good
      sequences is stronger evidence than one lucky one, and logsumexp is a
      smooth max so one outstanding sequence still wins over three mediocre.
    * the best sequence's coverage -- how many events actually cleared the bar.
    * KIS rank evidence -- how highly KIS itself ranked this video's frames,
      summed across events as `rrf_k / (rrf_k + rank)`.

    The correction is `length_penalty * log(num_keyframes)`. Every one of those
    signals is a max or a sum over the video's frames, and a max over more
    frames is larger for free: without this term, "rerank videos" degenerates
    into "sort videos by duration".
    """
    grouped: dict[str, list[ChainCandidate]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate.video_id, []).append(candidate)

    scores: dict[str, VideoScore] = {}
    for video_id, video_candidates in grouped.items():
        ordered = sorted(video_candidates, key=lambda item: -item.score)
        head = [candidate.score for candidate in ordered[: max(top_chains, 1)]]
        smooth_max = _logsumexp(head)
        best = ordered[0]
        num_keyframes = max(keyframe_counts.get(video_id, 1), 1)
        evidence = rank_evidence.get(video_id, 0.0)
        score = (
            logsumexp_weight * smooth_max
            + coverage_weight * best.coverage
            + rank_evidence_weight * evidence
            - length_penalty * math.log(num_keyframes)
        )
        scores[video_id] = VideoScore(
            video_id=video_id,
            score=score,
            coverage=best.coverage,
            best_chain_score=best.score,
            rank_evidence=evidence,
            num_keyframes=num_keyframes,
            num_chains=len(video_candidates),
        )
    return scores


def _logsumexp(values: Sequence[float]) -> float:
    if not values:
        return float("-inf")
    highest = max(values)
    if not math.isfinite(highest):
        return highest
    return highest + math.log(sum(math.exp(value - highest) for value in values))


def allocate_submission_slots(
    candidates: Sequence[ChainCandidate],
    video_scores: dict[str, VideoScore],
    *,
    max_sequences_per_video: int,
    limit: int,
) -> list[ChainCandidate]:
    """Lay the ranked sequences out across the submission's slots.

    The task score is the mean of R@k over k in {1, 5, 20, 50, 100}, and a
    wrong video makes all of its variants wrong at once. So the allocation is a
    round robin over videos: round 1 hands out each top video's best sequence
    (that is what R@1 and R@5 see, and it maximises the number of distinct
    videos covered early), later rounds add frame variants inside those same
    videos (that is what R@50 and R@100 see).

    Sorting the flat list by sequence score instead would let one video's near
    identical variants occupy the first ten slots.
    """
    if limit <= 0:
        return []
    grouped: dict[str, list[ChainCandidate]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate.video_id, []).append(candidate)
    for video_candidates in grouped.values():
        video_candidates.sort(key=lambda item: (-item.score, item.frame_indices))
        del video_candidates[max(max_sequences_per_video, 1) :]

    ordered_videos = sorted(
        grouped,
        key=lambda video_id: (
            -video_scores[video_id].score if video_id in video_scores else 0.0,
            video_id,
        ),
    )

    allocated: list[ChainCandidate] = []
    depth = 0
    deepest = max((len(group) for group in grouped.values()), default=0)
    while depth < deepest and len(allocated) < limit:
        for video_id in ordered_videos:
            if len(allocated) >= limit:
                break
            group = grouped[video_id]
            if depth < len(group):
                allocated.append(group[depth])
        depth += 1
    return allocated
