"""DANTE dynamic programming for TRAKE.

Finds the highest-scoring strictly ordered sequence of N events inside one
video. The temporal term now uses real keyframe timestamps in seconds, not
keyframe/global-index distance:

    S[i,t]    = cosine_similarity(u_i, E[t])
    DP[i,t]   = S[i,t] + max_tau(DP[i-1,tau] - lambda_time*(time[t]-time[tau]))
    DANTE[v]  = max_t DP[N,t]

The implementation keeps the original O(N*T) running-max shape. For each
event row, it accumulates prefixes as `DP[i-1,tau] + lambda_time*time[tau]`
and scores the current frame as
`S[i,t] + running_max - lambda_time*time[t]`.

Only candidates with `time[tau] < time[t]` are eligible, preserving DANTE's
strict forward temporal ordering on the timestamp axis. The first event has
no predecessor, so DP[0,t] = S[0,t].
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

LAMBDA_TIME: float = 0.01
DEFAULT_LAMBDA_PENALTY: float = LAMBDA_TIME


class DanteServiceError(Exception):
    """Raised for a caller-fixable input problem: a malformed similarity
    matrix, or a video whose `[s_v, e_v]` range has fewer frames than events
    requested (CLAUDE.md §7 -- missing/insufficient data must fail loudly,
    never silently return a shorter-than-requested sequence).

    Carries `error_code`/`status_code`, mirroring `SearchServiceError`
    (app/services/search_service.py) -- so the future `POST /api/trake`
    endpoint (plan §P2.2) can map each distinct cause to an HTTP response
    without parsing the message text (CLAUDE.md §7)."""

    def __init__(self, message: str, error_code: str, status_code: int) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.status_code = status_code


@dataclass(frozen=True)
class DanteStepDebug:
    event_index: int
    timestamp: float
    semantic_score: float
    gap_seconds: float | None
    temporal_penalty: float | None


@dataclass(frozen=True)
class DanteResult:
    """`global_indices` has exactly N entries, one per event. `debug_steps`
    keeps timestamp gaps and penalties inspectable without production logs."""

    score: float
    global_indices: tuple[int, ...]
    debug_steps: tuple[DanteStepDebug, ...] = ()


@dataclass(frozen=True)
class _LocalCandidate:
    score: float
    local_indices: tuple[int, ...]


@dataclass(frozen=True)
class _AdjustedCandidate:
    adjusted_score: float
    candidate: _LocalCandidate


def compute_similarity_matrix(event_vectors: np.ndarray, frame_vectors: np.ndarray) -> np.ndarray:
    """Eq (1): S[i,t] = cosine_similarity(u_i, E[t]).

    Both inputs are assumed already L2-normalized by the encoder (CLAUDE.md
    §3 -- "L2-normalize moi embedding ngay tai encoder"), so the plain dot
    product IS the cosine similarity. Re-normalizing here would silently
    paper over an encoder that violates that invariant instead of surfacing
    it.

    event_vectors: shape (N, dim) -- one query vector per event, in order.
    frame_vectors: shape (T, dim) -- one embedding per keyframe in [s_v, e_v].
    Returns: shape (N, T).
    """
    return event_vectors @ frame_vectors.T


def _validate_similarity(similarity: np.ndarray) -> tuple[int, int]:
    if similarity.ndim != 2:
        raise DanteServiceError(
            f"similarity must be a 2D (N events x T frames) array, got shape {similarity.shape}",
            error_code="invalid_similarity_shape",
            status_code=400,
        )
    n_events, n_frames = similarity.shape
    if n_events < 1:
        raise DanteServiceError(
            "similarity must contain at least one event (N>=1) row",
            error_code="invalid_similarity_shape",
            status_code=400,
        )
    if n_frames < n_events:
        raise DanteServiceError(
            f"not enough frames ({n_frames}) in [s_v, e_v] to place {n_events} ordered events",
            error_code="insufficient_frames",
            status_code=422,
        )
    return n_events, n_frames


def _validate_timestamps(
    timestamps: Sequence[float] | np.ndarray,
    n_frames: int,
) -> np.ndarray:
    timestamp_array = np.asarray(timestamps, dtype=np.float64)
    if timestamp_array.ndim != 1:
        raise DanteServiceError(
            f"timestamps must be a 1D array of seconds, got shape {timestamp_array.shape}",
            error_code="invalid_timestamps",
            status_code=400,
        )
    if timestamp_array.shape[0] != n_frames:
        raise DanteServiceError(
            "timestamps must contain one value per frame "
            f"({n_frames}), got {timestamp_array.shape[0]}",
            error_code="invalid_timestamps",
            status_code=400,
        )
    if not np.all(np.isfinite(timestamp_array)):
        raise DanteServiceError(
            "timestamps must be finite seconds",
            error_code="invalid_timestamps",
            status_code=400,
        )
    if np.any(np.diff(timestamp_array) < 0):
        raise DanteServiceError(
            "timestamps must be sorted in nondecreasing temporal order",
            error_code="invalid_timestamps",
            status_code=400,
        )
    return timestamp_array


def temporal_gap_seconds(prev_timestamp: float, curr_timestamp: float) -> float:
    return float(curr_timestamp - prev_timestamp)


def _compute_best_path_dp(
    similarity: np.ndarray,
    timestamps: np.ndarray,
    lambda_penalty: float,
) -> tuple[np.ndarray, np.ndarray]:
    n_events, n_frames = similarity.shape
    dp = np.full((n_events, n_frames), -np.inf, dtype=np.float64)
    predecessor = np.full((n_events, n_frames), -1, dtype=np.int64)
    dp[0, :] = similarity[0, :]

    for i in range(1, n_events):
        running_max = -np.inf
        running_max_idx = -1
        next_prefix_idx = 0
        for t in range(n_frames):
            curr_timestamp = timestamps[t]
            while next_prefix_idx < t and timestamps[next_prefix_idx] < curr_timestamp:
                candidate = (
                    dp[i - 1, next_prefix_idx] + lambda_penalty * timestamps[next_prefix_idx]
                )
                if candidate > running_max:
                    running_max = candidate
                    running_max_idx = next_prefix_idx
                next_prefix_idx += 1
            if running_max_idx == -1:
                continue
            dp[i, t] = similarity[i, t] + running_max - lambda_penalty * curr_timestamp
            predecessor[i, t] = running_max_idx

    return dp, predecessor


def _globalize(s_v: int, local_indices: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(s_v + idx for idx in local_indices)


def _backtrack_best_path(predecessor: np.ndarray, n_events: int, best_t: int) -> tuple[int, ...]:
    local_indices = [0] * n_events
    local_indices[n_events - 1] = best_t
    t = best_t
    for i in range(n_events - 1, 0, -1):
        t = int(predecessor[i, t])
        local_indices[i - 1] = t
    return tuple(local_indices)


def _build_debug_steps(
    similarity: np.ndarray,
    timestamps: np.ndarray,
    local_indices: tuple[int, ...],
    lambda_penalty: float,
) -> tuple[DanteStepDebug, ...]:
    steps: list[DanteStepDebug] = []
    previous_timestamp: float | None = None
    for event_index, local_index in enumerate(local_indices):
        timestamp = float(timestamps[local_index])
        gap = (
            None
            if previous_timestamp is None
            else temporal_gap_seconds(previous_timestamp, timestamp)
        )
        steps.append(
            DanteStepDebug(
                event_index=event_index,
                timestamp=timestamp,
                semantic_score=float(similarity[event_index, local_index]),
                gap_seconds=gap,
                temporal_penalty=None if gap is None else float(lambda_penalty * gap),
            )
        )
        previous_timestamp = timestamp
    return tuple(steps)


def dante_search(
    similarity: np.ndarray,
    s_v: int,
    *,
    timestamps: Sequence[float] | np.ndarray,
    lambda_penalty: float = DEFAULT_LAMBDA_PENALTY,
) -> DanteResult:
    """Run DANTE over a precomputed similarity matrix for one video.

    similarity: shape (N, T), row i = event i's similarity to every frame in
    `[s_v, e_v]` (column t corresponds to `global_index = s_v + t`).
    timestamps: shape (T,), seconds for each frame column, sorted ascending.
    lambda_penalty: temporal penalty coefficient in seconds; defaults to
    `LAMBDA_TIME == 0.01`.
    """
    n_events, n_frames = _validate_similarity(similarity)
    timestamp_array = _validate_timestamps(timestamps, n_frames)
    dp, predecessor = _compute_best_path_dp(similarity, timestamp_array, lambda_penalty)

    last_row = dp[n_events - 1, :]
    best_t = int(np.argmax(last_row))
    best_score = float(last_row[best_t])
    if not np.isfinite(best_score):
        raise DanteServiceError(
            "no valid strictly-ordered sequence exists for this similarity matrix",
            error_code="no_valid_sequence",
            status_code=422,
        )

    local_indices = _backtrack_best_path(predecessor, n_events, best_t)
    global_indices = _globalize(s_v, local_indices)
    return DanteResult(
        score=best_score,
        global_indices=global_indices,
        debug_steps=_build_debug_steps(similarity, timestamp_array, local_indices, lambda_penalty),
    )


def _append_descending_top_k(
    candidates: list[_AdjustedCandidate], candidate: _AdjustedCandidate, top_k: int
) -> None:
    candidates.append(candidate)
    candidates.sort(
        key=lambda item: (-item.adjusted_score, item.candidate.local_indices),
    )
    del candidates[top_k:]


def dante_search_sequences(
    similarity: np.ndarray,
    s_v: int,
    threshold: float,
    top_k: int,
    *,
    timestamps: Sequence[float] | np.ndarray,
    lambda_penalty: float = DEFAULT_LAMBDA_PENALTY,
) -> list[DanteResult]:
    """Return top strictly-ordered DANTE sequences whose score is >= threshold.

    This is the k-best version of the same recurrence used by `dante_search`.
    For each possible end frame `t`, it keeps the top prefixes by
    `score + lambda_time*timestamp[tau]`; extending those prefixes by `t`
    gives the best sequences ending at `t`. Keeping only `top_k` prefixes is
    sufficient because TRAKE only needs the global top-k sequences after
    thresholding.
    """
    n_events, n_frames = _validate_similarity(similarity)
    timestamp_array = _validate_timestamps(timestamps, n_frames)
    if top_k < 1:
        raise DanteServiceError(
            "top_k must be at least 1",
            error_code="invalid_top_k",
            status_code=400,
        )

    by_end: list[list[_LocalCandidate]] = []
    for t in range(n_frames):
        score = float(similarity[0, t])
        by_end.append(
            [_LocalCandidate(score=score, local_indices=(t,))] if np.isfinite(score) else []
        )

    for i in range(1, n_events):
        next_by_end: list[list[_LocalCandidate]] = []
        running_top: list[_AdjustedCandidate] = []
        next_prefix_idx = 0
        for t in range(n_frames):
            curr_timestamp = timestamp_array[t]
            while next_prefix_idx < t and timestamp_array[next_prefix_idx] < curr_timestamp:
                for prefix in by_end[next_prefix_idx]:
                    adjusted = prefix.score + lambda_penalty * timestamp_array[next_prefix_idx]
                    _append_descending_top_k(
                        running_top,
                        _AdjustedCandidate(adjusted_score=adjusted, candidate=prefix),
                        top_k,
                    )
                next_prefix_idx += 1

            similarity_score = float(similarity[i, t])
            if not running_top or not np.isfinite(similarity_score):
                next_by_end.append([])
                continue

            end_candidates = [
                _LocalCandidate(
                    score=similarity_score
                    + prefix.adjusted_score
                    - lambda_penalty * curr_timestamp,
                    local_indices=prefix.candidate.local_indices + (t,),
                )
                for prefix in running_top
            ]
            next_by_end.append(
                [candidate for candidate in end_candidates if np.isfinite(candidate.score)][:top_k]
            )
        by_end = next_by_end

    all_candidates = [candidate for candidates in by_end for candidate in candidates]
    if not all_candidates:
        raise DanteServiceError(
            "no valid strictly-ordered sequence exists for this similarity matrix",
            error_code="no_valid_sequence",
            status_code=422,
        )

    filtered = [candidate for candidate in all_candidates if candidate.score >= threshold]
    filtered.sort(key=lambda candidate: (-candidate.score, candidate.local_indices))
    return [
        DanteResult(
            score=candidate.score,
            global_indices=_globalize(s_v, candidate.local_indices),
            debug_steps=_build_debug_steps(
                similarity,
                timestamp_array,
                candidate.local_indices,
                lambda_penalty,
            ),
        )
        for candidate in filtered[:top_k]
    ]
