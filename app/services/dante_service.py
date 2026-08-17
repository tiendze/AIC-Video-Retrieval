"""DanteService — DANTE dynamic programming for TRAKE (plan §P2.1, PRD §F6,
architecture.md §3.5).

Finds the highest-scoring STRICTLY ORDERED sequence of N events inside one
video's keyframe range `[s_v, e_v]`. This is the paper's Eq (1)-(5):

    (1)  S[i,t]      = cosine_similarity(u_i, E[t])
    (2)  DP[i,t]     = S[i,t] + max_{tau in [s_v, t-1]} ( DP[i-1,tau] - lambda*(t-tau) )
    (3)  running_max = max( running_max, DP[i-1,t-1] + lambda*(t-1) )
    (4)  DP[i,t]     = S[i,t] + running_max - lambda*t
    (5)  DANTE[v]    = max_{t in [s_v, e_v]} DP[N,t]

Eq (2) is O(N*T^2) if implemented literally (a fresh max-scan over every
tau < t for every t). Eq (3)-(4) is the algebraic rewrite the paper gives to
avoid that: split lambda*(t-tau) into lambda*t - lambda*tau, move lambda*tau
into the max (so it only depends on tau, not t) and lambda*t out of it (so it
only depends on t). That turns the max into a single running accumulator
scanned left-to-right once per event row -- O(N*T) total, which is what
test_dante_service.py's runtime test empirically checks (the paper itself
never demonstrates this with numbers, plan §P2.1).

`tau < t` is enforced structurally, not by a post-hoc check: the running max
used to compute DP[i, t] is only ever updated using DP[i-1, t-1] (updated
strictly BEFORE reading it for column t), so a later event can never be
matched to an earlier-or-equal frame than the event before it (CLAUDE.md §4:
"DANTE giu rang buoc thu tu tau < t. Chuoi tra ve khong bao gio dao thu tu
thoi gian.").

`i=1` (index 0 here) is the base case: the first event has no predecessor to
respect an order constraint against, so DP[0, t] = S[0, t] directly, with no
lambda penalty applied.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "search" / "dante.yaml"


def _load_config() -> dict:
    with _CONFIG_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


_config = _load_config()
DEFAULT_LAMBDA_PENALTY: float = _config["lambda_penalty"]


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
class DanteResult:
    """`global_indices` has exactly N entries (one per event, in event
    order) and is strictly increasing -- CLAUDE.md §4's ordering invariant,
    expressed as a type-level guarantee callers can rely on rather than a
    convention they have to re-check."""

    score: float
    global_indices: tuple[int, ...]


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


def _compute_best_path_dp(
    similarity: np.ndarray, lambda_penalty: float
) -> tuple[np.ndarray, np.ndarray]:
    n_events, n_frames = similarity.shape
    dp = np.full((n_events, n_frames), -np.inf, dtype=np.float64)
    predecessor = np.full((n_events, n_frames), -1, dtype=np.int64)
    dp[0, :] = similarity[0, :]

    for i in range(1, n_events):
        running_max = -np.inf
        running_max_idx = -1
        for t in range(n_frames):
            if t > 0:
                # Eq (3): fold in tau=t-1 before computing DP[i,t], so the
                # running max only ever reflects tau values strictly less
                # than the t being computed this iteration.
                candidate = dp[i - 1, t - 1] + lambda_penalty * (t - 1)
                if candidate > running_max:
                    running_max = candidate
                    running_max_idx = t - 1
            if running_max_idx == -1:
                continue  # no valid tau < t yet (only possible at t == 0)
            # Eq (4).
            dp[i, t] = similarity[i, t] + running_max - lambda_penalty * t
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


def dante_search(
    similarity: np.ndarray,
    s_v: int,
    lambda_penalty: float = DEFAULT_LAMBDA_PENALTY,
) -> DanteResult:
    """Run the DANTE DP (Eq 2-5) over a precomputed similarity matrix for one
    video and return the best strictly-ordered N-event sequence.

    similarity: shape (N, T), row i = event i's similarity to every frame in
    `[s_v, e_v]` (column t corresponds to `global_index = s_v + t`).
    s_v: the video's start `global_index` (MongoDB `VideoDocument.s_v`,
    CLAUDE.md §3) -- used only to translate local column indices back to the
    `global_index` values callers submit.
    lambda_penalty: config-driven time penalty (`configs/search/dante.yaml`,
    CLAUDE.md §6 -- never hardcoded); defaults to that file's value.
    """
    n_events, _ = _validate_similarity(similarity)
    dp, predecessor = _compute_best_path_dp(similarity, lambda_penalty)

    last_row = dp[n_events - 1, :]
    best_t = int(np.argmax(last_row))
    best_score = float(last_row[best_t])
    if not np.isfinite(best_score):
        raise DanteServiceError(
            "no valid strictly-ordered sequence exists for this similarity matrix",
            error_code="no_valid_sequence",
            status_code=422,
        )

    global_indices = _globalize(s_v, _backtrack_best_path(predecessor, n_events, best_t))
    return DanteResult(score=best_score, global_indices=global_indices)


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
    lambda_penalty: float = DEFAULT_LAMBDA_PENALTY,
) -> list[DanteResult]:
    """Return top strictly-ordered DANTE sequences whose score is >= threshold.

    This is the k-best version of the same recurrence used by `dante_search`.
    For each possible end frame `t`, it keeps the top prefixes by
    `score + lambda*tau`; extending those prefixes by `t` gives the best
    sequences ending at `t`. Keeping only `top_k` prefixes is sufficient
    because TRAKE only needs the global top-k sequences after thresholding.
    """
    n_events, n_frames = _validate_similarity(similarity)
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
        for t in range(n_frames):
            if t > 0:
                for prefix in by_end[t - 1]:
                    adjusted = prefix.score + lambda_penalty * (t - 1)
                    _append_descending_top_k(
                        running_top,
                        _AdjustedCandidate(adjusted_score=adjusted, candidate=prefix),
                        top_k,
                    )

            similarity_score = float(similarity[i, t])
            if not running_top or not np.isfinite(similarity_score):
                next_by_end.append([])
                continue

            end_candidates = [
                _LocalCandidate(
                    score=similarity_score + prefix.adjusted_score - lambda_penalty * t,
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
        DanteResult(score=candidate.score, global_indices=_globalize(s_v, candidate.local_indices))
        for candidate in filtered[:top_k]
    ]
