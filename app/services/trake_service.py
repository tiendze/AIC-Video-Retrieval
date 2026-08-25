"""TrakeService -- `POST /api/trake` (plan §P2.2, PRD §F6, architecture.md
§3.5). The end-to-end bridge from N ordered event descriptions to
`dante_service.dante_search_sequences`'s per-video DP.

Encodes the events with the SAME encoder used at ingest time (CLAUDE.md §3),
then scores those query vectors against the process-local
`TrakeEmbeddingCache`'s full keyframe matrix. DANTE still runs per video over
each video's own `[s_v, e_v]` slice -- architecture.md §3.5's "Duyệt từng
video v" loop -- but request-time code no longer fetches the same vectors
from Milvus on every TRAKE request. Each video can contribute multiple
sequence candidates above the operator's score threshold; all candidates are
ranked together and the top-k sequences are returned.

Each event is translated VI->EN before encoding when a `QueryTranslator` is
injected -- same reasoning as `SearchService._search_semantic` (module
docstring there): the production vision-language text tower is
English-caption oriented, so raw Vietnamese captions are off-distribution for
it. Confirmed 2026-08-14 by a real `eval/run.py` run:
`video_level_recall_at_k` was 0.0000 across every TRAKE query before this was
wired in -- TrakeService (P2.2) had no translation path at all, unlike
SearchService.

A video with fewer keyframes than events, or no finite-scoring ordered
sequence, is skipped and logged -- one video's structural mismatch must not
fail the whole search (CLAUDE.md §7), mirroring
`SearchService._search_ocr` dropping a frame_idx-less hit rather than
raising.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from app.encoders.base import MultimodalEncoder
from app.services.dante_service import (
    DEFAULT_LAMBDA_PENALTY,
    DanteServiceError,
    DanteStepDebug,
    compute_similarity_matrix,
    dante_search_sequences,
)
from app.services.query_translator import QueryTranslator
from app.services.trake_embedding_cache import TrakeEmbeddingCache, TrakeVideoSlice

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "search" / "trake.yaml"


def _load_config() -> dict:
    with _CONFIG_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


_config = _load_config()
DEFAULT_SCORE_THRESHOLD: float = _config["default_score_threshold"]


@dataclass(frozen=True)
class TrakeKeyframe:
    keyframe_id: str
    frame_idx: int
    timestamp: float


@dataclass(frozen=True)
class TrakeVideoResult:
    """One sequence candidate. `keyframes` has exactly `len(events)` entries,
    in strict temporal order (CLAUDE.md §4's DANTE ordering invariant,
    inherited from `dante_service.DanteResult.global_indices`). Multiple
    candidates may share the same `video_id`."""

    video_id: str
    score: float
    keyframes: tuple[TrakeKeyframe, ...]
    debug_steps: tuple[DanteStepDebug, ...] = ()


class TrakeService:
    def __init__(
        self,
        encoder: MultimodalEncoder,
        trake_cache: TrakeEmbeddingCache,
        query_translator: QueryTranslator | None = None,
    ) -> None:
        self._encoder = encoder
        self._trake_cache = trake_cache
        self._query_translator = query_translator

    async def search(
        self,
        events: list[str],
        top_k: int,
        lambda_penalty: float | None = None,
        threshold: float = DEFAULT_SCORE_THRESHOLD,
    ) -> list[TrakeVideoResult]:
        total_start = time.perf_counter()
        self._trake_cache.ensure_ready()

        translate_start = time.perf_counter()
        if self._query_translator is not None:
            events = await asyncio.to_thread(self._translate_events, events)
        translate_ms = (time.perf_counter() - translate_start) * 1000

        encode_start = time.perf_counter()
        vectors = await asyncio.to_thread(self._encoder.encode_text, events)
        text_encode_ms = (time.perf_counter() - encode_start) * 1000

        similarity_start = time.perf_counter()
        scores = await asyncio.to_thread(
            compute_similarity_matrix, vectors, self._trake_cache.embeddings
        )
        similarity_ms = (time.perf_counter() - similarity_start) * 1000

        effective_lambda = DEFAULT_LAMBDA_PENALTY if lambda_penalty is None else lambda_penalty
        dante_start = time.perf_counter()
        results = await asyncio.to_thread(
            self._score_cached_videos, scores, effective_lambda, threshold, top_k
        )
        dante_ms = (time.perf_counter() - dante_start) * 1000

        sort_start = time.perf_counter()
        results.sort(key=lambda r: r.score, reverse=True)
        top_results = results[:top_k]
        sort_ms = (time.perf_counter() - sort_start) * 1000
        total_ms = (time.perf_counter() - total_start) * 1000
        cache_stats = self._trake_cache.stats
        logger.info(
            "[TRAKE TIMING] translate_ms=%.1f text_encode_ms=%.1f similarity_ms=%.1f "
            "dante_ms=%.1f sort_ms=%.1f total_ms=%.1f events=%d cache_keyframes=%d "
            "cache_videos=%d results=%d",
            translate_ms,
            text_encode_ms,
            similarity_ms,
            dante_ms,
            sort_ms,
            total_ms,
            len(events),
            cache_stats.num_keyframes,
            cache_stats.num_videos,
            len(top_results),
        )
        return top_results

    def _translate_events(self, events: list[str]) -> list[str]:
        return [self._query_translator.translate(event) for event in events]

    def _score_cached_videos(
        self,
        scores: np.ndarray,
        lambda_penalty: float,
        threshold: float,
        top_k: int,
    ) -> list[TrakeVideoResult]:
        results: list[TrakeVideoResult] = []
        n_events = scores.shape[0]
        for video_slice in self._trake_cache.video_slices:
            if video_slice.length < n_events:
                continue
            similarity = scores[:, video_slice.start : video_slice.stop]
            timestamps = self._trake_cache.timestamps[video_slice.start : video_slice.stop]
            try:
                dp_results = dante_search_sequences(
                    similarity,
                    s_v=video_slice.s_v,
                    threshold=threshold,
                    top_k=top_k,
                    timestamps=timestamps,
                    lambda_penalty=lambda_penalty,
                )
            except DanteServiceError as exc:
                logger.info("trake: skipping video_id=%s: %s", video_slice.video_id, exc)
                continue

            for dp_result in dp_results:
                keyframes = tuple(
                    self._to_trake_keyframe(video_slice, idx)
                    for idx in dp_result.global_indices
                )
                results.append(
                    TrakeVideoResult(
                        video_id=video_slice.video_id,
                        score=dp_result.score,
                        keyframes=keyframes,
                        debug_steps=dp_result.debug_steps,
                    )
                )
        return results

    def _to_trake_keyframe(
        self, video_slice: TrakeVideoSlice, global_index: int
    ) -> TrakeKeyframe:
        cached = self._trake_cache.keyframe_for_global_index(video_slice, global_index)
        return TrakeKeyframe(
            keyframe_id=cached.keyframe_id,
            frame_idx=cached.frame_idx,
            timestamp=cached.timestamp,
        )
