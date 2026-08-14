"""TrakeService -- `POST /api/trake` (plan §P2.2, PRD §F6, architecture.md
§3.5). The end-to-end bridge from N ordered event descriptions to
`dante_service.dante_search`'s per-video DP.

Encodes the events with the SAME encoder used at ingest time (CLAUDE.md §3),
then runs DANTE's DP over every ingest-"complete" video's own `[s_v, e_v]`
keyframe range -- architecture.md §3.5's "Duyệt từng video v" loop. The
paper's own design has no candidate-video pre-filter, so neither does this:
every video is scored, ranked by `DANTE[v]`, and the top-k returned. TRAKE is
not the latency-critical task (PRD §1.2 -- ranking, not speed, is what's
scored); a narrower candidate-restriction strategy is a future optimization,
not a correctness requirement.

Each event is translated VI->EN before encoding when a `QueryTranslator` is
injected -- same reasoning as `SearchService._search_semantic` (module
docstring there): BEiT-3's text tower is trained on English pairs, so raw
Vietnamese captions are off-distribution for it. Confirmed 2026-08-14 by a
real `eval/run.py` run: `video_level_recall_at_k` was 0.0000 across every
TRAKE query before this was wired in -- TrakeService (P2.2) had no
translation path at all, unlike SearchService.

A video with fewer keyframes than events, or no finite-scoring ordered
sequence, is skipped and logged -- one video's structural mismatch must not
fail the whole search (CLAUDE.md §7), mirroring
`SearchService._search_ocr` dropping a frame_idx-less hit rather than
raising.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

import numpy as np

from app.encoders.base import MultimodalEncoder
from app.services.dante_service import (
    DEFAULT_LAMBDA_PENALTY,
    DanteServiceError,
    compute_similarity_matrix,
    dante_search,
)
from app.services.query_translator import QueryTranslator
from app.stores.milvus_client import MilvusStore
from app.stores.mongo_client import MongoStore

logger = logging.getLogger(__name__)

_INGEST_COMPLETE_STATUS = "complete"


@dataclass(frozen=True)
class TrakeKeyframe:
    keyframe_id: str
    frame_idx: int
    timestamp: float


@dataclass(frozen=True)
class TrakeVideoResult:
    """`keyframes` has exactly `len(events)` entries, in strict temporal
    order (CLAUDE.md §4's DANTE ordering invariant, inherited from
    `dante_service.DanteResult.global_indices`)."""

    video_id: str
    score: float
    keyframes: tuple[TrakeKeyframe, ...]


@dataclass(frozen=True)
class _VideoRange:
    video_id: str
    s_v: int
    e_v: int


def _is_dense_global_index_range(rows: list[dict], s_v: int) -> bool:
    """`rows` is assumed already sorted ascending by `global_index`
    (`MilvusStore.get_vectors_for_video`'s contract)."""
    return all(row["global_index"] == s_v + i for i, row in enumerate(rows))


class TrakeService:
    def __init__(
        self,
        encoder: MultimodalEncoder,
        milvus_store: MilvusStore,
        mongo_store: MongoStore,
        query_translator: QueryTranslator | None = None,
    ) -> None:
        self._encoder = encoder
        self._milvus_store = milvus_store
        self._mongo_store = mongo_store
        self._query_translator = query_translator

    async def search(
        self, events: list[str], top_k: int, lambda_penalty: float | None = None
    ) -> list[TrakeVideoResult]:
        if self._query_translator is not None:
            events = await asyncio.to_thread(self._translate_events, events)
        vectors = await asyncio.to_thread(self._encoder.encode_text, events)
        videos = await self._complete_videos()
        effective_lambda = DEFAULT_LAMBDA_PENALTY if lambda_penalty is None else lambda_penalty
        results = await asyncio.to_thread(self._score_videos, vectors, videos, effective_lambda)
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def _translate_events(self, events: list[str]) -> list[str]:
        return [self._query_translator.translate(event) for event in events]

    async def _complete_videos(self) -> list[_VideoRange]:
        videos: list[_VideoRange] = []
        async for doc in self._mongo_store.scroll_all_videos():
            if doc.get("ingest_status") != _INGEST_COMPLETE_STATUS:
                continue
            videos.append(_VideoRange(video_id=doc["video_id"], s_v=doc["s_v"], e_v=doc["e_v"]))
        return videos

    def _score_videos(
        self, vectors: np.ndarray, videos: list[_VideoRange], lambda_penalty: float
    ) -> list[TrakeVideoResult]:
        results: list[TrakeVideoResult] = []
        for video in videos:
            rows = self._milvus_store.get_vectors_for_video(video.video_id, video.s_v, video.e_v)
            if len(rows) < vectors.shape[0]:
                continue
            if not _is_dense_global_index_range(rows, video.s_v):
                # dante_search reconstructs global_index as s_v + column_index
                # (CLAUDE.md §3: global_index is assigned continuously per
                # video), so a gap here would silently map the DP's answer to
                # the wrong keyframe_id -- skip and log rather than trust it.
                logger.warning(
                    "trake: skipping video_id=%s: keyframe global_index range [%d, %d] "
                    "is not contiguous (got %d row(s)) -- CLAUDE.md continuity invariant violated",
                    video.video_id,
                    video.s_v,
                    video.e_v,
                    len(rows),
                )
                continue
            frame_vectors = np.stack([row["embedding"] for row in rows])
            similarity = compute_similarity_matrix(vectors, frame_vectors)
            try:
                dp_result = dante_search(similarity, s_v=video.s_v, lambda_penalty=lambda_penalty)
            except DanteServiceError as exc:
                logger.info("trake: skipping video_id=%s: %s", video.video_id, exc)
                continue

            rows_by_global_index = {row["global_index"]: row for row in rows}
            keyframes = tuple(
                TrakeKeyframe(
                    keyframe_id=rows_by_global_index[idx]["keyframe_id"],
                    frame_idx=rows_by_global_index[idx]["frame_idx"],
                    timestamp=rows_by_global_index[idx]["timestamp"],
                )
                for idx in dp_result.global_indices
            )
            results.append(
                TrakeVideoResult(
                    video_id=video.video_id, score=dp_result.score, keyframes=keyframes
                )
            )
        return results
