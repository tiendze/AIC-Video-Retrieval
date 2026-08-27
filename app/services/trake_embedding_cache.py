"""In-memory TRAKE embedding cache.

TRAKE's old request path fetched every video's keyframe embeddings from
Milvus for every `/api/trake` call. This cache moves that vector transfer to
FastAPI startup: each process loads the complete TRAKE corpus once, keeps only
embeddings plus DANTE/result metadata in RAM, and request-time code slices the
preloaded matrix by video.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

import numpy as np

from app.stores.milvus_client import MilvusStore
from app.stores.mongo_client import MongoStore

logger = logging.getLogger(__name__)

_INGEST_COMPLETE_STATUS = "complete"


def compute_similarity_matrix(event_vectors: np.ndarray, frame_vectors: np.ndarray) -> np.ndarray:
    """S[i, t] = cosine_similarity(event_vectors[i], frame_vectors[t]).

    Both inputs are assumed already L2-normalized by the encoder (CLAUDE.md §3),
    so the plain dot product IS the cosine similarity. Re-normalizing here would
    silently paper over an encoder that violates that invariant instead of
    surfacing it.

    event_vectors: shape (N, dim); frame_vectors: shape (T, dim).
    Returns: shape (N, T).
    """
    return event_vectors @ frame_vectors.T


class TrakeEmbeddingCacheError(Exception):
    """Raised when the TRAKE cache cannot serve a request.

    Carries API-facing metadata so the router can return a clear 503 instead
    of silently falling back to the old per-video Milvus loop.
    """

    def __init__(
        self,
        message: str,
        error_code: str = "trake_cache_unavailable",
        status_code: int = 503,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.status_code = status_code


@dataclass(frozen=True)
class TrakeCachedKeyframe:
    keyframe_id: str
    video_id: str
    global_index: int
    frame_idx: int
    timestamp: float


@dataclass(frozen=True)
class TrakeVideoSlice:
    video_id: str
    s_v: int
    e_v: int
    start: int
    stop: int

    @property
    def length(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True)
class TrakeEmbeddingCacheStats:
    num_keyframes: int
    num_videos: int
    embedding_dim: int
    dtype: str
    embedding_memory_bytes: int
    load_time_ms: float
    source: str


@dataclass(frozen=True)
class _VideoRange:
    video_id: str
    s_v: int
    e_v: int


@dataclass(frozen=True)
class _LoadedCacheData:
    embeddings: np.ndarray
    timestamps: np.ndarray
    metadata: tuple[TrakeCachedKeyframe, ...]
    video_slices: tuple[TrakeVideoSlice, ...]
    source: str


def _is_complete_dense_global_index_range(rows: list[dict], s_v: int, e_v: int) -> bool:
    """Rows are assumed sorted ascending by global_index."""
    expected_count = e_v - s_v + 1
    if expected_count < 0 or len(rows) != expected_count:
        return False
    return all(row["global_index"] == s_v + i for i, row in enumerate(rows))


def _timestamp_seconds_for_rows(rows: list[dict], video_id: str) -> np.ndarray | None:
    timestamps: list[float] = []
    for row in rows:
        raw_timestamp = row.get("timestamp")
        if raw_timestamp is None:
            logger.warning(
                "trake cache: skipping video_id=%s: keyframe_id=%s has no timestamp",
                video_id,
                row.get("keyframe_id", "<unknown>"),
            )
            return None
        timestamp = float(raw_timestamp)
        if not np.isfinite(timestamp):
            logger.warning(
                "trake cache: skipping video_id=%s: keyframe_id=%s has non-finite timestamp=%r",
                video_id,
                row.get("keyframe_id", "<unknown>"),
                raw_timestamp,
            )
            return None
        timestamps.append(timestamp)
    return np.asarray(timestamps, dtype=np.float64)


class TrakeEmbeddingCache:
    """Process-local TRAKE corpus cache.

    Uvicorn's documented command in this project runs a single worker. If the
    operator starts multiple workers later, each worker process will own one
    independent copy of this cache.
    """

    def __init__(
        self,
        milvus_store: MilvusStore,
        mongo_store: MongoStore,
        expected_embedding_dim: int | None = None,
    ) -> None:
        self._milvus_store = milvus_store
        self._mongo_store = mongo_store
        self._expected_embedding_dim = expected_embedding_dim
        self._load_lock = asyncio.Lock()
        self._embeddings: np.ndarray | None = None
        self._timestamps: np.ndarray | None = None
        self._metadata: tuple[TrakeCachedKeyframe, ...] = ()
        self._video_slices: tuple[TrakeVideoSlice, ...] = ()
        self._stats: TrakeEmbeddingCacheStats | None = None
        self._load_count = 0

    @property
    def is_loaded(self) -> bool:
        return self._embeddings is not None

    @property
    def load_count(self) -> int:
        return self._load_count

    @property
    def stats(self) -> TrakeEmbeddingCacheStats:
        self.ensure_ready()
        assert self._stats is not None
        return self._stats

    @property
    def embeddings(self) -> np.ndarray:
        self.ensure_ready()
        assert self._embeddings is not None
        return self._embeddings

    @property
    def timestamps(self) -> np.ndarray:
        self.ensure_ready()
        assert self._timestamps is not None
        return self._timestamps

    @property
    def metadata(self) -> tuple[TrakeCachedKeyframe, ...]:
        self.ensure_ready()
        return self._metadata

    @property
    def video_slices(self) -> tuple[TrakeVideoSlice, ...]:
        self.ensure_ready()
        return self._video_slices

    def ensure_ready(self) -> None:
        if self._embeddings is None:
            raise TrakeEmbeddingCacheError("TRAKE embedding cache is unavailable")

    async def load(self) -> TrakeEmbeddingCacheStats:
        """Load once, then return the already-built stats on later calls."""
        if self.is_loaded:
            return self.stats

        async with self._load_lock:
            if self.is_loaded:
                return self.stats

            start = time.perf_counter()
            videos = await self._complete_videos()
            loaded = await asyncio.to_thread(self._load_from_milvus, videos)
            load_time_ms = (time.perf_counter() - start) * 1000
            self._install_loaded_data(loaded, load_time_ms)
            self._load_count += 1
            assert self._stats is not None
            logger.info(
                "[TRAKE CACHE] keyframes=%d embedding_dim=%d dtype=%s "
                "embedding_memory=%.1f MB load_time=%.2f s videos=%d source=%s",
                self._stats.num_keyframes,
                self._stats.embedding_dim,
                self._stats.dtype,
                self._stats.embedding_memory_bytes / (1024 * 1024),
                self._stats.load_time_ms / 1000,
                self._stats.num_videos,
                self._stats.source,
            )
            return self._stats

    async def _complete_videos(self) -> list[_VideoRange]:
        videos: list[_VideoRange] = []
        async for doc in self._mongo_store.scroll_all_videos():
            if doc.get("ingest_status") != _INGEST_COMPLETE_STATUS:
                continue
            videos.append(_VideoRange(video_id=doc["video_id"], s_v=doc["s_v"], e_v=doc["e_v"]))
        return videos

    def _load_from_milvus(self, videos: list[_VideoRange]) -> _LoadedCacheData:
        embedding_chunks: list[np.ndarray] = []
        timestamp_chunks: list[np.ndarray] = []
        metadata: list[TrakeCachedKeyframe] = []
        video_slices: list[TrakeVideoSlice] = []

        for video in videos:
            rows = self._milvus_store.get_vectors_for_video(video.video_id, video.s_v, video.e_v)
            if not rows:
                logger.warning(
                    "trake cache: skipping video_id=%s: Milvus returned no keyframe vectors",
                    video.video_id,
                )
                continue
            if not _is_complete_dense_global_index_range(rows, video.s_v, video.e_v):
                logger.warning(
                    "trake cache: skipping video_id=%s: keyframe global_index range [%d, %d] "
                    "is not complete and contiguous (got %d row(s))",
                    video.video_id,
                    video.s_v,
                    video.e_v,
                    len(rows),
                )
                continue

            timestamps = _timestamp_seconds_for_rows(rows, video.video_id)
            if timestamps is None:
                continue

            try:
                frame_vectors = np.stack([row["embedding"] for row in rows]).astype(
                    np.float32, copy=False
                )
            except ValueError as exc:
                raise TrakeEmbeddingCacheError(
                    f"TRAKE embedding cache failed for video_id={video.video_id}: "
                    "Milvus returned inconsistent embedding shapes",
                    error_code="trake_cache_invalid_embedding_shape",
                    status_code=503,
                ) from exc

            if frame_vectors.ndim != 2:
                raise TrakeEmbeddingCacheError(
                    f"TRAKE embedding cache failed for video_id={video.video_id}: "
                    f"expected 2D embeddings, got shape {frame_vectors.shape}",
                    error_code="trake_cache_invalid_embedding_shape",
                    status_code=503,
                )
            if (
                self._expected_embedding_dim is not None
                and frame_vectors.shape[1] != self._expected_embedding_dim
            ):
                raise TrakeEmbeddingCacheError(
                    f"TRAKE embedding cache dimension mismatch for video_id={video.video_id}: "
                    f"expected {self._expected_embedding_dim}, got {frame_vectors.shape[1]}",
                    error_code="trake_cache_encoder_mismatch",
                    status_code=503,
                )

            start = len(metadata)
            stop = start + len(rows)
            embedding_chunks.append(frame_vectors)
            timestamp_chunks.append(timestamps)
            metadata.extend(
                TrakeCachedKeyframe(
                    keyframe_id=row["keyframe_id"],
                    video_id=video.video_id,
                    global_index=row["global_index"],
                    frame_idx=row["frame_idx"],
                    timestamp=float(row["timestamp"]),
                )
                for row in rows
            )
            video_slices.append(
                TrakeVideoSlice(
                    video_id=video.video_id,
                    s_v=video.s_v,
                    e_v=video.e_v,
                    start=start,
                    stop=stop,
                )
            )

        if not embedding_chunks:
            raise TrakeEmbeddingCacheError(
                "TRAKE embedding cache loaded zero keyframes from Milvus",
                error_code="trake_cache_empty",
                status_code=503,
            )

        embeddings = np.concatenate(embedding_chunks, axis=0).astype(np.float32, copy=False)
        timestamps = np.concatenate(timestamp_chunks, axis=0).astype(np.float64, copy=False)
        return _LoadedCacheData(
            embeddings=embeddings,
            timestamps=timestamps,
            metadata=tuple(metadata),
            video_slices=tuple(video_slices),
            source=f"milvus:{self._milvus_store.collection_name}",
        )

    def _install_loaded_data(self, loaded: _LoadedCacheData, load_time_ms: float) -> None:
        self._embeddings = loaded.embeddings
        self._timestamps = loaded.timestamps
        self._metadata = loaded.metadata
        self._video_slices = loaded.video_slices
        self._stats = TrakeEmbeddingCacheStats(
            num_keyframes=loaded.embeddings.shape[0],
            num_videos=len(loaded.video_slices),
            embedding_dim=loaded.embeddings.shape[1],
            dtype=str(loaded.embeddings.dtype),
            embedding_memory_bytes=int(loaded.embeddings.nbytes),
            load_time_ms=load_time_ms,
            source=loaded.source,
        )

    def keyframe_for_global_index(
        self, video_slice: TrakeVideoSlice, global_index: int
    ) -> TrakeCachedKeyframe:
        self.ensure_ready()
        local_index = global_index - video_slice.s_v
        corpus_index = video_slice.start + local_index
        if local_index < 0 or corpus_index >= video_slice.stop:
            raise TrakeEmbeddingCacheError(
                f"TRAKE cache metadata lookup failed for video_id={video_slice.video_id}, "
                f"global_index={global_index}",
                error_code="trake_cache_invalid_metadata",
                status_code=503,
            )
        return self._metadata[corpus_index]
