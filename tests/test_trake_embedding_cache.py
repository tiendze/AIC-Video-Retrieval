"""Tests for the TRAKE process-local embedding cache."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from app.services.trake_embedding_cache import (
    TrakeEmbeddingCache,
    TrakeEmbeddingCacheError,
    compute_similarity_matrix,
)
from app.services.trake_service import TrakeService

_DIM = 4


def _unit(*, hot: int) -> np.ndarray:
    v = np.zeros(_DIM, dtype=np.float32)
    v[hot] = 1.0
    return v


def _video_doc(video_id: str, s_v: int, e_v: int, status: str = "complete") -> dict[str, Any]:
    return {"video_id": video_id, "s_v": s_v, "e_v": e_v, "ingest_status": status}


def _frame_row(
    keyframe_id: str, global_index: int, frame_idx: int, timestamp: float, hot: int
) -> dict[str, Any]:
    return {
        "keyframe_id": keyframe_id,
        "global_index": global_index,
        "frame_idx": frame_idx,
        "timestamp": timestamp,
        "embedding": _unit(hot=hot),
    }


class _FakeMilvusStore:
    collection_name = "keyframes_fgclip_base"

    def __init__(self, rows_by_video: dict[str, list[dict[str, Any]]]) -> None:
        self._rows_by_video = rows_by_video
        self.calls: list[tuple[str, int, int]] = []

    def get_vectors_for_video(self, video_id: str, s_v: int, e_v: int) -> list[dict[str, Any]]:
        self.calls.append((video_id, s_v, e_v))
        return sorted(self._rows_by_video.get(video_id, []), key=lambda r: r["global_index"])


class _FakeMongoStore:
    def __init__(self, videos: list[dict[str, Any]]) -> None:
        self._videos = videos

    async def scroll_all_videos(self):
        for video in self._videos:
            yield video


class _FakeEncoder:
    def __init__(self, vectors: np.ndarray) -> None:
        self._vectors = vectors
        self.encode_text_calls = 0

    def encode_text(self, texts):
        self.encode_text_calls += 1
        return self._vectors


async def _loaded_cache(
    rows_by_video: dict[str, list[dict[str, Any]]],
    videos: list[dict[str, Any]],
) -> tuple[TrakeEmbeddingCache, _FakeMilvusStore]:
    milvus = _FakeMilvusStore(rows_by_video)
    mongo = _FakeMongoStore(videos)
    cache = TrakeEmbeddingCache(
        milvus_store=milvus, mongo_store=mongo, expected_embedding_dim=_DIM
    )
    await cache.load()
    return cache, milvus


async def test_trake_embedding_cache_loads_complete_video_rows_once_and_reuses_them():
    rows = {
        "video_a": [
            _frame_row("video_a_f0", 0, 0, 0.0, hot=0),
            _frame_row("video_a_f1", 1, 1, 1.0, hot=1),
        ],
        "video_b": [_frame_row("video_b_f0", 10, 0, 0.0, hot=2)],
    }
    videos = [
        _video_doc("video_a", 0, 1),
        _video_doc("video_b", 10, 10, status="in_progress"),
    ]
    cache, milvus = await _loaded_cache(rows, videos)

    first_stats = cache.stats
    second_stats = await cache.load()

    assert second_stats is first_stats
    assert cache.load_count == 1
    assert milvus.calls == [("video_a", 0, 1)]
    assert cache.embeddings.shape == (2, _DIM)
    assert cache.stats.num_keyframes == 2
    assert cache.stats.num_videos == 1
    assert cache.stats.dtype == "float32"
    assert cache.stats.source == "milvus:keyframes_fgclip_base"


async def test_trake_embedding_cache_preserves_cached_semantic_scores():
    source_row = _frame_row("video_a_f0", 0, 0, 0.0, hot=2)
    cache, _ = await _loaded_cache(
        {"video_a": [source_row]},
        [_video_doc("video_a", 0, 0)],
    )
    query = np.stack([_unit(hot=2)])

    cached_score = compute_similarity_matrix(query, cache.embeddings)[0, 0]
    old_style_score = query[0] @ source_row["embedding"]

    assert cached_score == pytest.approx(old_style_score)


async def test_trake_embedding_cache_retains_full_complete_corpus_and_video_grouping():
    rows = {
        "video_a": [
            _frame_row("video_a_f0", 0, 0, 0.0, hot=0),
            _frame_row("video_a_f1", 1, 1, 1.0, hot=1),
        ],
        "video_b": [
            _frame_row("video_b_f0", 100, 0, 0.0, hot=2),
            _frame_row("video_b_f1", 101, 1, 1.0, hot=3),
            _frame_row("video_b_f2", 102, 2, 2.0, hot=0),
        ],
    }
    cache, _ = await _loaded_cache(
        rows,
        [_video_doc("video_a", 0, 1), _video_doc("video_b", 100, 102)],
    )

    assert len(cache.metadata) == 5
    assert cache.embeddings.shape[0] == 5
    assert [(s.video_id, s.start, s.stop, s.s_v, s.e_v) for s in cache.video_slices] == [
        ("video_a", 0, 2, 0, 1),
        ("video_b", 2, 5, 100, 102),
    ]
    assert cache.keyframe_for_global_index(cache.video_slices[1], 101).keyframe_id == "video_b_f1"
    assert cache.timestamps[2:5].tolist() == [0.0, 1.0, 2.0]


async def test_trake_embedding_cache_skips_video_with_global_index_gap(caplog):
    rows = {
        "video_a": [
            _frame_row("video_a_f0", 0, 0, 0.0, hot=0),
            _frame_row("video_a_f9", 9, 9, 9.0, hot=1),
        ],
        "video_b": [_frame_row("video_b_f0", 100, 0, 0.0, hot=2)],
    }
    with caplog.at_level("WARNING", logger="app.services.trake_embedding_cache"):
        cache, _ = await _loaded_cache(
            rows,
            [_video_doc("video_a", 0, 9), _video_doc("video_b", 100, 100)],
        )

    assert [video_slice.video_id for video_slice in cache.video_slices] == ["video_b"]
    assert "not complete and contiguous" in caplog.text


async def test_trake_embedding_cache_skips_video_when_a_keyframe_timestamp_is_missing(caplog):
    row_without_timestamp = _frame_row("video_a_f0", 0, 0, 0.0, hot=0)
    del row_without_timestamp["timestamp"]
    rows = {
        "video_a": [row_without_timestamp],
        "video_b": [_frame_row("video_b_f0", 100, 0, 0.0, hot=1)],
    }
    with caplog.at_level("WARNING", logger="app.services.trake_embedding_cache"):
        cache, _ = await _loaded_cache(
            rows,
            [_video_doc("video_a", 0, 0), _video_doc("video_b", 100, 100)],
        )

    assert [video_slice.video_id for video_slice in cache.video_slices] == ["video_b"]
    assert "has no timestamp" in caplog.text


async def test_trake_embedding_cache_fails_clearly_when_no_keyframes_can_be_loaded():
    milvus = _FakeMilvusStore({})
    mongo = _FakeMongoStore([_video_doc("video_a", 0, 0)])
    cache = TrakeEmbeddingCache(
        milvus_store=milvus, mongo_store=mongo, expected_embedding_dim=_DIM
    )

    with pytest.raises(TrakeEmbeddingCacheError, match="loaded zero keyframes"):
        await cache.load()


async def test_trake_request_uses_loaded_cache_without_per_video_milvus_fetching():
    rows = {"video_a": [_frame_row("video_a_f0", 0, 0, 0.0, hot=0)]}
    cache, milvus = await _loaded_cache(rows, [_video_doc("video_a", 0, 0)])
    milvus.calls.clear()
    service = TrakeService(
        encoder=_FakeEncoder(np.stack([_unit(hot=0)])),
        trake_cache=cache,
        query_translator=None,
    )

    first = await service.search(["a"], top_k=10)
    second = await service.search(["a"], top_k=10)

    assert [result.video_id for result in first] == ["video_a"]
    assert [result.video_id for result in second] == ["video_a"]
    assert milvus.calls == []
    assert cache.load_count == 1
