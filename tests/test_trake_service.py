"""TrakeService tests.

The service no longer fetches vectors from Milvus per request. It receives a
loaded TRAKE embedding cache, scores the full cached corpus once, then slices
that score matrix by video for the real DANTE DP.
"""

from __future__ import annotations

from typing import Any

import numpy as np

import app.services.trake_service as trake_module
from app.services.trake_embedding_cache import (
    TrakeCachedKeyframe,
    TrakeEmbeddingCacheStats,
    TrakeVideoSlice,
)
from app.services.trake_service import TrakeService

_DIM = 4


def _unit(*, hot: int, dim: int = _DIM) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    v[hot] = 1.0
    return v


class _FakeEncoder:
    """One event text -> its query vector, matched by list position."""

    def __init__(self, vectors: np.ndarray) -> None:
        self._vectors = vectors
        self.encode_text_calls: list[list[str]] = []

    def encode_text(self, texts):
        self.encode_text_calls.append(list(texts))
        return self._vectors


class _FakeTrakeEmbeddingCache:
    def __init__(
        self,
        rows_by_video: dict[str, list[dict[str, Any]]],
        videos: list[dict[str, Any]],
        embedding_dim: int,
    ) -> None:
        self.ensure_ready_calls = 0
        self.load_count = 1
        embedding_chunks: list[np.ndarray] = []
        timestamps: list[float] = []
        metadata: list[TrakeCachedKeyframe] = []
        video_slices: list[TrakeVideoSlice] = []

        for video in videos:
            if video.get("ingest_status") != "complete":
                continue
            rows = sorted(rows_by_video.get(video["video_id"], []), key=lambda r: r["global_index"])
            if not rows:
                continue
            start = len(metadata)
            embedding_chunks.append(
                np.stack([row["embedding"] for row in rows]).astype(np.float32, copy=False)
            )
            for row in rows:
                timestamps.append(float(row["timestamp"]))
                metadata.append(
                    TrakeCachedKeyframe(
                        keyframe_id=row["keyframe_id"],
                        video_id=video["video_id"],
                        global_index=row["global_index"],
                        frame_idx=row["frame_idx"],
                        timestamp=float(row["timestamp"]),
                    )
                )
            video_slices.append(
                TrakeVideoSlice(
                    video_id=video["video_id"],
                    s_v=video["s_v"],
                    e_v=video["e_v"],
                    start=start,
                    stop=len(metadata),
                )
            )

        self._embeddings = (
            np.concatenate(embedding_chunks, axis=0)
            if embedding_chunks
            else np.zeros((0, embedding_dim), dtype=np.float32)
        )
        self._timestamps = np.asarray(timestamps, dtype=np.float64)
        self._metadata = tuple(metadata)
        self._video_slices = tuple(video_slices)
        self._stats = TrakeEmbeddingCacheStats(
            num_keyframes=self._embeddings.shape[0],
            num_videos=len(self._video_slices),
            embedding_dim=embedding_dim,
            dtype=str(self._embeddings.dtype),
            embedding_memory_bytes=int(self._embeddings.nbytes),
            load_time_ms=0.0,
            source="test-cache",
        )

    def ensure_ready(self) -> None:
        self.ensure_ready_calls += 1

    @property
    def embeddings(self) -> np.ndarray:
        return self._embeddings

    @property
    def timestamps(self) -> np.ndarray:
        return self._timestamps

    @property
    def metadata(self) -> tuple[TrakeCachedKeyframe, ...]:
        return self._metadata

    @property
    def video_slices(self) -> tuple[TrakeVideoSlice, ...]:
        return self._video_slices

    @property
    def stats(self) -> TrakeEmbeddingCacheStats:
        return self._stats

    def keyframe_for_global_index(
        self, video_slice: TrakeVideoSlice, global_index: int
    ) -> TrakeCachedKeyframe:
        local_index = global_index - video_slice.s_v
        return self._metadata[video_slice.start + local_index]


def _video_doc(video_id: str, s_v: int, e_v: int, status: str = "complete") -> dict[str, Any]:
    return {"video_id": video_id, "s_v": s_v, "e_v": e_v, "ingest_status": status}


def _frame_row(
    keyframe_id: str, global_index: int, frame_idx: int, timestamp: float, hot: int
) -> dict:
    return {
        "keyframe_id": keyframe_id,
        "global_index": global_index,
        "frame_idx": frame_idx,
        "timestamp": timestamp,
        "embedding": _unit(hot=hot),
    }


class _FakeQueryTranslator:
    """Mirrors tests/test_search_service.py's own Fake -- one Vietnamese
    string in, one fixed English string out, call log kept."""

    def __init__(self, translated: str) -> None:
        self._translated = translated
        self.translate_calls: list[str] = []

    def translate(self, vietnamese: str) -> str:
        self.translate_calls.append(vietnamese)
        return self._translated


def _service(rows_by_video, videos, vectors, query_translator=None):
    encoder = _FakeEncoder(vectors)
    cache = _FakeTrakeEmbeddingCache(rows_by_video, videos, embedding_dim=vectors.shape[1])
    service = TrakeService(
        encoder=encoder, trake_cache=cache, query_translator=query_translator
    )
    return service, encoder, cache


async def test_trake_search_translates_every_event_when_a_translator_is_given():
    # Regression 2026-08-14 (real eval/run.py run against production data):
    # TrakeService encoded raw Vietnamese captions directly, the exact VI/EN
    # mismatch already diagnosed and fixed for SearchService._search_semantic
    # (English-trained text tower) -- video_level_recall_at_k was 0.0000
    # across every TRAKE query. Must translate each event BEFORE encode_text.
    events_vectors = np.stack([_unit(hot=0)])
    rows = {"video_a": [_frame_row("video_a_f0", 0, 0, 0.0, hot=0)]}
    videos = [_video_doc("video_a", 0, 0)]
    translator = _FakeQueryTranslator("translated")
    service, encoder, _ = _service(rows, videos, events_vectors, query_translator=translator)

    await service.search(["su kien tieng viet"], top_k=10)

    assert translator.translate_calls == ["su kien tieng viet"]
    assert encoder.encode_text_calls == [["translated"]]


async def test_trake_search_does_not_translate_when_no_translator_is_given():
    events_vectors = np.stack([_unit(hot=0)])
    rows = {"video_a": [_frame_row("video_a_f0", 0, 0, 0.0, hot=0)]}
    videos = [_video_doc("video_a", 0, 0)]
    service, encoder, _ = _service(rows, videos, events_vectors, query_translator=None)

    await service.search(["su kien tieng viet"], top_k=10)

    assert encoder.encode_text_calls == [["su kien tieng viet"]]


async def test_trake_search_scores_the_full_cached_corpus_before_video_slices(monkeypatch):
    events_vectors = np.stack([_unit(hot=0)])
    rows = {
        "video_a": [
            _frame_row("video_a_f0", 0, 0, 0.0, hot=0),
            _frame_row("video_a_f1", 1, 1, 1.0, hot=1),
        ],
        "video_b": [
            _frame_row("video_b_f0", 100, 0, 0.0, hot=2),
            _frame_row("video_b_f1", 101, 1, 1.0, hot=3),
        ],
    }
    videos = [_video_doc("video_a", 0, 1), _video_doc("video_b", 100, 101)]
    service, _, cache = _service(rows, videos, events_vectors)
    frame_counts_seen: list[int] = []
    real_compute_similarity_matrix = trake_module.compute_similarity_matrix

    def spy_compute_similarity_matrix(event_vectors, frame_vectors):
        frame_counts_seen.append(frame_vectors.shape[0])
        return real_compute_similarity_matrix(event_vectors, frame_vectors)

    monkeypatch.setattr(trake_module, "compute_similarity_matrix", spy_compute_similarity_matrix)

    await service.search(["a"], top_k=10)

    assert cache.stats.num_keyframes == 4
    assert frame_counts_seen == [4]


async def test_trake_search_returns_the_higher_scoring_video_first():
    # Two events, in order. video_a's frames match the events perfectly and
    # in order (hot=0 then hot=1); video_b's frames are a worse match.
    events_vectors = np.stack([_unit(hot=0), _unit(hot=1)])
    rows = {
        "video_a": [
            _frame_row("video_a_f0", 0, 0, 0.0, hot=0),
            _frame_row("video_a_f1", 1, 1, 0.1, hot=1),
        ],
        "video_b": [
            _frame_row("video_b_f0", 100, 0, 0.0, hot=2),
            _frame_row("video_b_f1", 101, 1, 0.1, hot=3),
        ],
    }
    videos = [_video_doc("video_a", 0, 1), _video_doc("video_b", 100, 101)]
    service, encoder, _ = _service(rows, videos, events_vectors)

    results = await service.search(["su kien 1", "su kien 2"], top_k=10, threshold=-1.0)

    assert [r.video_id for r in results] == ["video_a", "video_b"]
    assert results[0].score > results[1].score
    assert encoder.encode_text_calls == [["su kien 1", "su kien 2"]]


async def test_trake_search_returns_keyframes_in_strict_temporal_order():
    events_vectors = np.stack([_unit(hot=0), _unit(hot=1), _unit(hot=2)])
    rows = {
        "video_a": [
            _frame_row("video_a_f2", 2, 20, 2.0, hot=2),
            _frame_row("video_a_f0", 0, 0, 0.0, hot=0),
            _frame_row("video_a_f1", 1, 10, 1.0, hot=1),
        ]
    }
    videos = [_video_doc("video_a", 0, 2)]
    service, _, _ = _service(rows, videos, events_vectors)

    results = await service.search(["a", "b", "c"], top_k=10)

    assert len(results) == 1
    frame_indices = [kf.frame_idx for kf in results[0].keyframes]
    assert frame_indices == sorted(frame_indices)
    assert [kf.keyframe_id for kf in results[0].keyframes] == [
        "video_a_f0",
        "video_a_f1",
        "video_a_f2",
    ]


async def test_trake_search_skips_a_video_with_fewer_frames_than_events():
    events_vectors = np.stack([_unit(hot=0), _unit(hot=1), _unit(hot=2)])
    rows = {"video_a": [_frame_row("video_a_f0", 0, 0, 0.0, hot=0)]}  # only 1 frame, need 3
    videos = [_video_doc("video_a", 0, 0)]
    service, _, _ = _service(rows, videos, events_vectors)

    results = await service.search(["a", "b", "c"], top_k=10)

    assert results == []


async def test_trake_search_skips_a_video_when_dante_finds_no_finite_ordered_sequence():
    # Mirrors tests/test_dante_service.py's forced-diagonal -inf case (a
    # buggy encoder leaking -inf/NaN): TrakeService must catch
    # DanteServiceError from dante_search itself and skip+log rather than
    # crash.
    events_vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    rows = {
        "video_a": [
            {
                "keyframe_id": "video_a_f0",
                "global_index": 0,
                "frame_idx": 0,
                "timestamp": 0.0,
                "embedding": np.array([float("-inf"), 0.5], dtype=np.float32),
            },
            {
                "keyframe_id": "video_a_f1",
                "global_index": 1,
                "frame_idx": 1,
                "timestamp": 1.0,
                "embedding": np.array([0.5, float("-inf")], dtype=np.float32),
            },
        ]
    }
    videos = [_video_doc("video_a", 0, 1)]
    service, _, _ = _service(rows, videos, events_vectors)

    results = await service.search(["a", "b"], top_k=10)

    assert results == []


async def test_trake_search_never_links_candidates_from_different_videos():
    events_vectors = np.stack([_unit(hot=0), _unit(hot=1)])
    rows = {
        "video_a": [
            _frame_row("video_a_f0", 0, 0, 0.0, hot=0),
            _frame_row("video_a_f1", 1, 1, 1.0, hot=2),
        ],
        "video_b": [
            _frame_row("video_b_f0", 100, 0, 0.0, hot=2),
            _frame_row("video_b_f1", 101, 1, 1.0, hot=1),
        ],
    }
    videos = [_video_doc("video_a", 0, 1), _video_doc("video_b", 100, 101)]
    service, _, _ = _service(rows, videos, events_vectors)

    results = await service.search(["a", "b"], top_k=10, lambda_penalty=0.0, threshold=1.5)

    assert results == []


async def test_trake_search_only_uses_complete_videos_already_present_in_the_cache():
    events_vectors = np.stack([_unit(hot=0)])
    rows = {
        "video_a": [_frame_row("video_a_f0", 0, 0, 0.0, hot=0)],
        "video_b": [_frame_row("video_b_f0", 10, 0, 0.0, hot=0)],
    }
    videos = [
        _video_doc("video_a", 0, 0, status="in_progress"),
        _video_doc("video_b", 10, 10, status="complete"),
    ]
    service, _, cache = _service(rows, videos, events_vectors)

    results = await service.search(["a"], top_k=10)

    assert [result.video_id for result in results] == ["video_b"]
    assert [video_slice.video_id for video_slice in cache.video_slices] == ["video_b"]


async def test_trake_search_limits_results_to_top_k():
    events_vectors = np.stack([_unit(hot=0)])
    rows = {
        "video_a": [_frame_row("video_a_f0", 0, 0, 0.0, hot=0)],
        "video_b": [_frame_row("video_b_f0", 0, 0, 0.0, hot=0)],
        "video_c": [_frame_row("video_c_f0", 0, 0, 0.0, hot=0)],
    }
    videos = [_video_doc(v, 0, 0) for v in ("video_a", "video_b", "video_c")]
    service, _, _ = _service(rows, videos, events_vectors)

    results = await service.search(["a"], top_k=2)

    assert len(results) == 2


async def test_trake_search_returns_top_k_sequences_not_one_best_sequence_per_video():
    events_vectors = np.stack([_unit(hot=0), _unit(hot=1)])
    rows = {
        "video_a": [
            _frame_row("video_a_f0", 0, 0, 0.0, hot=0),
            _frame_row("video_a_f1", 1, 1, 1.0, hot=1),
            _frame_row("video_a_f2", 2, 2, 2.0, hot=1),
        ],
        "video_b": [
            _frame_row("video_b_f0", 100, 0, 0.0, hot=0),
            _frame_row("video_b_f1", 101, 1, 1.0, hot=2),
        ],
    }
    videos = [_video_doc("video_a", 0, 2), _video_doc("video_b", 100, 101)]
    service, _, _ = _service(rows, videos, events_vectors)

    results = await service.search(
        ["a", "b"],
        top_k=2,
        lambda_penalty=0.0,
        threshold=1.5,
    )

    assert [result.video_id for result in results] == ["video_a", "video_a"]
    assert [[kf.keyframe_id for kf in result.keyframes] for result in results] == [
        ["video_a_f0", "video_a_f1"],
        ["video_a_f0", "video_a_f2"],
    ]


async def test_trake_search_repeated_requests_reuse_the_same_loaded_cache():
    events_vectors = np.stack([_unit(hot=0)])
    rows = {"video_a": [_frame_row("video_a_f0", 0, 0, 0.0, hot=0)]}
    videos = [_video_doc("video_a", 0, 0)]
    service, _, cache = _service(rows, videos, events_vectors)

    await service.search(["a"], top_k=10)
    await service.search(["a"], top_k=10)

    assert cache.load_count == 1
    assert cache.ensure_ready_calls == 2


def _dense_video_rows(
    video_id: str, length: int, hot_at: dict[int, int], filler_hot: int
) -> list[dict]:
    """`length` CONSECUTIVE global_index rows (0..length-1)."""
    return [
        _frame_row(f"{video_id}_f{i}", i, i, float(i), hot=hot_at.get(i, filler_hot))
        for i in range(length)
    ]


async def test_trake_search_uses_the_dante_timestamp_default_lambda_when_none_given():
    from app.services.dante_service import DEFAULT_LAMBDA_PENALTY, dante_search

    events_vectors = np.stack([_unit(hot=0), _unit(hot=1)])
    video_rows = _dense_video_rows("video_a", length=6, hot_at={0: 0, 5: 1}, filler_hot=2)
    rows = {"video_a": video_rows}
    videos = [_video_doc("video_a", 0, 5)]
    service, _, _ = _service(rows, videos, events_vectors)

    results = await service.search(["a", "b"], top_k=10, lambda_penalty=None)

    similarity = np.array(
        [[events_vectors[i] @ video_rows[t]["embedding"] for t in range(6)] for i in range(2)]
    )
    expected = dante_search(
        similarity,
        s_v=0,
        timestamps=np.array([row["timestamp"] for row in video_rows], dtype=np.float64),
        lambda_penalty=DEFAULT_LAMBDA_PENALTY,
    )
    assert results[0].score == expected.score


async def test_trake_search_passing_a_larger_lambda_penalizes_a_more_spread_out_sequence():
    events_vectors = np.stack([_unit(hot=0), _unit(hot=1)])
    video_rows = _dense_video_rows("video_a", length=10, hot_at={0: 0, 9: 1}, filler_hot=2)
    rows = {"video_a": video_rows}
    videos = [_video_doc("video_a", 0, 9)]
    service, _, _ = _service(rows, videos, events_vectors)

    small_lambda = await service.search(["a", "b"], top_k=10, lambda_penalty=0.0)
    large_lambda = await service.search(["a", "b"], top_k=10, lambda_penalty=1.0, threshold=-10.0)

    assert large_lambda[0].score < small_lambda[0].score
