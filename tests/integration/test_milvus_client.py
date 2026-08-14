"""Integration tests for MilvusStore — requires the docker-compose stack (P0.2) running.

docker compose up -d
pytest -m integration tests/integration/test_milvus_client.py -v
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator

import numpy as np
import pytest
from pymilvus import MilvusClient

from app.stores.milvus_client import (
    KeyframeVector,
    MilvusStore,
    TranscriptMilvusStore,
    TranscriptVector,
)

pytestmark = pytest.mark.integration

# Override via env var when the test runner isn't on the same network
# namespace as the docker-compose stack's published host ports (e.g. a
# separate dev container attached to the `aic-video-retrieval` network,
# where the container-internal name/port must be used instead).
_MILVUS_URI = os.environ.get("MILVUS_URI", "http://localhost:19530")
_MILVUS_TOKEN = os.environ.get("MILVUS_TOKEN", "root:Milvus")
_EMBEDDING_DIM = 8
_HNSW_M = 16
_HNSW_EF_CONSTRUCTION = 200
_SEARCH_EF = 64


def _make_store(model_id: str) -> MilvusStore:
    return MilvusStore(
        uri=_MILVUS_URI,
        token=_MILVUS_TOKEN,
        model_id=model_id,
        embedding_dim=_EMBEDDING_DIM,
        hnsw_m=_HNSW_M,
        hnsw_ef_construction=_HNSW_EF_CONSTRUCTION,
        search_ef=_SEARCH_EF,
    )


@pytest.fixture()
def raw_client() -> MilvusClient:
    return MilvusClient(uri=_MILVUS_URI, token=_MILVUS_TOKEN)


@pytest.fixture()
def store(raw_client: MilvusClient) -> Iterator[MilvusStore]:
    model_id = f"test_{uuid.uuid4().hex[:8]}"
    created = _make_store(model_id)
    yield created
    raw_client.drop_collection(created.collection_name)


def _unit_vector(hot_index: int) -> np.ndarray:
    vector = np.zeros(_EMBEDDING_DIM, dtype=np.float32)
    vector[hot_index] = 1.0
    return vector


def test_collection_name_changes_when_model_id_changes(raw_client: MilvusClient) -> None:
    model_a = f"test_model_a_{uuid.uuid4().hex[:8]}"
    model_b = f"test_model_b_{uuid.uuid4().hex[:8]}"
    store_a = _make_store(model_a)
    store_b = _make_store(model_b)
    try:
        assert store_a.collection_name != store_b.collection_name
        assert model_a in store_a.collection_name
        assert model_b in store_b.collection_name
    finally:
        raw_client.drop_collection(store_a.collection_name)
        raw_client.drop_collection(store_b.collection_name)


def test_upsert_then_search_returns_the_matching_keyframe(store: MilvusStore) -> None:
    vector = _unit_vector(hot_index=0)
    store.upsert(
        [
            KeyframeVector(
                keyframe_id="video1_0",
                embedding=vector,
                video_id="video1",
                global_index=0,
                frame_idx=0,
                timestamp=0.0,
                author="channel_a",
                batch="L21",
            )
        ]
    )

    results = store.search(query_vector=vector, top_k=1)

    assert results[0]["entity"]["video_id"] == "video1"


def test_search_returns_frame_idx_needed_for_submission(store: MilvusStore) -> None:
    # The competition answer is `<video_id>, <frame_id>`, so a search hit must
    # already carry the absolute frame index — no second lookup, no rounding
    # from timestamp * fps.
    vector = _unit_vector(hot_index=0)
    store.upsert(
        [
            KeyframeVector(
                keyframe_id="video1_0",
                embedding=vector,
                video_id="video1",
                global_index=0,
                frame_idx=1234,
                timestamp=41.13,
                author="channel_a",
                batch="L21",
            )
        ]
    )

    results = store.search(query_vector=vector, top_k=1)

    assert results[0]["entity"]["frame_idx"] == 1234


def test_search_filter_is_passed_to_engine_not_applied_after(store: MilvusStore) -> None:
    vector = _unit_vector(hot_index=0)
    store.upsert(
        [
            KeyframeVector(
                keyframe_id="video_a_0",
                embedding=vector,
                video_id="video_a",
                global_index=0,
                frame_idx=0,
                timestamp=0.0,
                author="channel_a",
                batch="L21",
            ),
            KeyframeVector(
                keyframe_id="video_b_0",
                embedding=vector,
                video_id="video_b",
                global_index=1,
                frame_idx=90,
                timestamp=0.0,
                author="channel_b",
                batch="L21",
            ),
        ]
    )

    results = store.search(query_vector=vector, top_k=10, filter_expr='video_id == "video_b"')

    assert len(results) == 1
    assert results[0]["entity"]["video_id"] == "video_b"


def test_upsert_is_idempotent_on_same_keyframe_id(store: MilvusStore) -> None:
    vector = _unit_vector(hot_index=0)
    keyframe = KeyframeVector(
        keyframe_id="video1_0",
        embedding=vector,
        video_id="video1",
        global_index=0,
        frame_idx=0,
        timestamp=0.0,
        author="channel_a",
        batch="L21",
    )

    store.upsert([keyframe])
    store.upsert([keyframe])

    results = store.search(query_vector=vector, top_k=10)
    assert len(results) == 1


def test_get_vectors_for_video_round_trips_through_real_milvus_query(store: MilvusStore) -> None:
    # plan §P2.2: TrakeService's frame-vector source. Unlike search()
    # (exercised above), this proves the `&&`-joined boolean filter
    # expression (`video_id == "..." && global_index >= N && global_index
    # <= M`) is actually valid Milvus query() syntax, not just a string the
    # Fake client happened to accept.
    store.upsert(
        [
            KeyframeVector(
                keyframe_id="video_a_10",
                embedding=_unit_vector(hot_index=0),
                video_id="video_a",
                global_index=10,
                frame_idx=100,
                timestamp=4.0,
                author="channel_a",
                batch="L21",
            ),
            KeyframeVector(
                keyframe_id="video_a_11",
                embedding=_unit_vector(hot_index=1),
                video_id="video_a",
                global_index=11,
                frame_idx=110,
                timestamp=4.4,
                author="channel_a",
                batch="L21",
            ),
            # Outside [10, 11] and a different video_id -- must not leak in.
            KeyframeVector(
                keyframe_id="video_a_12",
                embedding=_unit_vector(hot_index=2),
                video_id="video_a",
                global_index=12,
                frame_idx=120,
                timestamp=4.8,
                author="channel_a",
                batch="L21",
            ),
            KeyframeVector(
                keyframe_id="video_b_10",
                embedding=_unit_vector(hot_index=3),
                video_id="video_b",
                global_index=10,
                frame_idx=100,
                timestamp=4.0,
                author="channel_b",
                batch="L21",
            ),
        ]
    )

    rows = store.get_vectors_for_video("video_a", s_v=10, e_v=11)

    assert [r["keyframe_id"] for r in rows] == ["video_a_10", "video_a_11"]
    assert [r["global_index"] for r in rows] == [10, 11]
    assert [r["frame_idx"] for r in rows] == [100, 110]
    assert isinstance(rows[0]["embedding"], np.ndarray)
    assert rows[0]["embedding"].dtype == np.float32
    np.testing.assert_allclose(rows[0]["embedding"], _unit_vector(hot_index=0))


@pytest.fixture()
def transcript_store(raw_client: MilvusClient) -> Iterator[TranscriptMilvusStore]:
    model_id = f"test_{uuid.uuid4().hex[:8]}"
    created = TranscriptMilvusStore(
        uri=_MILVUS_URI,
        token=_MILVUS_TOKEN,
        model_id=model_id,
        embedding_dim=_EMBEDDING_DIM,
        hnsw_m=_HNSW_M,
        hnsw_ef_construction=_HNSW_EF_CONSTRUCTION,
        search_ef=_SEARCH_EF,
    )
    yield created
    raw_client.drop_collection(created.collection_name)


def test_transcript_store_drop_collection_clears_rows_but_keeps_collection_usable(
    transcript_store: TranscriptMilvusStore,
) -> None:
    vector = _unit_vector(hot_index=0)
    transcript_store.upsert(
        [
            TranscriptVector(
                segment_id="video1_seg0",
                embedding=vector,
                video_id="video1",
                start_frame_idx=0,
                end_frame_idx=30,
                start_time=0.0,
                end_time=1.0,
            )
        ]
    )
    name_before = transcript_store.collection_name

    transcript_store.drop_collection()

    assert transcript_store.collection_name == name_before  # same name, fresh collection
    results = transcript_store.search(query_vector=vector, top_k=10)
    assert results == []
    # collection is usable again -- a fresh upsert doesn't error
    transcript_store.upsert(
        [
            TranscriptVector(
                segment_id="video2_seg0",
                embedding=vector,
                video_id="video2",
                start_frame_idx=0,
                end_frame_idx=30,
                start_time=0.0,
                end_time=1.0,
            )
        ]
    )
    results = transcript_store.search(query_vector=vector, top_k=10)
    assert results[0]["entity"]["video_id"] == "video2"
