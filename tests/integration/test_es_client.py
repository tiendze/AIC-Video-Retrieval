"""Integration tests for EsStore — requires the docker-compose stack (P0.2) running.

docker compose up -d
pytest -m integration tests/integration/test_es_client.py -v
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator

import pytest

from app.stores.es_client import AsrEsStore, AsrSegmentDocument, EsStore, OcrDocument

pytestmark = pytest.mark.integration

# See test_milvus_client.py for why this is overridable via env var.
_ELASTICSEARCH_URL = os.environ.get("ELASTICSEARCH_URL", "http://localhost:9200")


@pytest.fixture()
async def store() -> AsyncIterator[EsStore]:
    index_name = f"test-ocr-{uuid.uuid4().hex[:8]}"
    created = EsStore(url=_ELASTICSEARCH_URL, index_name=index_name)
    await created.ensure_index()
    yield created
    await created._client.indices.delete(index=index_name)  # test-only cleanup
    await created.close()


async def test_bulk_index_then_refresh_search_finds_document(store: EsStore) -> None:
    await store.bulk_index(
        [
            OcrDocument(
                keyframe_id="video1_0",
                ocr_text="Cộng hòa Xã hội chủ nghĩa Việt Nam",
                video_id="video1",
                timestamp=0.0,
            )
        ]
    )
    await store._client.indices.refresh(index=store.index_name)

    results = await store.search(query_text="việt nam", top_k=10)

    assert len(results) == 1
    assert results[0]["keyframe_id"] == "video1_0"


async def test_search_matches_diacritic_free_query_via_asciifolding(store: EsStore) -> None:
    await store.bulk_index(
        [
            OcrDocument(
                keyframe_id="video1_0",
                ocr_text="Cộng hòa Xã hội chủ nghĩa Việt Nam",
                video_id="video1",
                timestamp=0.0,
            )
        ]
    )
    await store._client.indices.refresh(index=store.index_name)

    results = await store.search(query_text="viet nam", top_k=10)

    assert len(results) == 1


async def test_search_video_id_filter_is_passed_to_engine_not_applied_after(
    store: EsStore,
) -> None:
    await store.bulk_index(
        [
            OcrDocument(
                keyframe_id="video_a_0", ocr_text="tin tức", video_id="video_a", timestamp=0.0
            ),
            OcrDocument(
                keyframe_id="video_b_0", ocr_text="tin tức", video_id="video_b", timestamp=0.0
            ),
        ]
    )
    await store._client.indices.refresh(index=store.index_name)

    results = await store.search(query_text="tin tức", top_k=10, video_id="video_b")

    assert len(results) == 1
    assert results[0]["video_id"] == "video_b"


@pytest.fixture()
async def asr_store() -> AsyncIterator[AsrEsStore]:
    index_name = f"test-asr-{uuid.uuid4().hex[:8]}"
    created = AsrEsStore(url=_ELASTICSEARCH_URL, index_name=index_name)
    await created.ensure_index()
    yield created
    await created._client.indices.delete(index=index_name, ignore_unavailable=True)
    await created.close()


async def test_delete_by_video_removes_only_that_video_segments(asr_store: AsrEsStore) -> None:
    await asr_store.bulk_index(
        [
            AsrSegmentDocument(
                segment_id="video_a_seg0",
                video_id="video_a",
                start_time=0.0,
                end_time=1.0,
                text="doan a",
                start_frame_idx=0,
                end_frame_idx=30,
            ),
            AsrSegmentDocument(
                segment_id="video_b_seg0",
                video_id="video_b",
                start_time=0.0,
                end_time=1.0,
                text="doan b",
                start_frame_idx=0,
                end_frame_idx=30,
            ),
        ]
    )
    await asr_store.refresh()

    await asr_store.delete_by_video("video_a")
    await asr_store.refresh()

    remaining = [doc async for doc in asr_store.scroll_all()]
    assert [doc["video_id"] for doc in remaining] == ["video_b"]


async def test_recreate_index_leaves_an_empty_but_usable_index(asr_store: AsrEsStore) -> None:
    await asr_store.bulk_index(
        [
            AsrSegmentDocument(
                segment_id="video_a_seg0",
                video_id="video_a",
                start_time=0.0,
                end_time=1.0,
                text="doan a",
                start_frame_idx=0,
                end_frame_idx=30,
            )
        ]
    )
    await asr_store.refresh()

    await asr_store.recreate_index()
    await asr_store.refresh()

    assert [doc async for doc in asr_store.scroll_all()] == []
    # index is usable again -- a fresh bulk_index doesn't error
    await asr_store.bulk_index(
        [
            AsrSegmentDocument(
                segment_id="video_c_seg0",
                video_id="video_c",
                start_time=0.0,
                end_time=1.0,
                text="doan c",
                start_frame_idx=0,
                end_frame_idx=30,
            )
        ]
    )
    await asr_store.refresh()
    assert [doc["video_id"] async for doc in asr_store.scroll_all()] == ["video_c"]
