"""Integration tests for EsStore — requires the docker-compose stack (P0.2) running.

docker compose up -d
pytest -m integration tests/integration/test_es_client.py -v
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator

import pytest

from app.stores.es_client import EsStore, OcrDocument

pytestmark = pytest.mark.integration

# See test_milvus_client.py for why this is overridable via env var.
_ELASTICSEARCH_URL = os.environ.get("ELASTICSEARCH_URL", "http://localhost:9201")


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
