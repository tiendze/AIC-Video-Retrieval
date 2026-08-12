"""RED reproducer for SearchService (P0.8, plan §P0.8) -- the first
end-to-end query bridge. `mode` is a parameter of one function, not two
copy-pasted code paths (CLAUDE.md §4): `semantic` embeds the query with the
injected encoder and searches Milvus; `ocr` full-text-searches Elasticsearch
and must join to MongoDB afterward because ES OCR hits carry no `frame_idx`
at all (CLAUDE.md §3 only requires frame_idx in Milvus/MongoDB).

Fakes stand in for every store/encoder -- no Docker, no GPU needed here; see
tests/integration/ for the real-server counterparts.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from PIL import Image

from app.services.search_service import SearchHit, SearchService, SearchServiceError


class _FakeEncoder:
    def __init__(self, vector: np.ndarray) -> None:
        self._vector = vector
        self.encode_text_calls: list[list[str]] = []
        self.encode_image_calls: list[list[Any]] = []

    @property
    def embedding_dim(self) -> int:
        return self._vector.shape[-1]

    @property
    def model_id(self) -> str:
        return "fake-encoder"

    def encode_image(self, images):
        self.encode_image_calls.append(list(images))
        return self._vector

    def encode_text(self, texts):
        self.encode_text_calls.append(list(texts))
        return self._vector


class _FakeMilvusStore:
    def __init__(self, hits: list[dict[str, Any]]) -> None:
        self._hits = hits
        self.search_calls: list[tuple[np.ndarray, int]] = []

    def search(self, query_vector: np.ndarray, top_k: int, filter_expr: str = "") -> list[dict]:
        self.search_calls.append((query_vector, top_k))
        return self._hits


class _FakeEsStore:
    def __init__(self, hits: list[dict[str, Any]]) -> None:
        self._hits = hits
        self.search_calls: list[tuple[str, int]] = []

    async def search(self, query_text: str, top_k: int, video_id: str | None = None) -> list[dict]:
        self.search_calls.append((query_text, top_k))
        return self._hits


class _FakeMongoStore:
    def __init__(self, docs_by_id: dict[str, dict[str, Any]]) -> None:
        self._docs_by_id = docs_by_id
        self.get_keyframes_by_ids_calls: list[list[str]] = []

    async def get_keyframes_by_ids(self, keyframe_ids) -> dict[str, dict[str, Any]]:
        self.get_keyframes_by_ids_calls.append(list(keyframe_ids))
        return {kid: self._docs_by_id[kid] for kid in keyframe_ids if kid in self._docs_by_id}


class _FakeQueryTranslator:
    def __init__(self, translated: str) -> None:
        self._translated = translated
        self.translate_calls: list[str] = []

    @property
    def model_id(self) -> str:
        return "fake-translator"

    def translate(self, vietnamese: str) -> str:
        self.translate_calls.append(vietnamese)
        return self._translated


def _service(
    milvus_hits: list[dict] | None = None,
    es_hits: list[dict] | None = None,
    mongo_docs: dict[str, dict] | None = None,
    query_vector: np.ndarray | None = None,
    query_translator: _FakeQueryTranslator | None = None,
) -> tuple[SearchService, _FakeEncoder, _FakeMilvusStore, _FakeEsStore, _FakeMongoStore]:
    default_vector = np.zeros((1, 4), dtype=np.float32)
    encoder = _FakeEncoder(query_vector if query_vector is not None else default_vector)
    milvus_store = _FakeMilvusStore(milvus_hits or [])
    es_store = _FakeEsStore(es_hits or [])
    mongo_store = _FakeMongoStore(mongo_docs or {})
    service = SearchService(
        encoder=encoder,
        milvus_store=milvus_store,
        es_store=es_store,
        mongo_store=mongo_store,
        query_translator=query_translator,
    )
    return service, encoder, milvus_store, es_store, mongo_store


async def test_semantic_mode_embeds_query_and_searches_milvus_with_the_query_vector():
    vector = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
    milvus_hits = [
        {
            "keyframe_id": "L21_V001_f10",
            "distance": 0.87,
            "entity": {"video_id": "L21_V001", "frame_idx": 10, "timestamp": 0.4},
        }
    ]
    mongo_docs = {"L21_V001_f10": {"thumbnail_path": "thumbs/L21_V001_f10.jpg"}}
    service, encoder, milvus_store, _, mongo_store = _service(
        milvus_hits=milvus_hits, mongo_docs=mongo_docs, query_vector=vector
    )

    results = await service.search("thu tuong", mode="semantic", top_k=5)

    assert encoder.encode_text_calls == [["thu tuong"]]
    assert milvus_store.search_calls[0][1] == 5
    assert np.array_equal(milvus_store.search_calls[0][0], vector[0])
    assert mongo_store.get_keyframes_by_ids_calls == [["L21_V001_f10"]]
    assert results == [
        SearchHit(
            keyframe_id="L21_V001_f10",
            video_id="L21_V001",
            score=0.87,
            frame_idx=10,
            timestamp=0.4,
            thumbnail_path="thumbs/L21_V001_f10.jpg",
        )
    ]


async def test_semantic_mode_never_drops_a_hit_missing_from_mongo():
    milvus_hits = [
        {
            "keyframe_id": "orphan_f1",
            "distance": 0.5,
            "entity": {"video_id": "L21_V001", "frame_idx": 1, "timestamp": 0.04},
        }
    ]
    service, *_ = _service(milvus_hits=milvus_hits, mongo_docs={})

    results = await service.search("query", mode="semantic", top_k=5)

    assert len(results) == 1
    assert results[0].keyframe_id == "orphan_f1"
    assert results[0].frame_idx == 1  # already carried by Milvus, not lost
    assert results[0].thumbnail_path == ""


async def test_ocr_mode_searches_elasticsearch_and_enriches_with_mongo_frame_idx():
    es_hits = [
        {
            "keyframe_id": "L21_V001_f25",
            "ocr_text": "tin tuc",
            "video_id": "L21_V001",
            "timestamp": 1.0,
            "score": 3.2,
        }
    ]
    mongo_docs = {"L21_V001_f25": {"frame_idx": 25, "thumbnail_path": "thumbs/L21_V001_f25.jpg"}}
    service, _, _, es_store, mongo_store = _service(es_hits=es_hits, mongo_docs=mongo_docs)

    results = await service.search("tin tuc", mode="ocr", top_k=5)

    assert es_store.search_calls == [("tin tuc", 5)]
    assert mongo_store.get_keyframes_by_ids_calls == [["L21_V001_f25"]]
    assert results == [
        SearchHit(
            keyframe_id="L21_V001_f25",
            video_id="L21_V001",
            score=3.2,
            frame_idx=25,
            timestamp=1.0,
            thumbnail_path="thumbs/L21_V001_f25.jpg",
        )
    ]


async def test_ocr_mode_drops_and_does_not_raise_a_hit_missing_from_mongo():
    # frame_idx is mandatory for every keyframe result (CLAUDE.md §3) and ES
    # OCR hits never carry it -- a keyframe_id that can't be resolved in
    # Mongo has no frame_idx to report, so it must be dropped, not raised.
    es_hits = [
        {
            "keyframe_id": "missing_from_mongo",
            "ocr_text": "x",
            "video_id": "L21_V001",
            "timestamp": 2.0,
            "score": 1.1,
        },
        {
            "keyframe_id": "found",
            "ocr_text": "y",
            "video_id": "L21_V001",
            "timestamp": 3.0,
            "score": 0.9,
        },
    ]
    mongo_docs = {"found": {"frame_idx": 30, "thumbnail_path": "t.jpg"}}
    service, *_ = _service(es_hits=es_hits, mongo_docs=mongo_docs)

    results = await service.search("x", mode="ocr", top_k=5)

    assert [hit.keyframe_id for hit in results] == ["found"]


async def test_unknown_mode_raises_search_service_error():
    service, *_ = _service()

    with pytest.raises(SearchServiceError):
        await service.search("query", mode="video", top_k=5)  # type: ignore[arg-type]


async def test_semantic_mode_translates_query_before_encoding_when_translator_configured():
    # Confirmed 2026-08-11 (reports/20260811T080907Z_fc0edde): scoring the
    # same BEiT-3 collection with English queries lifted AIC Final Score from
    # 0.0451 to 0.6208 -- the text tower is English-only, translate at query
    # time rather than re-embed the corpus.
    translator = _FakeQueryTranslator("a scene of a burnt forest")
    service, encoder, *_ = _service(query_translator=translator)

    await service.search("một khu rừng bị cháy", mode="semantic", top_k=5)

    assert translator.translate_calls == ["một khu rừng bị cháy"]
    assert encoder.encode_text_calls == [["a scene of a burnt forest"]]


async def test_ocr_mode_never_translates_the_query():
    # The `ocr_keyframes` ES index holds Vietnamese OCR text -- translating
    # an OCR query would measure translation loss, not the retrieval channel.
    translator = _FakeQueryTranslator("should never be used")
    service, _, _, es_store, _ = _service(es_hits=[], query_translator=translator)

    await service.search("tin tuc", mode="ocr", top_k=5)

    assert translator.translate_calls == []
    assert es_store.search_calls == [("tin tuc", 5)]


async def test_search_by_image_encodes_with_vision_branch_and_searches_milvus():
    # plan §P1.1 / US-3: image-to-image search must encode with the vision
    # branch of the SAME encoder used at ingest, then search the SAME
    # Milvus collection semantic text queries hit -- one shared vector space.
    vector = np.array([[0.5, 0.6, 0.7, 0.8]], dtype=np.float32)
    milvus_hits = [
        {
            "keyframe_id": "L21_V001_f10",
            "distance": 0.99,
            "entity": {"video_id": "L21_V001", "frame_idx": 10, "timestamp": 0.4},
        }
    ]
    mongo_docs = {"L21_V001_f10": {"thumbnail_path": "thumbs/L21_V001_f10.jpg"}}
    service, encoder, milvus_store, _, mongo_store = _service(
        milvus_hits=milvus_hits, mongo_docs=mongo_docs, query_vector=vector
    )
    image = Image.new("RGB", (8, 8), color=(10, 20, 30))

    results = await service.search_by_image(image, top_k=5)

    assert encoder.encode_image_calls == [[image]]
    assert milvus_store.search_calls[0][1] == 5
    assert np.array_equal(milvus_store.search_calls[0][0], vector[0])
    assert mongo_store.get_keyframes_by_ids_calls == [["L21_V001_f10"]]
    assert results == [
        SearchHit(
            keyframe_id="L21_V001_f10",
            video_id="L21_V001",
            score=0.99,
            frame_idx=10,
            timestamp=0.4,
            thumbnail_path="thumbs/L21_V001_f10.jpg",
        )
    ]


async def test_search_by_image_never_calls_the_query_translator():
    # Translation only applies to the `semantic` text path -- an uploaded
    # image has no text query to translate.
    translator = _FakeQueryTranslator("should never be used")
    service, encoder, *_ = _service(query_translator=translator)
    image = Image.new("RGB", (4, 4))

    await service.search_by_image(image, top_k=5)

    assert translator.translate_calls == []
    assert encoder.encode_text_calls == []


async def test_search_by_image_never_drops_a_hit_missing_from_mongo():
    milvus_hits = [
        {
            "keyframe_id": "orphan_f1",
            "distance": 0.5,
            "entity": {"video_id": "L21_V001", "frame_idx": 1, "timestamp": 0.04},
        }
    ]
    service, *_ = _service(milvus_hits=milvus_hits, mongo_docs={})

    results = await service.search_by_image(Image.new("RGB", (4, 4)), top_k=5)

    assert len(results) == 1
    assert results[0].keyframe_id == "orphan_f1"
    assert results[0].thumbnail_path == ""


# NOTE: hybrid-mode RRF fusion tests were removed here 2026-08-12 along with
# the `hybrid` SearchService mode itself (see app/services/search_service.py
# module docstring for why). `_reciprocal_rank_fusion` itself is untested by
# this file now -- it's an unused-for-now pure utility kept for a future
# same-kind-channel fusion (docs/research-encoder-vietnamese.md §8.3); add
# coverage back when a caller wires it.
