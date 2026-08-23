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
from elasticsearch import ConnectionError as EsConnectionError
from PIL import Image

from app.services.abe import AbeQueryComponents, AbeRankedQuery
from app.services.kis_scoring import (
    AbePair,
    AbePairScore,
    KisConfig,
    KisQuery,
    QueryType,
    RankedScore,
)
from app.services.search_service import SearchHit, SearchService, SearchServiceError


class _FakeEncoder:
    def __init__(
        self,
        vector: np.ndarray,
        text_vectors_by_query: dict[str, np.ndarray] | None = None,
        model_id: str = "fake-encoder",
    ) -> None:
        self._vector = vector
        self._text_vectors_by_query = text_vectors_by_query or {}
        self._model_id = model_id
        self.encode_text_calls: list[list[str]] = []
        self.encode_image_calls: list[list[Any]] = []

    @property
    def embedding_dim(self) -> int:
        if self._text_vectors_by_query:
            return next(iter(self._text_vectors_by_query.values())).shape[-1]
        return self._vector.shape[-1]

    @property
    def model_id(self) -> str:
        return self._model_id

    def encode_image(self, images):
        self.encode_image_calls.append(list(images))
        return self._vector

    def encode_text(self, texts):
        self.encode_text_calls.append(list(texts))
        if self._text_vectors_by_query:
            return np.asarray(
                [self._text_vectors_by_query[text] for text in texts], dtype=np.float32
            )
        return self._vector


class _FakeMilvusStore:
    def __init__(
        self,
        hits: list[dict[str, Any]],
        vector_rows: dict[str, dict[str, Any]] | None = None,
        hits_by_call: list[list[dict[str, Any]]] | None = None,
    ) -> None:
        self._hits = hits
        self._vector_rows = vector_rows or {}
        self._hits_by_call = hits_by_call
        self.search_calls: list[tuple[np.ndarray, int]] = []
        self.get_vectors_by_ids_calls: list[list[str]] = []

    def search(self, query_vector: np.ndarray, top_k: int, filter_expr: str = "") -> list[dict]:
        self.search_calls.append((query_vector, top_k))
        if self._hits_by_call is not None:
            return self._hits_by_call[len(self.search_calls) - 1]
        return self._hits

    def get_vectors_by_ids(self, keyframe_ids) -> dict[str, dict[str, Any]]:
        self.get_vectors_by_ids_calls.append(list(keyframe_ids))
        return {
            keyframe_id: self._vector_rows[keyframe_id]
            for keyframe_id in keyframe_ids
            if keyframe_id in self._vector_rows
        }


class _FakeEsStore:
    def __init__(
        self,
        hits: list[dict[str, Any]],
        texts_by_id: dict[str, str] | None = None,
        scores_by_query: dict[str, dict[str, float]] | None = None,
    ) -> None:
        self._hits = hits
        self._texts_by_id = texts_by_id or {}
        self._scores_by_query = scores_by_query or {}
        self.search_calls: list[tuple[str, int]] = []
        self.get_by_ids_calls: list[list[str]] = []
        self.score_by_ids_calls: list[tuple[str, list[str]]] = []

    async def search(self, query_text: str, top_k: int, video_id: str | None = None) -> list[dict]:
        self.search_calls.append((query_text, top_k))
        return self._hits

    async def get_by_ids(self, keyframe_ids) -> dict[str, str]:
        self.get_by_ids_calls.append(list(keyframe_ids))
        return {
            keyframe_id: self._texts_by_id[keyframe_id]
            for keyframe_id in keyframe_ids
            if keyframe_id in self._texts_by_id
        }

    async def score_by_ids(self, query_text: str, keyframe_ids) -> dict[str, float]:
        self.score_by_ids_calls.append((query_text, list(keyframe_ids)))
        return {
            keyframe_id: self._scores_by_query.get(query_text, {}).get(keyframe_id, 0.0)
            for keyframe_id in keyframe_ids
            if self._scores_by_query.get(query_text, {}).get(keyframe_id, 0.0) > 0
        }


class _FakeEsStoreDown:
    """Simulates Elasticsearch unreachable mid-request (plan §P1.8) --
    distinct from `_FakeEsStore` (which returns real hits) so a test can
    choose which failure mode it wants."""

    async def search(self, query_text: str, top_k: int, video_id: str | None = None) -> list[dict]:
        raise EsConnectionError("simulated ES down")


class _FakeMongoStore:
    def __init__(
        self,
        docs_by_id: dict[str, dict[str, Any]],
        videos_by_title: list[dict[str, Any]] | None = None,
        keyframes_by_video: dict[str, list[dict[str, Any]]] | None = None,
    ) -> None:
        self._docs_by_id = docs_by_id
        self._videos_by_title = videos_by_title or []
        self._keyframes_by_video = keyframes_by_video or {}
        self.get_keyframes_by_ids_calls: list[list[str]] = []
        self.search_videos_by_title_calls: list[tuple[str, int]] = []
        self.get_keyframes_for_video_calls: list[str] = []

    async def get_keyframes_by_ids(self, keyframe_ids) -> dict[str, dict[str, Any]]:
        self.get_keyframes_by_ids_calls.append(list(keyframe_ids))
        return {kid: self._docs_by_id[kid] for kid in keyframe_ids if kid in self._docs_by_id}

    async def search_videos_by_title(self, query_text: str, top_k: int) -> list[dict[str, Any]]:
        self.search_videos_by_title_calls.append((query_text, top_k))
        return self._videos_by_title[:top_k]

    async def get_keyframes_for_video(self, video_id: str) -> list[dict[str, Any]]:
        self.get_keyframes_for_video_calls.append(video_id)
        return self._keyframes_by_video.get(video_id, [])


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
    vector_rows: dict[str, dict] | None = None,
    ocr_texts_by_id: dict[str, str] | None = None,
    bm25_scores_by_query: dict[str, dict[str, float]] | None = None,
    text_vectors_by_query: dict[str, np.ndarray] | None = None,
    model_id: str = "fake-encoder",
    milvus_hits_by_call: list[list[dict[str, Any]]] | None = None,
    title_video_hits: list[dict[str, Any]] | None = None,
    keyframes_by_video: dict[str, list[dict[str, Any]]] | None = None,
) -> tuple[SearchService, _FakeEncoder, _FakeMilvusStore, _FakeEsStore, _FakeMongoStore]:
    default_vector = np.zeros((1, 4), dtype=np.float32)
    encoder = _FakeEncoder(
        query_vector if query_vector is not None else default_vector,
        text_vectors_by_query=text_vectors_by_query,
        model_id=model_id,
    )
    milvus_store = _FakeMilvusStore(
        milvus_hits or [], vector_rows=vector_rows, hits_by_call=milvus_hits_by_call
    )
    es_store = _FakeEsStore(
        es_hits or [], texts_by_id=ocr_texts_by_id, scores_by_query=bm25_scores_by_query
    )
    mongo_store = _FakeMongoStore(
        mongo_docs or {}, videos_by_title=title_video_hits, keyframes_by_video=keyframes_by_video
    )
    service = SearchService(
        encoder=encoder,
        milvus_store=milvus_store,
        es_store=es_store,
        mongo_store=mongo_store,
        query_translator=query_translator,
        kis_config=KisConfig(candidate_multiplier=2, candidate_min=4),
    )
    return service, encoder, milvus_store, es_store, mongo_store


def _fake_abe_ranked(
    query_index: int,
    candidate_ids: list[str],
    percentiles_by_id: dict[str, float],
) -> dict[int, AbeRankedQuery]:
    components_by_id = {
        keyframe_id: AbeQueryComponents(
            pairs=[
                AbePairScore(
                    attribute="red",
                    object="packet",
                    positive_attributes=["red"],
                    negative_attributes=["white"],
                )
            ],
            base_local=0.1,
            refined_local=0.3,
            binding_delta=0.2,
            local_score=0.5,
            global_abe=0.6,
            raw_score=percentiles_by_id[keyframe_id],
            patch_heatmap=None,
        )
        for keyframe_id in candidate_ids
    }
    rankings = {
        keyframe_id: RankedScore(
            raw_score=percentile,
            rank=rank,
            rank_percentile=percentile,
        )
        for rank, (keyframe_id, percentile) in enumerate(percentiles_by_id.items(), start=1)
    }
    return {
        query_index: AbeRankedQuery(
            components_by_id=components_by_id,
            rankings=rankings,
        )
    }


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


def test_unknown_mode_error_keeps_the_original_error_code_and_status():
    # plan §P1.8: SearchServiceError grew error_code/status_code so
    # app/api/search.py can distinguish "caller's mistake" (this case) from
    # "dependency is down" (test_ocr_mode_raises_clear_error_when...
    # below) -- the default must stay exactly what it was before that change.
    error = SearchServiceError("Unknown search mode: 'video'")

    assert error.error_code == "invalid_search_mode"
    assert error.status_code == 400


async def test_ocr_mode_raises_clear_error_when_elasticsearch_unavailable():
    # plan §P1.8, PRD T5: an ES-down OCR request must surface as a readable
    # degraded-mode error (CLAUDE.md §7), not an unhandled exception that
    # falls through to the generic "Internal server error" 500 handler.
    encoder = _FakeEncoder(np.zeros((1, 4), dtype=np.float32))
    milvus_store = _FakeMilvusStore([])
    es_store = _FakeEsStoreDown()
    mongo_store = _FakeMongoStore({})
    service = SearchService(
        encoder=encoder, milvus_store=milvus_store, es_store=es_store, mongo_store=mongo_store
    )

    with pytest.raises(SearchServiceError) as exc_info:
        await service.search("x", mode="ocr", top_k=5)

    assert exc_info.value.error_code == "ocr_unavailable"
    assert exc_info.value.status_code == 503


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


async def test_kis_stage1_text_only_retrieves_configured_text_candidates():
    vector = np.array([[1.0, 0.0]], dtype=np.float32)
    milvus_hits = [
        {
            "keyframe_id": "A",
            "distance": 0.9,
            "entity": {"video_id": "V1", "frame_idx": 10, "timestamp": 0.4},
        },
        {
            "keyframe_id": "B",
            "distance": 0.8,
            "entity": {"video_id": "V1", "frame_idx": 20, "timestamp": 0.8},
        },
    ]
    mongo_docs = {
        "A": {"video_id": "V1", "frame_idx": 10, "timestamp": 0.4, "thumbnail_path": "a.jpg"},
        "B": {"video_id": "V1", "frame_idx": 20, "timestamp": 0.8, "thumbnail_path": "b.jpg"},
    }
    service, encoder, milvus_store, es_store, _ = _service(
        milvus_hits=milvus_hits,
        mongo_docs=mongo_docs,
        query_vector=vector,
    )

    pool = await service.retrieve_kis_candidates(
        text_query="red shirt",
        text_top_k=2,
        ocr_query=None,
        ocr_top_k=5,
    )

    assert encoder.encode_text_calls == [["red shirt"]]
    assert milvus_store.search_calls[0][1] == 2
    assert es_store.search_calls == []
    assert pool.candidate_ids == ["A", "B"]
    assert pool.candidate_pool_size == 2
    assert pool.text_candidate_count == 2
    assert pool.ocr_candidate_count == 0
    assert pool.results[0].final_score is None
    assert pool.results[0].stage1.text.raw_cosine == pytest.approx(0.9)
    assert pool.results[0].stage1.text.rank == 1


async def test_kis_stage1_text_rejects_beit3_encoder_before_query_encoding():
    service, encoder, milvus_store, _, _ = _service(
        query_vector=np.array([[1.0, 0.0]], dtype=np.float32),
        model_id="beit3_base_itc_patch16_224",
    )

    with pytest.raises(SearchServiceError, match="FG-CLIP"):
        await service.retrieve_kis_candidates(
            text_query="red shirt",
            text_top_k=2,
            ocr_query=None,
            ocr_top_k=5,
        )

    assert encoder.encode_text_calls == []
    assert milvus_store.search_calls == []


async def test_kis_stage1_text_and_ocr_use_independent_k_and_deduplicate_union():
    vector = np.array([[1.0, 0.0]], dtype=np.float32)
    milvus_hits = [
        {
            "keyframe_id": "A",
            "distance": 0.9,
            "entity": {"video_id": "V1", "frame_idx": 10, "timestamp": 0.4},
        },
        {
            "keyframe_id": "B",
            "distance": 0.8,
            "entity": {"video_id": "V1", "frame_idx": 20, "timestamp": 0.8},
        },
    ]
    es_hits = [
        {
            "keyframe_id": "B",
            "ocr_text": "VINAMILK",
            "video_id": "V1",
            "timestamp": 0.8,
            "score": 10.0,
        },
        {
            "keyframe_id": "C",
            "ocr_text": "VINAMILK",
            "video_id": "V1",
            "timestamp": 1.2,
            "score": 9.0,
        },
        {
            "keyframe_id": "D",
            "ocr_text": "VINAMILK",
            "video_id": "V1",
            "timestamp": 1.6,
            "score": 8.0,
        },
    ]
    mongo_docs = {
        keyframe_id: {
            "video_id": "V1",
            "frame_idx": index * 10,
            "timestamp": index * 0.4,
            "thumbnail_path": f"{keyframe_id}.jpg",
        }
        for index, keyframe_id in enumerate(["A", "B", "C", "D"], start=1)
    }
    service, _, milvus_store, es_store, _ = _service(
        milvus_hits=milvus_hits,
        es_hits=es_hits,
        mongo_docs=mongo_docs,
        query_vector=vector,
        ocr_texts_by_id={"A": "", "B": "VINAMILK", "C": "VINAMILK"},
        bm25_scores_by_query={"VINAMILK": {"B": 10.0, "C": 9.0}},
    )

    pool = await service.retrieve_kis_candidates(
        text_query="boat",
        text_top_k=2,
        ocr_query="VINAMILK",
        ocr_top_k=2,
    )

    assert milvus_store.search_calls[0][1] == 2
    assert es_store.search_calls == [("VINAMILK", 4)]
    assert pool.candidate_ids == ["A", "B", "C"]
    assert pool.candidate_pool_size == 3
    assert pool.text_candidate_count == 2
    assert pool.ocr_candidate_count == 2
    assert [hit.keyframe_id for hit in pool.results] == ["A", "B", "C"]
    assert pool.results[0].score == pytest.approx(0.9)
    assert pool.results[1].score == pytest.approx(0.8)
    assert pool.results[1].stage1.source_modalities == ["text", "ocr"]
    assert pool.results[1].stage1.ocr.query_score == pytest.approx(1.0)
    assert pool.results[0].stage1.source_modalities == ["text"]
    assert pool.results[0].stage1.ocr.query_score == pytest.approx(0.0)


async def test_kis_stage1_multi_text_queries_union_deduplicate_and_fuse_true_cosines():
    hits_q1 = [
        {
            "keyframe_id": "A",
            "distance": 1.0,
            "entity": {"video_id": "VA", "frame_idx": 10, "timestamp": 0.4},
        },
        {
            "keyframe_id": "B",
            "distance": 0.9,
            "entity": {"video_id": "VB", "frame_idx": 20, "timestamp": 0.8},
        },
        {
            "keyframe_id": "C",
            "distance": 0.2,
            "entity": {"video_id": "VC", "frame_idx": 30, "timestamp": 1.2},
        },
    ]
    hits_q2 = [
        {
            "keyframe_id": "B",
            "distance": 0.1,
            "entity": {"video_id": "VB", "frame_idx": 20, "timestamp": 0.8},
        },
        {
            "keyframe_id": "C",
            "distance": 0.8,
            "entity": {"video_id": "VC", "frame_idx": 30, "timestamp": 1.2},
        },
        {
            "keyframe_id": "D",
            "distance": 1.0,
            "entity": {"video_id": "VD", "frame_idx": 40, "timestamp": 1.6},
        },
    ]
    vector_rows = {
        "A": {
            "keyframe_id": "A",
            "embedding": np.array([1.0, 0.0], dtype=np.float32),
            "video_id": "VA",
            "frame_idx": 10,
            "timestamp": 0.4,
        },
        "B": {
            "keyframe_id": "B",
            "embedding": np.array([1.0, 1.0], dtype=np.float32),
            "video_id": "VB",
            "frame_idx": 20,
            "timestamp": 0.8,
        },
        "C": {
            "keyframe_id": "C",
            "embedding": np.array([0.0, 1.0], dtype=np.float32),
            "video_id": "VC",
            "frame_idx": 30,
            "timestamp": 1.2,
        },
        "D": {
            "keyframe_id": "D",
            "embedding": np.array([0.5, 1.0], dtype=np.float32),
            "video_id": "VD",
            "frame_idx": 40,
            "timestamp": 1.6,
        },
    }
    mongo_docs = {
        keyframe_id: {
            "video_id": row["video_id"],
            "frame_idx": row["frame_idx"],
            "timestamp": row["timestamp"],
            "thumbnail_path": f"{keyframe_id}.jpg",
        }
        for keyframe_id, row in vector_rows.items()
    }
    service, encoder, milvus_store, es_store, _ = _service(
        mongo_docs=mongo_docs,
        vector_rows=vector_rows,
        milvus_hits_by_call=[hits_q1, hits_q2],
        text_vectors_by_query={
            "horizontal": np.array([1.0, 0.0], dtype=np.float32),
            "vertical": np.array([0.0, 1.0], dtype=np.float32),
        },
    )

    pool = await service.retrieve_kis_candidates(
        text_query=None,
        text_queries=["horizontal", "vertical"],
        text_top_k=3,
        ocr_query=None,
        ocr_top_k=5,
    )

    assert encoder.encode_text_calls == [["horizontal", "vertical"]]
    assert [call[1] for call in milvus_store.search_calls] == [3, 3]
    assert es_store.search_calls == []
    assert pool.candidate_ids == ["A", "B", "C", "D"]
    assert pool.text_candidate_count == 4
    assert len({(hit.video_id, hit.frame_idx) for hit in pool.results}) == 4
    by_id = {hit.keyframe_id: hit for hit in pool.results}
    assert by_id["A"].stage1.text_queries[0].original_rank == 1
    assert by_id["A"].stage1.text_queries[1].original_rank is None
    assert by_id["A"].stage1.text_queries[1].raw_cosine == pytest.approx(0.0)
    assert by_id["B"].score == pytest.approx(2 * (0.5**0.5))
    assert [hit.keyframe_id for hit in pool.results][:2] == ["B", "D"]


async def test_kis_stage1_multi_text_deduplicates_identical_video_frame_identity():
    hits_q1 = [
        {
            "keyframe_id": "A1",
            "distance": 0.9,
            "entity": {"video_id": "V1", "frame_idx": 10, "timestamp": 0.4},
        },
        {
            "keyframe_id": "A2",
            "distance": 0.8,
            "entity": {"video_id": "V1", "frame_idx": 10, "timestamp": 0.4},
        },
        {
            "keyframe_id": "B",
            "distance": 0.7,
            "entity": {"video_id": "V1", "frame_idx": 20, "timestamp": 0.8},
        },
    ]
    mongo_docs = {
        "A1": {"video_id": "V1", "frame_idx": 10, "timestamp": 0.4, "thumbnail_path": "a.jpg"},
        "B": {"video_id": "V1", "frame_idx": 20, "timestamp": 0.8, "thumbnail_path": "b.jpg"},
    }
    service, *_ = _service(
        mongo_docs=mongo_docs,
        milvus_hits_by_call=[hits_q1],
        text_vectors_by_query={"query": np.array([1.0, 0.0], dtype=np.float32)},
    )

    pool = await service.retrieve_kis_candidates(
        text_query=None,
        text_queries=["query"],
        text_top_k=3,
        ocr_query=None,
        ocr_top_k=5,
    )

    assert pool.candidate_ids == ["A1", "B"]
    assert [(hit.video_id, hit.frame_idx) for hit in pool.results] == [("V1", 10), ("V1", 20)]


async def test_kis_stage1_title_query_uses_video_title_metadata_without_ocr_or_milvus():
    keyframes_by_video = {
        "V1": [
            {
                "keyframe_id": "A",
                "video_id": "V1",
                "frame_idx": 10,
                "timestamp": 0.4,
                "thumbnail_path": "a.jpg",
            }
        ]
    }
    mongo_docs = {
        "A": {"video_id": "V1", "frame_idx": 10, "timestamp": 0.4, "thumbnail_path": "a.jpg"}
    }
    service, _, milvus_store, es_store, mongo_store = _service(
        mongo_docs=mongo_docs,
        title_video_hits=[{"video_id": "V1", "title": "Chicken soup recipe", "score": 2.0}],
        keyframes_by_video=keyframes_by_video,
    )

    pool = await service.retrieve_kis_candidates(
        text_query=None,
        text_top_k=5,
        ocr_query=None,
        ocr_top_k=5,
        title_query="chicken",
        title_top_k=1,
    )

    assert milvus_store.search_calls == []
    assert es_store.search_calls == []
    assert mongo_store.search_videos_by_title_calls == [("chicken", 1)]
    assert pool.title_candidate_count == 1
    assert pool.results[0].stage1.source_modalities == ["title"]
    assert pool.results[0].stage1.title.video_title == "Chicken soup recipe"


async def test_kis_stage1_allows_ocr_only_candidate_retrieval():
    es_hits = [
        {
            "keyframe_id": "A",
            "ocr_text": "SAIGON",
            "video_id": "V1",
            "timestamp": 0.4,
            "score": 7.5,
        }
    ]
    mongo_docs = {
        "A": {"video_id": "V1", "frame_idx": 10, "timestamp": 0.4, "thumbnail_path": "a.jpg"}
    }
    service, _, milvus_store, es_store, _ = _service(
        es_hits=es_hits,
        mongo_docs=mongo_docs,
        ocr_texts_by_id={"A": "SAIGON"},
        bm25_scores_by_query={"SAIGON": {"A": 7.5}},
    )

    pool = await service.retrieve_kis_candidates(
        text_query=None,
        text_top_k=5,
        ocr_query="SAIGON",
        ocr_top_k=1,
    )

    assert milvus_store.search_calls == []
    assert es_store.search_calls == [("SAIGON", 4)]
    assert pool.candidate_ids == ["A"]
    assert pool.results[0].stage1.text is None
    assert pool.results[0].stage1.ocr.query_score == pytest.approx(1.0)


async def test_kis_stage2_text_subquery_reranks_candidate_pool_without_global_search_or_q0():
    vector_rows = {
        "A": {
            "keyframe_id": "A",
            "embedding": np.array([1.0, 0.0], dtype=np.float32),
            "video_id": "V1",
            "frame_idx": 10,
            "timestamp": 0.4,
        },
        "B": {
            "keyframe_id": "B",
            "embedding": np.array([0.0, 1.0], dtype=np.float32),
            "video_id": "V1",
            "frame_idx": 20,
            "timestamp": 0.8,
        },
    }
    mongo_docs = {
        "A": {"video_id": "V1", "frame_idx": 10, "timestamp": 0.4, "thumbnail_path": "a.jpg"},
        "B": {"video_id": "V1", "frame_idx": 20, "timestamp": 0.8, "thumbnail_path": "b.jpg"},
    }
    service, encoder, milvus_store, _, _ = _service(
        vector_rows=vector_rows,
        mongo_docs=mongo_docs,
        text_vectors_by_query={
            "global q0": np.array([1.0, 0.0], dtype=np.float32),
            "vertical": np.array([0.0, 1.0], dtype=np.float32),
        },
    )

    results = await service.rerank_kis_candidates(
        candidate_ids=["A", "B"],
        queries=[KisQuery(query_index=0, type=QueryType.TEXT, text="vertical")],
        final_top_k=2,
    )

    assert encoder.encode_text_calls == [["vertical"]]
    assert milvus_store.search_calls == []
    assert milvus_store.get_vectors_by_ids_calls == [["A", "B"]]
    assert [hit.keyframe_id for hit in results] == ["B", "A"]
    assert results[0].final_score == pytest.approx(1.0)
    assert results[0].modalities["text"].queries[0].query == "vertical"


async def test_kis_stage2_multiple_text_queries_use_text_geometric_mean():
    vector_rows = {
        "A": {
            "keyframe_id": "A",
            "embedding": np.array([1.0, 0.0], dtype=np.float32),
            "video_id": "V1",
            "frame_idx": 10,
            "timestamp": 0.4,
        },
        "B": {
            "keyframe_id": "B",
            "embedding": np.array([0.0, 1.0], dtype=np.float32),
            "video_id": "V1",
            "frame_idx": 20,
            "timestamp": 0.8,
        },
        "C": {
            "keyframe_id": "C",
            "embedding": np.array([1.0, 1.0], dtype=np.float32),
            "video_id": "V1",
            "frame_idx": 30,
            "timestamp": 1.2,
        },
    }
    mongo_docs = {
        keyframe_id: {
            "video_id": "V1",
            "frame_idx": index * 10,
            "timestamp": index * 0.4,
            "thumbnail_path": f"{keyframe_id}.jpg",
        }
        for index, keyframe_id in enumerate(["A", "B", "C"], start=1)
    }
    service, _, _, _, _ = _service(
        vector_rows=vector_rows,
        mongo_docs=mongo_docs,
        text_vectors_by_query={
            "horizontal": np.array([1.0, 0.0], dtype=np.float32),
            "vertical": np.array([0.0, 1.0], dtype=np.float32),
        },
    )

    results = await service.rerank_kis_candidates(
        candidate_ids=["A", "B", "C"],
        queries=[
            KisQuery(query_index=0, type=QueryType.TEXT, text="horizontal"),
            KisQuery(query_index=1, type=QueryType.TEXT, text="vertical"),
        ],
        final_top_k=3,
    )

    assert [hit.keyframe_id for hit in results] == ["C", "A", "B"]
    assert results[0].modalities["text"].score == pytest.approx(0.5)
    assert [query.rank for query in results[0].modalities["text"].queries] == [2, 2]
    assert [query.rank_percentile for query in results[0].modalities["text"].queries] == [
        pytest.approx(0.5),
        pytest.approx(0.5),
    ]


async def test_kis_stage2_multiple_ocr_queries_use_ocr_geometric_mean_without_global_search():
    mongo_docs = {
        "A": {"video_id": "V1", "frame_idx": 10, "timestamp": 0.4, "thumbnail_path": "a.jpg"},
        "B": {"video_id": "V1", "frame_idx": 20, "timestamp": 0.8, "thumbnail_path": "b.jpg"},
        "C": {"video_id": "V1", "frame_idx": 30, "timestamp": 1.2, "thumbnail_path": "c.jpg"},
    }
    service, _, _, es_store, _ = _service(
        mongo_docs=mongo_docs,
        ocr_texts_by_id={"A": "AAA", "B": "BBB", "C": "AAA BBB"},
        bm25_scores_by_query={
            "AAA": {"A": 10.0, "C": 7.0},
            "BBB": {"B": 10.0, "C": 7.0},
        },
    )

    results = await service.rerank_kis_candidates(
        candidate_ids=["A", "B", "C"],
        queries=[
            KisQuery(query_index=0, type=QueryType.OCR, text="AAA"),
            KisQuery(query_index=1, type=QueryType.OCR, text="BBB"),
        ],
        final_top_k=3,
    )

    assert es_store.search_calls == []
    assert es_store.score_by_ids_calls == [("AAA", ["A", "B", "C"]), ("BBB", ["A", "B", "C"])]
    assert [hit.keyframe_id for hit in results] == ["C", "A", "B"]
    assert results[0].modalities["ocr"].score == pytest.approx(0.5)


async def test_kis_stage2_text_and_ocr_fuses_active_modalities_with_final_geomean():
    vector_rows = {
        "semantic_only": {
            "keyframe_id": "semantic_only",
            "embedding": np.array([1.0, 0.0], dtype=np.float32),
            "video_id": "V1",
            "frame_idx": 10,
            "timestamp": 0.4,
        },
        "balanced": {
            "keyframe_id": "balanced",
            "embedding": np.array([1.0, 1.0], dtype=np.float32),
            "video_id": "V1",
            "frame_idx": 20,
            "timestamp": 0.8,
        },
        "ocr_only": {
            "keyframe_id": "ocr_only",
            "embedding": np.array([0.0, 1.0], dtype=np.float32),
            "video_id": "V1",
            "frame_idx": 30,
            "timestamp": 1.2,
        },
    }
    mongo_docs = {
        keyframe_id: {
            "video_id": "V1",
            "frame_idx": index * 10,
            "timestamp": index * 0.4,
            "thumbnail_path": f"{keyframe_id}.jpg",
        }
        for index, keyframe_id in enumerate(["semantic_only", "balanced", "ocr_only"], start=1)
    }
    service, _, milvus_store, es_store, _ = _service(
        mongo_docs=mongo_docs,
        vector_rows=vector_rows,
        ocr_texts_by_id={"semantic_only": "", "balanced": "VINAMILK", "ocr_only": "VINAMILK"},
        bm25_scores_by_query={"VINAMILK": {"ocr_only": 10.0, "balanced": 8.0}},
        text_vectors_by_query={"red shirt": np.array([1.0, 0.0], dtype=np.float32)},
    )

    results = await service.rerank_kis_candidates(
        candidate_ids=["semantic_only", "balanced", "ocr_only"],
        queries=[
            KisQuery(query_index=0, type=QueryType.TEXT, text="red shirt"),
            KisQuery(query_index=1, type=QueryType.OCR, text="VINAMILK"),
        ],
        final_top_k=3,
    )

    assert milvus_store.get_vectors_by_ids_calls == [["semantic_only", "balanced", "ocr_only"]]
    assert es_store.score_by_ids_calls == [("VINAMILK", ["semantic_only", "balanced", "ocr_only"])]
    assert [hit.keyframe_id for hit in results] == ["balanced", "semantic_only", "ocr_only"]
    assert results[0].final_score == pytest.approx(0.5)
    assert results[0].active_modalities == 2
    assert 0 <= results[0].final_score <= 1
    assert results[0].modalities["text"].queries[0].rank_percentile == pytest.approx(0.5)
    assert results[0].modalities["ocr"].queries[0].query_score == pytest.approx(0.8)


async def test_kis_stage2_text_and_abe_fuses_active_modalities_with_final_geomean(
    monkeypatch: pytest.MonkeyPatch,
):
    vector_rows = {
        "A": {
            "keyframe_id": "A",
            "embedding": np.array([1.0, 0.0], dtype=np.float32),
            "video_id": "V1",
            "frame_idx": 10,
            "timestamp": 0.4,
        },
        "B": {
            "keyframe_id": "B",
            "embedding": np.array([0.0, 1.0], dtype=np.float32),
            "video_id": "V1",
            "frame_idx": 20,
            "timestamp": 0.8,
        },
    }
    mongo_docs = {
        "A": {"video_id": "V1", "frame_idx": 10, "timestamp": 0.4, "thumbnail_path": "a.jpg"},
        "B": {"video_id": "V1", "frame_idx": 20, "timestamp": 0.8, "thumbnail_path": "b.jpg"},
    }
    service, _, _, _, _ = _service(
        mongo_docs=mongo_docs,
        vector_rows=vector_rows,
        text_vectors_by_query={"horizontal": np.array([1.0, 0.0], dtype=np.float32)},
    )
    monkeypatch.setattr(
        service,
        "_rank_abe_queries",
        lambda candidate_pool_ids, abe_queries, mongo_docs, vector_rows_by_id: _fake_abe_ranked(
            1, list(candidate_pool_ids), {"A": 0.25, "B": 1.0}
        ),
    )

    results = await service.rerank_kis_candidates(
        candidate_ids=["A", "B"],
        queries=[
            KisQuery(query_index=0, type=QueryType.TEXT, text="horizontal"),
            KisQuery(
                query_index=1,
                type=QueryType.ABE,
                pairs=(AbePair(attribute="red", object="packet"),),
            ),
        ],
        final_top_k=2,
    )

    assert results[0].active_modalities == 2
    assert results[0].modalities.keys() == {"text", "abe"}
    assert results[0].modalities["abe"].queries[0].rank_percentile == pytest.approx(0.25)
    assert results[0].final_score == pytest.approx((1.0 * 0.25) ** 0.5)


async def test_kis_stage2_ocr_and_abe_fuses_active_modalities_with_final_geomean(
    monkeypatch: pytest.MonkeyPatch,
):
    mongo_docs = {
        "A": {"video_id": "V1", "frame_idx": 10, "timestamp": 0.4, "thumbnail_path": "a.jpg"},
        "B": {"video_id": "V1", "frame_idx": 20, "timestamp": 0.8, "thumbnail_path": "b.jpg"},
    }
    service, _, _, _, _ = _service(
        mongo_docs=mongo_docs,
        ocr_texts_by_id={"A": "VINAMILK", "B": ""},
        bm25_scores_by_query={"VINAMILK": {"A": 10.0}},
    )
    monkeypatch.setattr(
        service,
        "_rank_abe_queries",
        lambda candidate_pool_ids, abe_queries, mongo_docs, vector_rows_by_id: _fake_abe_ranked(
            1, list(candidate_pool_ids), {"A": 0.25, "B": 1.0}
        ),
    )

    results = await service.rerank_kis_candidates(
        candidate_ids=["A", "B"],
        queries=[
            KisQuery(query_index=0, type=QueryType.OCR, text="VINAMILK"),
            KisQuery(
                query_index=1,
                type=QueryType.ABE,
                pairs=(AbePair(attribute="red", object="packet"),),
            ),
        ],
        final_top_k=2,
    )

    assert results[0].active_modalities == 2
    assert results[0].modalities.keys() == {"abe", "ocr"}
    assert results[0].final_score == pytest.approx((1.0 * 0.25) ** 0.5)


async def test_kis_stage2_text_ocr_and_abe_fuses_only_active_stage2_modalities(
    monkeypatch: pytest.MonkeyPatch,
):
    vector_rows = {
        "A": {
            "keyframe_id": "A",
            "embedding": np.array([1.0, 0.0], dtype=np.float32),
            "video_id": "V1",
            "frame_idx": 10,
            "timestamp": 0.4,
        },
        "B": {
            "keyframe_id": "B",
            "embedding": np.array([0.0, 1.0], dtype=np.float32),
            "video_id": "V1",
            "frame_idx": 20,
            "timestamp": 0.8,
        },
    }
    mongo_docs = {
        "A": {"video_id": "V1", "frame_idx": 10, "timestamp": 0.4, "thumbnail_path": "a.jpg"},
        "B": {"video_id": "V1", "frame_idx": 20, "timestamp": 0.8, "thumbnail_path": "b.jpg"},
    }
    service, encoder, _, _, _ = _service(
        mongo_docs=mongo_docs,
        vector_rows=vector_rows,
        ocr_texts_by_id={"A": "VINAMILK", "B": ""},
        bm25_scores_by_query={"VINAMILK": {"A": 10.0}},
        text_vectors_by_query={
            "global q0 must never be encoded": np.array([0.0, 1.0], dtype=np.float32),
            "horizontal": np.array([1.0, 0.0], dtype=np.float32),
        },
    )
    monkeypatch.setattr(
        service,
        "_rank_abe_queries",
        lambda candidate_pool_ids, abe_queries, mongo_docs, vector_rows_by_id: _fake_abe_ranked(
            2, list(candidate_pool_ids), {"A": 0.25, "B": 1.0}
        ),
    )

    results = await service.rerank_kis_candidates(
        candidate_ids=["A", "B"],
        queries=[
            KisQuery(query_index=0, type=QueryType.TEXT, text="horizontal"),
            KisQuery(query_index=1, type=QueryType.OCR, text="VINAMILK"),
            KisQuery(
                query_index=2,
                type=QueryType.ABE,
                pairs=(AbePair(attribute="red", object="packet"),),
            ),
        ],
        final_top_k=2,
    )

    assert encoder.encode_text_calls == [["horizontal"]]
    assert results[0].active_modalities == 3
    assert results[0].modalities.keys() == {"text", "abe", "ocr"}
    assert results[0].final_score == pytest.approx((1.0 * 1.0 * 0.25) ** (1 / 3))


async def test_kis_stage2_rejects_final_top_k_larger_than_candidate_pool():
    service, *_ = _service()

    with pytest.raises(SearchServiceError) as exc_info:
        await service.rerank_kis_candidates(
            candidate_ids=["A"],
            queries=[KisQuery(query_index=0, type=QueryType.TEXT, text="x")],
            final_top_k=2,
        )

    assert exc_info.value.error_code == "invalid_kis_request"


async def test_kis_stage2_rejects_empty_subqueries_without_fake_final_score():
    service, *_ = _service()

    with pytest.raises(SearchServiceError) as exc_info:
        await service.rerank_kis_candidates(candidate_ids=["A"], queries=[], final_top_k=1)

    assert "Stage 2 requires at least one subquery" in str(exc_info.value)


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
