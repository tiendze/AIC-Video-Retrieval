"""RED reproducer for `POST /api/search` (plan §P0.8) -- the router is thin
(validate + delegate, .claude/rules/ecc/python/fastapi.md): request schema
validates `mode`/`top_k`, the route builds a `SearchService` from injected
stores/encoder via `Depends`, and maps `SearchServiceError` onto the unified
`APIError` schema (CLAUDE.md §7).

A Fake `SearchService` is injected via `app.dependency_overrides` -- no
Docker/GPU needed; SearchService's own behavior is already covered by
tests/test_search_service.py.
"""

from __future__ import annotations

import io
import logging

import pytest
from fastapi import FastAPI, UploadFile
from fastapi.testclient import TestClient
from PIL import Image
from starlette.datastructures import Headers

from app.api import search as search_api
from app.api.errors import APIError, register_error_handling
from app.dependencies import get_search_service
from app.schemas.request.search import MAX_UPLOAD_SIZE_BYTES, KisCandidateRequest
from app.services.asr_retrieval_service import (
    AsrEvidenceSegment,
    AsrGateResult,
    AsrQueryMatch,
    AsrVideoEvidence,
)
from app.services.kis_scoring import (
    AbeModalityScore,
    AbePairScore,
    AbeQueryScore,
    OcrModalityScore,
    OcrQueryScore,
    Stage1RetrievalScore,
    Stage1TextScore,
)
from app.services.search_service import KisCandidatePool, SearchHit, SearchServiceError


def _png_bytes(size: tuple[int, int] = (8, 8), color: tuple[int, int, int] = (10, 20, 30)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color=color).save(buf, format="PNG")
    return buf.getvalue()


def _bmp_bytes(size: tuple[int, int] = (8, 8), color: tuple[int, int, int] = (10, 20, 30)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color=color).save(buf, format="BMP")
    return buf.getvalue()


def _truncated_jpeg_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color=(10, 20, 30)).save(buf, format="JPEG")
    full = buf.getvalue()
    return full[: len(full) // 2]  # header identifies as JPEG; body decode fails with OSError


def _decompression_bomb_png_bytes() -> bytes:
    # Solid color compresses to well under MAX_UPLOAD_SIZE_BYTES while the
    # declared pixel count (196M) exceeds Pillow's default 178956970 limit.
    buf = io.BytesIO()
    Image.new("RGB", (14000, 14000), color=(10, 20, 30)).save(buf, format="PNG")
    return buf.getvalue()


class _FakeSearchService:
    def __init__(
        self,
        hits: list[SearchHit] | None = None,
        candidate_pool: KisCandidatePool | None = None,
        error: Exception | None = None,
    ) -> None:
        self._hits = hits or []
        self._candidate_pool = candidate_pool or KisCandidatePool(
            candidate_ids=[],
            text_candidate_count=0,
            ocr_candidate_count=0,
            results=[],
        )
        self._error = error
        self.search_calls: list[tuple[str, str, int]] = []
        self.retrieve_kis_candidates_calls: list[
            tuple[
                str | None,
                list[str],
                int,
                str | None,
                int,
                str | None,
                int,
                str | None,
                list[str],
                str,
                int,
            ]
        ] = []
        self.rerank_kis_candidates_calls: list[
            tuple[list[str], list[tuple[int, str, str]], int]
        ] = []
        self.rerank_kis_query_objects = []
        self.search_by_image_calls: list[int] = []

    async def search(self, query: str, mode: str, top_k: int) -> list[SearchHit]:
        self.search_calls.append((query, mode, top_k))
        if self._error is not None:
            raise self._error
        return self._hits

    async def retrieve_kis_candidates(
        self,
        text_query: str | None,
        text_top_k: int,
        ocr_query: str | None,
        ocr_top_k: int,
        text_queries=None,
        title_query: str | None = None,
        title_top_k: int = 0,
        asr_query: str | None = None,
        asr_queries=None,
        asr_match_mode: str = "any",
        asr_video_top_k: int = 20,
    ) -> KisCandidatePool:
        self.retrieve_kis_candidates_calls.append(
            (
                text_query,
                list(text_queries or []),
                text_top_k,
                ocr_query,
                ocr_top_k,
                title_query,
                title_top_k,
                asr_query,
                list(asr_queries or []),
                asr_match_mode,
                asr_video_top_k,
            )
        )
        if self._error is not None:
            raise self._error
        return self._candidate_pool

    async def rerank_kis_candidates(
        self, candidate_ids, queries, final_top_k: int
    ) -> list[SearchHit]:
        self.rerank_kis_candidates_calls.append(
            (
                list(candidate_ids),
                [(query.query_index, query.type.value, query.text) for query in queries],
                final_top_k,
            )
        )
        self.rerank_kis_query_objects.append(list(queries))
        if self._error is not None:
            raise self._error
        return self._hits

    async def search_by_image(self, image, top_k: int) -> list[SearchHit]:
        self.search_by_image_calls.append(top_k)
        if self._error is not None:
            raise self._error
        return self._hits


def _client(fake_service: _FakeSearchService) -> TestClient:
    app = FastAPI()
    register_error_handling(app)
    search_api.register_upload_guard(app)
    app.include_router(search_api.router)
    app.dependency_overrides[get_search_service] = lambda: fake_service
    return TestClient(app, raise_server_exceptions=False)


_ONE_HIT = [
    SearchHit(
        keyframe_id="L21_V001_f10",
        video_id="L21_V001",
        score=0.9,
        frame_idx=10,
        timestamp=0.4,
        thumbnail_path="t.jpg",
    )
]

_ONE_KIS_HIT = [
    SearchHit(
        keyframe_id="L21_V001_f10",
        video_id="L21_V001",
        score=1.0,
        frame_idx=10,
        timestamp=0.4,
        thumbnail_path="t.jpg",
        final_score=1.0,
        active_modalities=1,
        modalities={
            "ocr": OcrModalityScore(
                geometric_mean=1.0,
                score=1.0,
                queries=[
                    OcrQueryScore(
                        query_index=0,
                        query="VINAMILK",
                        raw_bm25=7.5,
                        bm25_reference=7.5,
                        bm25_normalized=1.0,
                        coverage=1.0,
                        query_score=1.0,
                        rank=1,
                        rank_percentile=1.0,
                    )
                ],
            )
        },
    )
]

_ONE_CANDIDATE_POOL = KisCandidatePool(
    candidate_ids=["L21_V001_f10"],
    text_candidate_count=1,
    ocr_candidate_count=0,
    results=[
        SearchHit(
            keyframe_id="L21_V001_f10",
            video_id="L21_V001",
            score=0.91,
            frame_idx=10,
            timestamp=0.4,
            thumbnail_path="t.jpg",
            stage1=Stage1RetrievalScore(
                source_modalities=["text"],
                text=Stage1TextScore(query="a man in a red shirt", raw_cosine=0.91, rank=1),
            ),
        )
    ],
)


def test_search_returns_200_with_mapped_hits():
    service = _FakeSearchService(hits=_ONE_HIT)
    client = _client(service)

    response = client.post("/api/search", json={"query": "thu tuong", "mode": "semantic"})

    assert response.status_code == 200
    assert response.json() == {
        "results": [
            {
                "keyframe_id": "L21_V001_f10",
                "video_id": "L21_V001",
                "score": 0.9,
                "frame_idx": 10,
                "timestamp": 0.4,
                "thumbnail_path": "t.jpg",
                "thumbnail_url": "/api/keyframes/L21_V001_f10/thumbnail",
                "image_url": "/api/keyframes/L21_V001_f10/image",
            }
        ]
    }


def test_search_uses_default_top_k_when_omitted():
    service = _FakeSearchService(hits=[])
    client = _client(service)

    client.post("/api/search", json={"query": "x", "mode": "ocr"})

    assert service.search_calls == [("x", "ocr", 20)]


def test_search_passes_through_an_explicit_top_k():
    service = _FakeSearchService(hits=[])
    client = _client(service)

    client.post("/api/search", json={"query": "x", "mode": "semantic", "top_k": 5})

    assert service.search_calls == [("x", "semantic", 5)]


def test_search_rejects_top_k_above_the_configured_max():
    service = _FakeSearchService(hits=[])
    client = _client(service)

    response = client.post("/api/search", json={"query": "x", "mode": "semantic", "top_k": 101})

    assert response.status_code == 422
    assert not service.search_calls


def test_search_rejects_an_unknown_mode_before_reaching_the_service():
    service = _FakeSearchService(hits=[])
    client = _client(service)

    response = client.post("/api/search", json={"query": "x", "mode": "video"})

    assert response.status_code == 422
    assert not service.search_calls


def test_search_service_error_maps_to_unified_400_error_schema():
    service = _FakeSearchService(error=SearchServiceError("Unknown search mode: 'x'"))
    client = _client(service)

    response = client.post("/api/search", json={"query": "x", "mode": "semantic"})

    assert response.status_code == 400
    body = response.json()
    assert body["error_code"] == "invalid_search_mode"
    assert set(body.keys()) == {"detail", "error_code", "request_id"}


def test_ocr_unavailable_error_maps_to_a_503_not_the_generic_400():
    # plan §P1.8: SearchServiceError's error_code/status_code must actually
    # reach the response, not get collapsed onto the old hardcoded
    # invalid_search_mode/400 pair.
    error = SearchServiceError(
        "OCR search is unavailable: Elasticsearch is unreachable",
        error_code="ocr_unavailable",
        status_code=503,
    )
    service = _FakeSearchService(error=error)
    client = _client(service)

    response = client.post("/api/search", json={"query": "x", "mode": "ocr"})

    assert response.status_code == 503
    assert response.json()["error_code"] == "ocr_unavailable"


def test_search_logs_mode_latency_and_num_results(caplog):
    # plan §P1.7, PRD §6.6: every search request must log mode/latency/result
    # count for operational latency measurement. No cache_hit field -- there
    # is no result cache for /api/search (only GET /api/keyframes/*
    # thumbnails cache; a different endpoint entirely), so a field that would
    # always read False belongs nowhere near this log line (chủ dự án
    # 2026-08-13: drop rather than fake it).
    service = _FakeSearchService(hits=_ONE_HIT)
    client = _client(service)

    with caplog.at_level(logging.INFO, logger="app.api.search"):
        response = client.post("/api/search", json={"query": "thu tuong", "mode": "semantic"})

    assert response.status_code == 200
    records = [r for r in caplog.records if r.name == "app.api.search"]
    assert len(records) == 1
    message = records[0].getMessage()
    assert "mode=semantic" in message
    assert "num_results=1" in message
    assert "latency_ms=" in message
    # cache_hit must never appear -- there is nothing behind it to measure.
    assert "cache_hit" not in message.lower()


def test_search_logs_zero_results_and_correct_mode_for_ocr(caplog):
    service = _FakeSearchService(hits=[])
    client = _client(service)

    with caplog.at_level(logging.INFO, logger="app.api.search"):
        client.post("/api/search", json={"query": "x", "mode": "ocr"})

    records = [r for r in caplog.records if r.name == "app.api.search"]
    assert len(records) == 1
    message = records[0].getMessage()
    assert "mode=ocr" in message
    assert "num_results=0" in message


def test_kis_candidate_retrieval_accepts_stage1_shape_and_uses_candidate_defaults():
    service = _FakeSearchService(candidate_pool=_ONE_CANDIDATE_POOL)
    client = _client(service)

    response = client.post(
        "/api/search/kis/candidates",
        json={"text_query": "a man in a red shirt"},
    )

    assert response.status_code == 200
    assert service.retrieve_kis_candidates_calls == [
        (
            "a man in a red shirt",
            ["a man in a red shirt"],
            300,
            None,
            100,
            None,
            100,
            None,
            [],
            "any",
            20,
        )
    ]


def test_kis_candidate_request_normalizes_blank_asr_and_uses_default_video_k():
    body = KisCandidateRequest(text_query="dam", asr_query="   ")

    assert body.asr_query is None
    assert body.asr_video_top_k == 20


def test_kis_candidate_request_rejects_overlong_asr_query():
    with pytest.raises(ValueError):
        KisCandidateRequest(text_query="dam", asr_query="x" * 1001)


def test_asr_alone_satisfies_stage1_candidate_retrieval_requirement():
    service = _FakeSearchService()
    response = _client(service).post(
        "/api/search/kis/candidates",
        json={"text_query": None, "ocr_query": None, "asr_query": "thủy lợi"},
    )

    assert response.status_code == 200
    assert service.retrieve_kis_candidates_calls == [
        (None, [], 300, None, 100, None, 100, "thủy lợi", ["thủy lợi"], "any", 20)
    ]


def test_kis_candidate_response_maps_top_level_asr_gate_once():
    segment = AsrEvidenceSegment(
        segment_id="L21_V001_seg10",
        text="công trình thủy lợi",
        start_time=151.2,
        end_time=158.4,
    )
    match = AsrQueryMatch(query="thủy lợi", detected=True, matches=[segment])
    asr_gate = AsrGateResult(
        queries=["thủy lợi"],
        match_mode="any",
        scanned_segment_count=1,
        asr_matches=[match],
        videos=[
            AsrVideoEvidence(
                video_id="L21_V001",
                detected=True,
                asr_matches=[match],
                segments=[segment],
            )
        ],
    )
    pool = KisCandidatePool(
        candidate_ids=_ONE_CANDIDATE_POOL.candidate_ids,
        text_candidate_count=1,
        ocr_candidate_count=0,
        results=_ONE_CANDIDATE_POOL.results,
        asr_gate=asr_gate,
    )

    response = _client(_FakeSearchService(candidate_pool=pool)).post(
        "/api/search/kis/candidates",
        json={"text_query": "dam", "asr_query": "thủy lợi", "asr_video_top_k": 20},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["asr_gate"]["candidate_video_count"] == 1
    assert body["asr_gate"]["asr_detected_any"] is True
    assert body["asr_gate"]["videos"][0]["segments"][0] == {
        "segment_id": "L21_V001_seg10",
        "text": "công trình thủy lợi",
        "start_time": 151.2,
        "end_time": 158.4,
    }
    assert "asr_gate" not in body["results"][0]


def test_active_asr_zero_hit_returns_200_with_empty_active_gate():
    empty_gate = AsrGateResult(
        queries=["không có kết quả"],
        match_mode="any",
        scanned_segment_count=0,
        asr_matches=[AsrQueryMatch(query="không có kết quả", detected=False, matches=[])],
        videos=[],
    )
    empty_pool = KisCandidatePool([], 0, 0, [], asr_gate=empty_gate)

    response = _client(_FakeSearchService(candidate_pool=empty_pool)).post(
        "/api/search/kis/candidates",
        json={"text_query": "dam", "asr_query": "không có kết quả"},
    )

    assert response.status_code == 200
    assert response.json()["asr_gate"]["videos"] == []
    assert response.json()["asr_gate"]["detected"] is False
    assert response.json()["candidate_ids"] == []


def test_asr_unavailable_error_maps_to_503():
    service = _FakeSearchService(
        error=SearchServiceError("ASR unavailable", "asr_unavailable", 503)
    )

    response = _client(service).post(
        "/api/search/kis/candidates",
        json={"text_query": "dam", "asr_query": "thủy lợi"},
    )

    assert response.status_code == 503
    assert response.json()["error_code"] == "asr_unavailable"


def test_kis_candidate_retrieval_returns_pool_metadata_and_stage1_breakdown():
    service = _FakeSearchService(candidate_pool=_ONE_CANDIDATE_POOL)
    client = _client(service)

    response = client.post(
        "/api/search/kis/candidates",
        json={"text_query": "a man in a red shirt", "text_top_k": 50},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["candidate_ids"] == ["L21_V001_f10"]
    assert body["candidate_pool_size"] == 1
    assert body["text_candidate_count"] == 1
    assert body["ocr_candidate_count"] == 0
    assert body["title_candidate_count"] == 0
    assert body["rrf_k"] == 60
    hit = body["results"][0]
    assert "final_score" not in hit
    assert hit["stage1"] == {
        "source_modalities": ["text"],
        "text": {"query": "a man in a red shirt", "raw_cosine": 0.91, "rank": 1},
    }


def test_kis_candidate_retrieval_accepts_multi_text_queries_and_title_query():
    service = _FakeSearchService(candidate_pool=_ONE_CANDIDATE_POOL)
    client = _client(service)

    response = client.post(
        "/api/search/kis/candidates",
        json={
            "text_queries": ["a dam", "reservoir"],
            "text_top_k": 50,
            "title_query": "hydroelectric",
            "title_top_k": 20,
        },
    )

    assert response.status_code == 200
    assert service.retrieve_kis_candidates_calls == [
        (
            "a dam",
            ["a dam", "reservoir"],
            50,
            None,
            100,
            "hydroelectric",
            20,
            None,
            [],
            "any",
            20,
        )
    ]


def test_kis_candidate_retrieval_rejects_blank_stage1_before_reaching_service():
    service = _FakeSearchService()
    client = _client(service)

    response = client.post(
        "/api/search/kis/candidates",
        json={"text_query": "  ", "ocr_query": ""},
    )

    assert response.status_code == 422
    assert service.retrieve_kis_candidates_calls == []


def test_kis_rerank_accepts_candidate_pool_subqueries_and_final_top_k():
    service = _FakeSearchService(hits=_ONE_KIS_HIT)
    client = _client(service)

    response = client.post(
        "/api/search/kis/rerank",
        json={
            "candidate_ids": ["A", "B"],
            "subqueries": [
                {"type": "text", "text": "night scene"},
                {"type": "ocr", "text": "VINAMILK"},
            ],
            "final_top_k": 2,
        },
    )

    assert response.status_code == 200
    assert service.rerank_kis_candidates_calls == [
        (
            ["A", "B"],
            [(0, "text", "night scene"), (1, "ocr", "VINAMILK")],
            2,
        )
    ]
    assert response.json()["candidate_pool_size"] == 2
    assert response.json()["final_top_k"] == 2


def test_kis_rerank_accepts_abe_pair_subqueries():
    service = _FakeSearchService(hits=_ONE_KIS_HIT)
    client = _client(service)

    response = client.post(
        "/api/search/kis/rerank",
        json={
            "candidate_ids": ["A", "B"],
            "subqueries": [
                {
                    "type": "abe",
                    "pairs": [
                        {"attribute": " red ", "object": " seasoning packet "},
                        {"attribute": "white", "object": "chef uniform"},
                    ],
                }
            ],
            "final_top_k": 2,
        },
    )

    assert response.status_code == 200
    query = service.rerank_kis_query_objects[0][0]
    assert query.type.value == "abe"
    assert query.text == ""
    assert [(pair.attribute, pair.object) for pair in query.pairs] == [
        ("red", "seasoning packet"),
        ("white", "chef uniform"),
    ]


def test_kis_rerank_returns_nested_score_breakdown():
    service = _FakeSearchService(hits=_ONE_KIS_HIT)
    client = _client(service)

    response = client.post(
        "/api/search/kis/rerank",
        json={
            "candidate_ids": ["L21_V001_f10"],
            "subqueries": [{"type": "ocr", "text": "VINAMILK"}],
            "final_top_k": 1,
        },
    )

    assert response.status_code == 200
    hit = response.json()["results"][0]
    assert hit["score"] == 1.0
    assert hit["final_score"] == 1.0
    assert hit["active_modalities"] == 1
    assert hit["modalities"]["ocr"]["queries"][0] == {
        "query_index": 0,
        "query": "VINAMILK",
        "raw_bm25": 7.5,
        "bm25_reference": 7.5,
        "bm25_normalized": 1.0,
        "coverage": 1.0,
        "query_score": 1.0,
        "rank": 1,
        "rank_percentile": 1.0,
    }


def test_kis_rerank_returns_nested_abe_score_breakdown():
    hit = SearchHit(
        keyframe_id="L21_V001_f10",
        video_id="L21_V001",
        score=1.0,
        frame_idx=10,
        timestamp=0.4,
        thumbnail_path="t.jpg",
        final_score=1.0,
        active_modalities=1,
        modalities={
            "abe": AbeModalityScore(
                geometric_mean=1.0,
                score=1.0,
                queries=[
                    AbeQueryScore(
                        query_index=0,
                        pairs=[
                            AbePairScore(
                                attribute="red",
                                object="packet",
                                positive_attributes=["red"],
                                negative_attributes=["white"],
                            )
                        ],
                        base_local=0.2,
                        refined_local=0.4,
                        binding_delta=0.2,
                        local_score=0.6,
                        global_abe=0.7,
                        raw_score=0.63,
                        rank=1,
                        rank_percentile=1.0,
                    )
                ],
            )
        },
    )
    service = _FakeSearchService(hits=[hit])
    client = _client(service)

    response = client.post(
        "/api/search/kis/rerank",
        json={
            "candidate_ids": ["L21_V001_f10"],
            "subqueries": [{"type": "abe", "pairs": [{"attribute": "red", "object": "packet"}]}],
            "final_top_k": 1,
        },
    )

    assert response.status_code == 200
    query = response.json()["results"][0]["modalities"]["abe"]["queries"][0]
    assert query["pairs"][0] == {
        "attribute": "red",
        "object": "packet",
        "positive_attributes": ["red"],
        "negative_attributes": ["white"],
    }
    assert query["base_local"] == 0.2
    assert query["refined_local"] == 0.4
    assert query["binding_delta"] == 0.2
    assert query["local_score"] == 0.6
    assert query["global_abe"] == 0.7
    assert query["raw_score"] == 0.63
    assert query["rank_percentile"] == 1.0


def test_kis_rerank_rejects_final_top_k_above_candidate_pool_before_reaching_service():
    service = _FakeSearchService(hits=[])
    client = _client(service)

    response = client.post(
        "/api/search/kis/rerank",
        json={
            "candidate_ids": ["A"],
            "subqueries": [{"type": "text", "text": "x"}],
            "final_top_k": 2,
        },
    )

    assert response.status_code == 422
    assert service.rerank_kis_candidates_calls == []


def test_kis_rerank_rejects_unknown_subquery_type_before_reaching_service():
    service = _FakeSearchService(hits=[])
    client = _client(service)

    response = client.post(
        "/api/search/kis/rerank",
        json={
            "candidate_ids": ["A"],
            "subqueries": [{"type": "object", "text": "person"}],
            "final_top_k": 1,
        },
    )

    assert response.status_code == 422
    assert service.rerank_kis_candidates_calls == []


def test_search_does_not_log_on_service_error(caplog):
    # A failed search never completed, so there is no request latency worth
    # logging for it -- the error path already has its own handling
    # (test_search_service_error_maps_to_unified_400_error_schema).
    service = _FakeSearchService(error=SearchServiceError("Unknown search mode: 'x'"))
    client = _client(service)

    with caplog.at_level(logging.INFO, logger="app.api.search"):
        client.post("/api/search", json={"query": "x", "mode": "semantic"})

    assert not [r for r in caplog.records if r.name == "app.api.search"]


def test_search_by_image_returns_200_with_mapped_hits():
    service = _FakeSearchService(hits=_ONE_HIT)
    client = _client(service)

    response = client.post(
        "/api/search/image", files={"file": ("query.png", _png_bytes(), "image/png")}
    )

    assert response.status_code == 200
    assert response.json() == {
        "results": [
            {
                "keyframe_id": "L21_V001_f10",
                "video_id": "L21_V001",
                "score": 0.9,
                "frame_idx": 10,
                "timestamp": 0.4,
                "thumbnail_path": "t.jpg",
                "thumbnail_url": "/api/keyframes/L21_V001_f10/thumbnail",
                "image_url": "/api/keyframes/L21_V001_f10/image",
            }
        ]
    }
    assert service.search_by_image_calls == [20]  # DEFAULT_TOP_K


def test_search_by_image_passes_through_an_explicit_top_k():
    service = _FakeSearchService(hits=[])
    client = _client(service)

    client.post(
        "/api/search/image",
        files={"file": ("query.png", _png_bytes(), "image/png")},
        data={"top_k": "7"},
    )

    assert service.search_by_image_calls == [7]


def test_search_by_image_rejects_top_k_above_the_configured_max():
    service = _FakeSearchService(hits=[])
    client = _client(service)

    response = client.post(
        "/api/search/image",
        files={"file": ("query.png", _png_bytes(), "image/png")},
        data={"top_k": "101"},
    )

    assert response.status_code == 422
    assert not service.search_by_image_calls


def test_search_by_image_allows_top_k_at_the_configured_max():
    service = _FakeSearchService(hits=[])
    client = _client(service)

    response = client.post(
        "/api/search/image",
        files={"file": ("query.png", _png_bytes(), "image/png")},
        data={"top_k": "100"},
    )

    assert response.status_code == 200
    assert service.search_by_image_calls == [100]


def test_missing_file_field_returns_422():
    service = _FakeSearchService(hits=[])
    client = _client(service)

    response = client.post("/api/search/image")

    assert response.status_code == 422
    assert not service.search_by_image_calls


def test_oversized_upload_is_rejected():
    service = _FakeSearchService(hits=[])
    client = _client(service)

    oversized = b"\x00" * (MAX_UPLOAD_SIZE_BYTES + 1)
    response = client.post(
        "/api/search/image", files={"file": ("query.png", oversized, "image/png")}
    )

    assert response.status_code == 413
    body = response.json()
    assert body["error_code"] == "upload_too_large"
    assert not service.search_by_image_calls


async def test_route_itself_still_rejects_an_oversized_body_without_a_content_length_header():
    # The middleware only catches a truthful Content-Length (module
    # docstring in app/api/search.py); a request that omits/understates it
    # (e.g. chunked transfer) reaches the route, which must still enforce
    # MAX_UPLOAD_SIZE_BYTES itself as the second layer of defense.
    oversized = b"\x00" * (MAX_UPLOAD_SIZE_BYTES + 1)
    upload = UploadFile(
        io.BytesIO(oversized), filename="query.png", headers=Headers({"content-type": "image/png"})
    )
    service = _FakeSearchService(hits=[])

    with pytest.raises(APIError) as exc_info:
        await search_api.search_by_image(file=upload, top_k=20, service=service)

    assert exc_info.value.status_code == 413
    assert exc_info.value.error_code == "upload_too_large"
    assert not service.search_by_image_calls


async def test_content_length_guard_rejects_before_reading_the_body():
    # Direct unit test of the middleware itself (registered via
    # register_upload_guard): a spoofed Content-Length over the configured
    # limit must short-circuit with 413 without ever calling `call_next`,
    # i.e. before Starlette's multipart parser touches the request body.
    from starlette.requests import Request

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/search/image",
        "headers": [(b"content-length", str(MAX_UPLOAD_SIZE_BYTES + 1).encode())],
    }
    request = Request(scope)
    call_next_invoked = False

    async def call_next(_request):
        nonlocal call_next_invoked
        call_next_invoked = True
        raise AssertionError("call_next must not run for an oversized Content-Length")

    response = await search_api._content_length_guard(request, call_next)

    assert response.status_code == 413
    assert call_next_invoked is False


def test_non_image_mime_is_rejected():
    service = _FakeSearchService(hits=[])
    client = _client(service)

    response = client.post(
        "/api/search/image", files={"file": ("query.txt", b"not an image", "text/plain")}
    )

    assert response.status_code == 415
    body = response.json()
    assert body["error_code"] == "unsupported_media_type"
    assert not service.search_by_image_calls


def test_corrupt_image_bytes_are_rejected():
    service = _FakeSearchService(hits=[])
    client = _client(service)

    response = client.post(
        "/api/search/image",
        files={"file": ("query.png", b"not actually a png", "image/png")},
    )

    assert response.status_code == 400
    body = response.json()
    assert body["error_code"] == "invalid_image"
    assert not service.search_by_image_calls


def test_zero_byte_upload_is_rejected():
    service = _FakeSearchService(hits=[])
    client = _client(service)

    response = client.post("/api/search/image", files={"file": ("query.png", b"", "image/png")})

    assert response.status_code == 400
    assert response.json()["error_code"] == "invalid_image"
    assert not service.search_by_image_calls


def test_truncated_image_bytes_are_rejected():
    # Regression: a truncated-but-identifiable file raises plain OSError
    # from Image.load(), not UnidentifiedImageError -- must still map to 400.
    service = _FakeSearchService(hits=[])
    client = _client(service)

    response = client.post(
        "/api/search/image",
        files={"file": ("query.jpg", _truncated_jpeg_bytes(), "image/jpeg")},
    )

    assert response.status_code == 400
    assert response.json()["error_code"] == "invalid_image"
    assert not service.search_by_image_calls


def test_oversized_image_dimensions_are_rejected():
    # Regression: Pillow raises DecompressionBombError (not OSError) for a
    # small file with a huge declared pixel count -- must still map to 400,
    # not the unhandled-exception 500 path.
    service = _FakeSearchService(hits=[])
    client = _client(service)

    response = client.post(
        "/api/search/image",
        files={"file": ("bomb.png", _decompression_bomb_png_bytes(), "image/png")},
    )

    assert response.status_code == 400
    assert response.json()["error_code"] == "invalid_image"
    assert not service.search_by_image_calls


def test_declared_mime_type_mismatching_actual_format_is_rejected():
    # Regression: Content-Type is only a caller-supplied label -- Image.open()
    # detects format from magic bytes. A BMP declared as image/png must be
    # rejected, not silently encoded as whatever format it actually is.
    service = _FakeSearchService(hits=[])
    client = _client(service)

    response = client.post(
        "/api/search/image", files={"file": ("query.png", _bmp_bytes(), "image/png")}
    )

    assert response.status_code == 415
    assert response.json()["error_code"] == "unsupported_media_type"
    assert not service.search_by_image_calls
