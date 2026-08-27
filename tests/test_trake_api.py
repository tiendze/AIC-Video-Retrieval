"""`POST /api/trake` router tests -- validation, delegation, response mapping.

No pipeline math here: that is tests/test_trake_chain.py (algorithm) and
tests/test_trake_service.py (orchestration). A fake `TrakeService` is injected
through `app.dependency_overrides`, mirroring tests/test_search_api.py -- no
Docker, no GPU.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import trake as trake_api
from app.api.errors import register_error_handling
from app.dependencies import get_trake_service
from app.services.trake_chain import CandidateFrame, ChainCandidate, VideoScore
from app.services.trake_embedding_cache import TrakeEmbeddingCacheError
from app.services.trake_service import (
    TrakeEventDiagnostics,
    TrakeSearchResult,
    TrakeServiceError,
)


class _FakeTrakeService:
    def __init__(
        self,
        result: TrakeSearchResult | None = None,
        error: Exception | None = None,
    ) -> None:
        self._result = result or TrakeSearchResult(candidates=(), video_scores={})
        self._error = error
        self.search_calls: list[tuple[list, int, float]] = []

    async def search(self, events, top_k, threshold=0.0):
        self.search_calls.append((list(events), top_k, threshold))
        if self._error is not None:
            raise self._error
        return self._result


def _client(fake_service: _FakeTrakeService) -> TestClient:
    app = FastAPI()
    register_error_handling(app)
    app.include_router(trake_api.router)
    app.dependency_overrides[get_trake_service] = lambda: fake_service
    return TestClient(app, raise_server_exceptions=False)


def _frame(keyframe_id: str, frame_idx: int, timestamp: float) -> CandidateFrame:
    return CandidateFrame(
        corpus_index=frame_idx,
        keyframe_id=keyframe_id,
        video_id="L21_V001",
        frame_idx=frame_idx,
        timestamp=timestamp,
        score=1.5,
        semantic_z=1.2,
        kis_rank=3,
    )


_ONE_RESULT = TrakeSearchResult(
    candidates=(
        ChainCandidate(
            video_id="L21_V001",
            frames=(
                _frame("L21_V001_f10", 10, 0.4),
                _frame("L21_V001_f20", 20, 0.8),
            ),
            min_score=1.1,
            mean_score=1.3,
            coverage=1.0,
            score=1.23,
        ),
    ),
    video_scores={
        "L21_V001": VideoScore(
            video_id="L21_V001",
            score=2.5,
            coverage=1.0,
            best_chain_score=1.23,
            rank_evidence=0.8,
            num_keyframes=120,
            num_chains=1,
        )
    },
    events=(TrakeEventDiagnostics(0, 12, 5, True), TrakeEventDiagnostics(1, 9, 4, True)),
    candidate_video_count=7,
    chained_video_count=3,
    dropped_video_count=4,
)


def _body(**overrides):
    payload = {"events": [{"text_query": "su kien 1"}, {"text_query": "su kien 2"}]}
    payload.update(overrides)
    return payload


def test_trake_returns_200_and_maps_the_sequence_onto_the_response_schema():
    client = _client(_FakeTrakeService(result=_ONE_RESULT))

    response = client.post("/api/trake", json=_body())

    assert response.status_code == 200
    payload = response.json()
    assert payload["results"] == [
        {
            "video_id": "L21_V001",
            "score": 1.23,
            "min_score": 1.1,
            "mean_score": 1.3,
            "coverage": 1.0,
            "video_score": 2.5,
            "keyframes": [
                {
                    "keyframe_id": "L21_V001_f10",
                    "frame_idx": 10,
                    "timestamp": 0.4,
                    "thumbnail_url": "/api/keyframes/L21_V001_f10/thumbnail",
                    "image_url": "/api/keyframes/L21_V001_f10/image",
                    "score": 1.5,
                    "semantic_z": 1.2,
                    "kis_rank": 3,
                    "local_score": None,
                    "local_z": 0.0,
                },
                {
                    "keyframe_id": "L21_V001_f20",
                    "frame_idx": 20,
                    "timestamp": 0.8,
                    "thumbnail_url": "/api/keyframes/L21_V001_f20/thumbnail",
                    "image_url": "/api/keyframes/L21_V001_f20/image",
                    "score": 1.5,
                    "semantic_z": 1.2,
                    "kis_rank": 3,
                    "local_score": None,
                    "local_z": 0.0,
                },
            ],
        }
    ]


def test_trake_reports_pipeline_diagnostics_alongside_the_results():
    client = _client(_FakeTrakeService(result=_ONE_RESULT))

    payload = client.post("/api/trake", json=_body()).json()

    assert payload["candidate_video_count"] == 7
    assert payload["chained_video_count"] == 3
    assert payload["dropped_video_count"] == 4
    assert [event["kis_candidate_count"] for event in payload["events"]] == [12, 9]
    # Defaults for a result that never ran B2b (plan §T3).
    assert payload["local_rerank_video_count"] == 0
    assert payload["local_missing_frame_count"] == 0
    assert [event["text_truncated"] for event in payload["events"]] == [False, False]


def test_trake_maps_b2b_local_rerank_diagnostics_onto_the_response():
    result = TrakeSearchResult(
        candidates=(),
        video_scores={},
        events=(TrakeEventDiagnostics(0, 1, 1, True, text_truncated=True),),
        local_rerank_video_count=5,
        local_missing_frame_count=2,
    )
    client = _client(_FakeTrakeService(result=result))

    payload = client.post("/api/trake", json=_body()).json()

    assert payload["local_rerank_video_count"] == 5
    assert payload["local_missing_frame_count"] == 2
    assert payload["events"][0]["text_truncated"] is True


def test_trake_passes_local_phrases_through_to_the_service():
    service = _FakeTrakeService()
    client = _client(service)

    client.post(
        "/api/trake",
        json={
            "events": [
                {"text_query": "cau thu chay len", "local_phrases": ["ao do", "ao do"]},
                {"text_query": "sut bong"},
            ],
        },
    )

    events, _, _ = service.search_calls[0]
    assert events[0].local_phrases == ("ao do",)
    assert events[1].local_phrases == ()


def test_trake_passes_per_event_channels_through_to_the_service():
    service = _FakeTrakeService()
    client = _client(service)

    client.post(
        "/api/trake",
        json={
            "events": [
                {
                    "text_query": "cau thu chay len",
                    "ocr_query": "ty so 0-0",
                    "title_query": "chung ket",
                    "asr_query": "mo man tran dau",
                    "asr_match_mode": "all",
                },
                {"text_query": "sut bong"},
            ],
            "top_k": 5,
        },
    )

    events, top_k, _ = service.search_calls[0]
    assert top_k == 5
    assert events[0].text_queries == ("cau thu chay len",)
    assert events[0].ocr_query == "ty so 0-0"
    assert events[0].title_query == "chung ket"
    assert events[0].asr_queries == ("mo man tran dau",)
    assert events[0].asr_match_mode == "all"
    assert events[1].ocr_query is None


def test_trake_accepts_an_event_with_only_an_ocr_box():
    service = _FakeTrakeService()
    client = _client(service)

    response = client.post(
        "/api/trake",
        json={"events": [{"text_query": "a"}, {"ocr_query": "ty so 1-0"}]},
    )

    assert response.status_code == 200
    events, _, _ = service.search_calls[0]
    assert events[1].text_queries == ()
    assert events[1].ocr_query == "ty so 1-0"


def test_trake_rejects_an_event_with_no_channel_at_all():
    service = _FakeTrakeService()
    client = _client(service)

    response = client.post("/api/trake", json={"events": [{"text_query": "a"}, {}]})

    assert response.status_code == 422
    assert not service.search_calls


def test_trake_treats_a_whitespace_only_box_as_absent():
    service = _FakeTrakeService()
    client = _client(service)

    response = client.post("/api/trake", json={"events": [{"text_query": "   "}]})

    assert response.status_code == 422
    assert not service.search_calls


def test_trake_passes_an_explicit_threshold_through():
    service = _FakeTrakeService()
    client = _client(service)

    client.post("/api/trake", json=_body(threshold=1.25))

    assert service.search_calls[0][2] == 1.25


def test_trake_rejects_an_empty_events_list_before_reaching_the_service():
    service = _FakeTrakeService()
    client = _client(service)

    response = client.post("/api/trake", json={"events": []})

    assert response.status_code == 422
    assert not service.search_calls


def test_trake_rejects_top_k_above_the_configured_max():
    service = _FakeTrakeService()
    client = _client(service)

    response = client.post("/api/trake", json=_body(top_k=101))

    assert response.status_code == 422
    assert not service.search_calls


def test_trake_returns_an_empty_results_list_when_no_video_matches():
    client = _client(_FakeTrakeService())

    response = client.post("/api/trake", json=_body())

    assert response.status_code == 200
    assert response.json()["results"] == []


def test_trake_service_error_keeps_its_own_error_code_and_status():
    service = _FakeTrakeService(
        error=TrakeServiceError(
            "Event 1: Elasticsearch unavailable",
            error_code="ocr_unavailable",
            status_code=503,
        )
    )
    client = _client(service)

    response = client.post("/api/trake", json=_body())

    assert response.status_code == 503
    assert response.json()["error_code"] == "ocr_unavailable"


def test_trake_cache_unavailable_error_maps_to_a_503():
    service = _FakeTrakeService(error=TrakeEmbeddingCacheError("TRAKE cache unavailable"))
    client = _client(service)

    response = client.post("/api/trake", json=_body())

    assert response.status_code == 503
    assert response.json()["error_code"] == "trake_cache_unavailable"
