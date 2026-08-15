"""RED reproducer for `POST /api/quest/image-candidates` and `POST
/api/quest/image-search` (plan §P2.5, PRD §F5) -- thin routers: validate the
request, delegate to `QuestService`, map `QuestServiceError` onto the
unified `{detail, error_code, request_id}` schema (CLAUDE.md §7,
.claude/rules/ecc/python/fastapi.md).

A Fake `QuestService` is injected via `app.dependency_overrides` -- no
ddgs/network/GPU needed; QuestService's own behavior is already covered by
tests/test_quest_service.py.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import quest as quest_api
from app.api.errors import register_error_handling
from app.dependencies import get_quest_service
from app.schemas.request.search import DEFAULT_TOP_K, MAX_TOP_K
from app.services.quest_service import (
    ImageCandidate,
    ImageCandidatesResult,
    ImageSearchResult,
    QuestServiceError,
)
from app.services.search_service import SearchHit


class _FakeQuestService:
    def __init__(
        self,
        candidates_result: ImageCandidatesResult | None = None,
        search_result: ImageSearchResult | None = None,
        search_error: Exception | None = None,
    ) -> None:
        self._candidates_result = candidates_result
        self._search_result = search_result
        self._search_error = search_error
        self.find_image_candidates_calls: list[tuple[str, int]] = []
        self.search_by_candidate_calls: list[tuple[str, str, int]] = []

    async def find_image_candidates(self, query: str, top_k: int) -> ImageCandidatesResult:
        self.find_image_candidates_calls.append((query, top_k))
        return self._candidates_result

    async def search_by_candidate(
        self, candidate_set_id: str, candidate_id: str, top_k: int
    ) -> ImageSearchResult:
        self.search_by_candidate_calls.append((candidate_set_id, candidate_id, top_k))
        if self._search_error is not None:
            raise self._search_error
        return self._search_result


def _client(fake_service: _FakeQuestService) -> TestClient:
    app = FastAPI()
    register_error_handling(app)
    app.include_router(quest_api.router)
    app.dependency_overrides[get_quest_service] = lambda: fake_service
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
_ONE_HIT_RESPONSE = [
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


def test_image_candidates_returns_200_with_status_candidates():
    result = ImageCandidatesResult(
        status="candidates",
        candidate_set_id="set-1",
        candidates=[ImageCandidate(candidate_id="c1", thumbnail_url="https://x/t1.jpg", title="A")],
        fallback_hits=[],
    )
    service = _FakeQuestService(candidates_result=result)
    client = _client(service)

    response = client.post("/api/quest/image-candidates", json={"query": "thu tuong"})

    assert response.status_code == 200
    assert response.json() == {
        "status": "candidates",
        "candidate_set_id": "set-1",
        "candidates": [{"candidate_id": "c1", "thumbnail_url": "https://x/t1.jpg", "title": "A"}],
        "fallback_hits": [],
    }


def test_image_candidates_returns_200_with_status_fallback_and_hits():
    result = ImageCandidatesResult(
        status="fallback", candidate_set_id=None, candidates=[], fallback_hits=_ONE_HIT
    )
    service = _FakeQuestService(candidates_result=result)
    client = _client(service)

    response = client.post("/api/quest/image-candidates", json={"query": "thu tuong"})

    assert response.status_code == 200
    assert response.json() == {
        "status": "fallback",
        "candidate_set_id": None,
        "candidates": [],
        "fallback_hits": _ONE_HIT_RESPONSE,
    }


def test_image_candidates_uses_default_top_k_when_omitted():
    result = ImageCandidatesResult(
        status="fallback", candidate_set_id=None, candidates=[], fallback_hits=[]
    )
    service = _FakeQuestService(candidates_result=result)
    client = _client(service)

    client.post("/api/quest/image-candidates", json={"query": "x"})

    assert service.find_image_candidates_calls == [("x", DEFAULT_TOP_K)]


def test_image_candidates_passes_through_an_explicit_top_k():
    result = ImageCandidatesResult(
        status="fallback", candidate_set_id=None, candidates=[], fallback_hits=[]
    )
    service = _FakeQuestService(candidates_result=result)
    client = _client(service)

    client.post("/api/quest/image-candidates", json={"query": "x", "top_k": 5})

    assert service.find_image_candidates_calls == [("x", 5)]


def test_image_candidates_rejects_empty_query():
    service = _FakeQuestService()
    client = _client(service)

    response = client.post("/api/quest/image-candidates", json={"query": ""})

    assert response.status_code == 422
    assert not service.find_image_candidates_calls


def test_image_candidates_rejects_top_k_above_the_configured_max():
    service = _FakeQuestService()
    client = _client(service)

    response = client.post(
        "/api/quest/image-candidates", json={"query": "x", "top_k": MAX_TOP_K + 1}
    )

    assert response.status_code == 422
    assert not service.find_image_candidates_calls


def test_image_search_returns_200_with_status_ok():
    result = ImageSearchResult(status="ok", hits=_ONE_HIT)
    service = _FakeQuestService(search_result=result)
    client = _client(service)

    response = client.post(
        "/api/quest/image-search",
        json={"candidate_set_id": "set-1", "candidate_id": "c1"},
    )

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "hits": _ONE_HIT_RESPONSE}
    assert service.search_by_candidate_calls == [("set-1", "c1", DEFAULT_TOP_K)]


def test_image_search_returns_200_with_status_fallback():
    result = ImageSearchResult(status="fallback", hits=[])
    service = _FakeQuestService(search_result=result)
    client = _client(service)

    response = client.post(
        "/api/quest/image-search",
        json={"candidate_set_id": "set-1", "candidate_id": "c1"},
    )

    assert response.status_code == 200
    assert response.json() == {"status": "fallback", "hits": []}


def test_image_search_passes_through_an_explicit_top_k():
    result = ImageSearchResult(status="ok", hits=[])
    service = _FakeQuestService(search_result=result)
    client = _client(service)

    client.post(
        "/api/quest/image-search",
        json={"candidate_set_id": "set-1", "candidate_id": "c1", "top_k": 7},
    )

    assert service.search_by_candidate_calls == [("set-1", "c1", 7)]


def test_image_search_rejects_missing_fields():
    service = _FakeQuestService()
    client = _client(service)

    response = client.post("/api/quest/image-search", json={"candidate_id": "c1"})

    assert response.status_code == 422
    assert not service.search_by_candidate_calls


def test_image_search_candidate_set_not_found_maps_to_unified_404_error_schema():
    error = QuestServiceError(
        "Unknown or expired candidate_set_id: 'set-x'",
        error_code="candidate_set_not_found",
        status_code=404,
    )
    service = _FakeQuestService(search_error=error)
    client = _client(service)

    response = client.post(
        "/api/quest/image-search",
        json={"candidate_set_id": "set-x", "candidate_id": "c1"},
    )

    assert response.status_code == 404
    body = response.json()
    assert body["error_code"] == "candidate_set_not_found"
    assert set(body.keys()) == {"detail", "error_code", "request_id"}


def test_image_search_candidate_not_found_maps_to_unified_404_error_schema():
    error = QuestServiceError(
        "Unknown candidate_id: 'c-x'", error_code="candidate_not_found", status_code=404
    )
    service = _FakeQuestService(search_error=error)
    client = _client(service)

    response = client.post(
        "/api/quest/image-search",
        json={"candidate_set_id": "set-1", "candidate_id": "c-x"},
    )

    assert response.status_code == 404
    assert response.json()["error_code"] == "candidate_not_found"
