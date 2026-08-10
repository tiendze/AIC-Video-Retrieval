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

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.errors import register_error_handling
from app.api.search import router as search_router
from app.dependencies import get_search_service
from app.services.search_service import SearchHit, SearchServiceError


class _FakeSearchService:
    def __init__(self, hits: list[SearchHit] | None = None, error: Exception | None = None) -> None:
        self._hits = hits or []
        self._error = error
        self.search_calls: list[tuple[str, str, int]] = []

    async def search(self, query: str, mode: str, top_k: int) -> list[SearchHit]:
        self.search_calls.append((query, mode, top_k))
        if self._error is not None:
            raise self._error
        return self._hits


def _client(fake_service: _FakeSearchService) -> TestClient:
    app = FastAPI()
    register_error_handling(app)
    app.include_router(search_router)
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

    response = client.post("/api/search", json={"query": "x", "mode": "hybrid"})

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
