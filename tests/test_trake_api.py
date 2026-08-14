"""RED reproducer for `POST /api/trake` (plan §P2.2, architecture.md §3.5) --
thin router: validate the request, delegate to `TrakeService`, map its
results onto the response schema. No DANTE math here -- that's
tests/test_trake_service.py and tests/test_dante_service.py.

A Fake `TrakeService` is injected via `app.dependency_overrides`, mirroring
tests/test_search_api.py -- no Docker/GPU needed.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import trake as trake_api
from app.api.errors import register_error_handling
from app.dependencies import get_trake_service
from app.services.trake_service import TrakeKeyframe, TrakeVideoResult


class _FakeTrakeService:
    def __init__(self, results: list[TrakeVideoResult] | None = None) -> None:
        self._results = results or []
        self.search_calls: list[tuple[list[str], int, float | None]] = []

    async def search(self, events, top_k, lambda_penalty=None):
        self.search_calls.append((list(events), top_k, lambda_penalty))
        return self._results


def _client(fake_service: _FakeTrakeService) -> TestClient:
    app = FastAPI()
    register_error_handling(app)
    app.include_router(trake_api.router)
    app.dependency_overrides[get_trake_service] = lambda: fake_service
    return TestClient(app, raise_server_exceptions=False)


_ONE_RESULT = [
    TrakeVideoResult(
        video_id="L21_V001",
        score=1.23,
        keyframes=(
            TrakeKeyframe(keyframe_id="L21_V001_f10", frame_idx=10, timestamp=0.4),
            TrakeKeyframe(keyframe_id="L21_V001_f20", frame_idx=20, timestamp=0.8),
        ),
    )
]


def test_trake_returns_200_with_mapped_results():
    service = _FakeTrakeService(results=_ONE_RESULT)
    client = _client(service)

    response = client.post("/api/trake", json={"events": ["su kien 1", "su kien 2"]})

    assert response.status_code == 200
    assert response.json() == {
        "results": [
            {
                "video_id": "L21_V001",
                "score": 1.23,
                "keyframes": [
                    {
                        "keyframe_id": "L21_V001_f10",
                        "frame_idx": 10,
                        "timestamp": 0.4,
                        "thumbnail_url": "/api/keyframes/L21_V001_f10/thumbnail",
                        "image_url": "/api/keyframes/L21_V001_f10/image",
                    },
                    {
                        "keyframe_id": "L21_V001_f20",
                        "frame_idx": 20,
                        "timestamp": 0.8,
                        "thumbnail_url": "/api/keyframes/L21_V001_f20/thumbnail",
                        "image_url": "/api/keyframes/L21_V001_f20/image",
                    },
                ],
            }
        ]
    }


def test_trake_passes_events_and_top_k_through_to_the_service():
    service = _FakeTrakeService()
    client = _client(service)

    client.post("/api/trake", json={"events": ["a", "b"], "top_k": 5})

    assert service.search_calls == [(["a", "b"], 5, None)]


def test_trake_passes_an_explicit_lambda_penalty_through():
    service = _FakeTrakeService()
    client = _client(service)

    client.post("/api/trake", json={"events": ["a"], "lambda_penalty": 0.01})

    assert service.search_calls == [(["a"], 10, 0.01)]  # 10 = configured default_top_k


def test_trake_rejects_an_empty_events_list_before_reaching_the_service():
    service = _FakeTrakeService()
    client = _client(service)

    response = client.post("/api/trake", json={"events": []})

    assert response.status_code == 422
    assert not service.search_calls


def test_trake_rejects_top_k_above_the_configured_max():
    service = _FakeTrakeService()
    client = _client(service)

    response = client.post("/api/trake", json={"events": ["a"], "top_k": 101})

    assert response.status_code == 422
    assert not service.search_calls


def test_trake_returns_an_empty_results_list_when_no_video_matches():
    service = _FakeTrakeService(results=[])
    client = _client(service)

    response = client.post("/api/trake", json={"events": ["a"]})

    assert response.status_code == 200
    assert response.json() == {"results": []}
