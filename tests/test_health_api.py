"""RED reproducer for `GET /health` (plan §P0.8) -- the router is thin
(validate + delegate, .claude/rules/ecc/python/fastapi.md): it only builds
a `HealthService` from injected stores and returns its report. Store
instances are faked directly on `app.state` here since `app/main.py`'s
lifespan doesn't build real ones yet (that's a later step) -- this proves
the router + dependency wiring independent of that.
"""

from __future__ import annotations

from elasticsearch import ApiError
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.health import router as health_router


class _FakeMilvusStore:
    def __init__(self, healthy: bool) -> None:
        self._healthy = healthy

    def ping(self) -> bool:
        return self._healthy


class _FakeEsStore:
    def __init__(self, healthy: bool) -> None:
        self._healthy = healthy

    async def ping(self) -> bool:
        return self._healthy


class _FakeMongoStore:
    def __init__(self, healthy: bool) -> None:
        self._healthy = healthy

    async def ping(self) -> bool:
        return self._healthy


class _FakeAsrEsStore:
    def __init__(
        self,
        index_exists: bool = True,
        segment_count: int = 873,
        error: Exception | None = None,
    ) -> None:
        self._index_exists = index_exists
        self._segment_count = segment_count
        self._error = error

    async def index_exists(self) -> bool:
        if self._error is not None:
            raise self._error
        return self._index_exists

    async def count_segments(self) -> int:
        if self._error is not None:
            raise self._error
        return self._segment_count


def _client(
    milvus_ok: bool,
    es_ok: bool,
    mongo_ok: bool,
    asr_store: _FakeAsrEsStore | None = None,
) -> TestClient:
    app = FastAPI()
    app.include_router(health_router)
    app.state.milvus_store = _FakeMilvusStore(milvus_ok)
    app.state.es_store = _FakeEsStore(es_ok)
    app.state.mongo_store = _FakeMongoStore(mongo_ok)
    app.state.asr_es_store = asr_store or _FakeAsrEsStore()
    return TestClient(app)


def test_health_returns_200_and_ok_when_all_stores_are_healthy():
    client = _client(milvus_ok=True, es_ok=True, mongo_ok=True)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "milvus_ok": True,
        "elasticsearch_ok": True,
        "mongodb_ok": True,
        "asr_status": "ready",
        "asr_ready": True,
        "asr_segment_count": 873,
    }


def test_health_returns_200_and_degraded_when_a_store_is_down():
    # /health must never itself 500 just because a dependency is down
    # (CLAUDE.md §4 degraded mode) -- it reports the fact as data.
    client = _client(milvus_ok=True, es_ok=False, mongo_ok=True)

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["elasticsearch_ok"] is False
    assert body["asr_status"] == "unreachable"
    assert body["asr_ready"] is False
    assert body["asr_segment_count"] is None


def test_health_returns_200_when_asr_probe_raises_api_error():
    client = _client(
        milvus_ok=True,
        es_ok=True,
        mongo_ok=True,
        asr_store=_FakeAsrEsStore(error=ApiError("503", {}, {})),
    )

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "degraded",
        "milvus_ok": True,
        "elasticsearch_ok": True,
        "mongodb_ok": True,
        "asr_status": "unavailable",
        "asr_ready": False,
        "asr_segment_count": None,
    }
