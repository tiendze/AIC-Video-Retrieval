"""RED reproducer for HealthService (plan §P0.8) -- `GET /health` must check
all three stores at once and report degraded state as data, never a 500
(CLAUDE.md §4: turning off Elasticsearch must still leave semantic search
usable, so `/health` itself must survive a store being down).

Fakes stand in for every store -- no Docker needed here.
"""

from __future__ import annotations

from app.services.health_service import HealthReport, HealthService


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


def _service(milvus_ok: bool, es_ok: bool, mongo_ok: bool) -> HealthService:
    return HealthService(
        milvus_store=_FakeMilvusStore(milvus_ok),
        es_store=_FakeEsStore(es_ok),
        mongo_store=_FakeMongoStore(mongo_ok),
    )


async def test_reports_ok_when_all_three_stores_are_healthy():
    service = _service(milvus_ok=True, es_ok=True, mongo_ok=True)

    report = await service.check()

    assert report == HealthReport(
        status="ok", milvus_ok=True, elasticsearch_ok=True, mongodb_ok=True
    )


async def test_reports_degraded_when_elasticsearch_is_down_but_does_not_raise():
    service = _service(milvus_ok=True, es_ok=False, mongo_ok=True)

    report = await service.check()

    assert report.status == "degraded"
    assert report.elasticsearch_ok is False
    assert report.milvus_ok is True
    assert report.mongodb_ok is True


async def test_reports_degraded_when_any_single_store_is_down():
    for milvus_ok, es_ok, mongo_ok in [
        (False, True, True),
        (True, True, False),
    ]:
        report = await _service(milvus_ok, es_ok, mongo_ok).check()
        assert report.status == "degraded"


async def test_reports_degraded_when_every_store_is_down():
    service = _service(milvus_ok=False, es_ok=False, mongo_ok=False)

    report = await service.check()

    assert report == HealthReport(
        status="degraded", milvus_ok=False, elasticsearch_ok=False, mongodb_ok=False
    )
