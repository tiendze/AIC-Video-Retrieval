"""RED reproducer for HealthService (plan §P0.8) -- `GET /health` must check
all three stores at once and report degraded state as data, never a 500
(CLAUDE.md §4: turning off Elasticsearch must still leave semantic search
usable, so `/health` itself must survive a store being down).

Fakes stand in for every store -- no Docker needed here.
"""

from __future__ import annotations

import pytest
from elasticsearch import ApiError
from elasticsearch import ConnectionError as EsConnectionError

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


class _FakeAsrEsStore:
    def __init__(
        self,
        index_exists: bool | None,
        segment_count: int | None,
        *,
        index_error: Exception | None = None,
        count_error: Exception | None = None,
    ) -> None:
        self._index_exists = index_exists
        self._segment_count = segment_count
        self._index_error = index_error
        self._count_error = count_error
        self.index_exists_calls = 0
        self.count_calls = 0

    async def index_exists(self) -> bool:
        self.index_exists_calls += 1
        if self._index_error is not None:
            raise self._index_error
        return bool(self._index_exists)

    async def count_segments(self) -> int:
        self.count_calls += 1
        if self._count_error is not None:
            raise self._count_error
        assert self._segment_count is not None
        return self._segment_count


def _service(
    milvus_ok: bool,
    es_ok: bool,
    mongo_ok: bool,
    asr_store: _FakeAsrEsStore | None = None,
) -> HealthService:
    return HealthService(
        milvus_store=_FakeMilvusStore(milvus_ok),
        es_store=_FakeEsStore(es_ok),
        mongo_store=_FakeMongoStore(mongo_ok),
        asr_es_store=asr_store or _FakeAsrEsStore(index_exists=True, segment_count=1),
    )


async def test_reports_ok_when_all_three_stores_are_healthy():
    service = _service(milvus_ok=True, es_ok=True, mongo_ok=True)

    report = await service.check()

    assert report == HealthReport(
        status="ok",
        milvus_ok=True,
        elasticsearch_ok=True,
        mongodb_ok=True,
        asr_status="ready",
        asr_ready=True,
        asr_segment_count=1,
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
        status="degraded",
        milvus_ok=False,
        elasticsearch_ok=False,
        mongodb_ok=False,
        asr_status="unreachable",
        asr_ready=False,
        asr_segment_count=None,
    )


@pytest.mark.parametrize(
    ("es_ok", "index_exists", "count", "asr_status", "asr_count"),
    [
        (False, None, None, "unreachable", None),
        (True, False, None, "unavailable", None),
        (True, True, 0, "empty", 0),
        (True, True, 873, "ready", 873),
    ],
)
async def test_health_reports_explicit_asr_readiness(
    es_ok, index_exists, count, asr_status, asr_count
):
    asr_store = _FakeAsrEsStore(index_exists=index_exists, segment_count=count)

    report = await _service(True, es_ok, True, asr_store).check()

    assert report.asr_status == asr_status
    assert report.asr_ready is (asr_status == "ready")
    assert report.asr_segment_count == asr_count
    assert report.status == ("ok" if asr_status == "ready" else "degraded")
    if not es_ok:
        assert asr_store.index_exists_calls == 0
        assert asr_store.count_calls == 0


@pytest.mark.parametrize(
    ("index_error", "count_error"),
    [
        (EsConnectionError("probe failed"), None),
        (None, ApiError("503", {}, {})),
    ],
)
async def test_asr_probe_failure_is_unavailable_not_a_health_exception(index_error, count_error):
    asr_store = _FakeAsrEsStore(
        index_exists=True,
        segment_count=873,
        index_error=index_error,
        count_error=count_error,
    )

    report = await _service(True, True, True, asr_store).check()

    assert report.status == "degraded"
    assert report.asr_status == "unavailable"
    assert report.asr_ready is False
    assert report.asr_segment_count is None
