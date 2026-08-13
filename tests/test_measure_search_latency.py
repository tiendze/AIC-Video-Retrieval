"""RED reproducer for scripts/measure_search_latency.py (plan §P1.7,
PRD §6.6) -- benchmarks POST /api/search's real p50/p95 against a LIVE
server. The HTTP layer is faked with `httpx.MockTransport` (no live
docker-compose/uvicorn needed for these tests -- CLAUDE.md §10: the script
itself is prepared here for the project owner to run against their own
live stack, not run automatically by this agent).
"""

from __future__ import annotations

import json

import httpx
import pytest

from scripts.measure_search_latency import (
    _benchmark_mode,
    _load_queries,
    _render_report,
)
from eval.metrics.latency import LatencyPercentiles


def _write_eval_set(tmp_path, records: list[dict]) -> "Path":  # noqa: F821
    path = tmp_path / "eval_set.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    return path


def test_load_queries_reads_captions_from_eval_set_json(tmp_path):
    path = _write_eval_set(
        tmp_path,
        [
            {"caption": "first caption", "video_id": "L21_V001", "order_index": 0},
            {"caption": "second caption", "video_id": "L21_V001", "order_index": 1},
        ],
    )

    queries = _load_queries(path, limit=None)

    assert queries == ["first caption", "second caption"]


def test_load_queries_respects_limit(tmp_path):
    path = _write_eval_set(
        tmp_path, [{"caption": f"caption {i}", "video_id": "v", "order_index": i} for i in range(10)]
    )

    queries = _load_queries(path, limit=3)

    assert queries == ["caption 0", "caption 1", "caption 2"]


def test_benchmark_mode_posts_mode_and_query_and_returns_one_latency_per_query():
    posted_bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        posted_bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"results": []})

    client = httpx.Client(transport=httpx.MockTransport(handler))

    latencies_ms = _benchmark_mode(
        client, "http://testserver", "semantic", ["query one", "query two"], top_k=20
    )

    assert len(latencies_ms) == 2
    assert all(isinstance(latency, float) and latency >= 0.0 for latency in latencies_ms)
    assert posted_bodies == [
        {"query": "query one", "mode": "semantic", "top_k": 20},
        {"query": "query two", "mode": "semantic", "top_k": 20},
    ]


def test_benchmark_mode_raises_on_http_error_status():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"detail": "boom"})

    client = httpx.Client(transport=httpx.MockTransport(handler))

    with pytest.raises(httpx.HTTPStatusError):
        _benchmark_mode(client, "http://testserver", "semantic", ["query"], top_k=20)


def test_render_report_formats_a_markdown_table_with_p50_p95_and_count():
    results = {
        "semantic": LatencyPercentiles(p50_ms=120.4, p95_ms=280.9, num_requests=173),
        "ocr": LatencyPercentiles(p50_ms=45.0, p95_ms=90.0, num_requests=173),
    }

    report = _render_report(results)

    assert "semantic" in report
    assert "ocr" in report
    assert "120" in report  # p50 rendered (rounded)
    assert "281" in report or "280" in report  # p95 rendered (rounded)
    assert "173" in report
