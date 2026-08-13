"""RED reproducer for eval/metrics/latency.py (plan §P1.7, PRD §6.6) -- pure
percentile aggregation over a list of per-request latencies. No HTTP/server
involved: scripts/measure_search_latency.py collects the raw latencies_ms,
this module only turns them into p50/p95.
"""

from __future__ import annotations

import pytest

from eval.metrics.latency import LatencyPercentiles, compute_latency_percentiles


def test_compute_latency_percentiles_returns_p50_and_p95():
    # 1..100 ms: p50 is the median (50.5), p95 is the 95th percentile (95.05)
    # -- both computed via numpy's default linear interpolation, matched here
    # by hand for a round input so the test doesn't just re-implement numpy.
    latencies_ms = [float(i) for i in range(1, 101)]

    result = compute_latency_percentiles(latencies_ms)

    assert result.p50_ms == pytest.approx(50.5)
    assert result.p95_ms == pytest.approx(95.05)
    assert result.num_requests == 100


def test_compute_latency_percentiles_single_value():
    result = compute_latency_percentiles([42.0])

    assert result.p50_ms == pytest.approx(42.0)
    assert result.p95_ms == pytest.approx(42.0)
    assert result.num_requests == 1


def test_compute_latency_percentiles_raises_on_empty():
    with pytest.raises(ValueError, match="latencies_ms is empty"):
        compute_latency_percentiles([])


def test_latency_percentiles_is_frozen():
    result = LatencyPercentiles(p50_ms=1.0, p95_ms=2.0, num_requests=1)

    with pytest.raises(AttributeError):
        result.p50_ms = 99.0  # type: ignore[misc]
