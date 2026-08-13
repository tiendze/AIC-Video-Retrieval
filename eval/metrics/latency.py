"""Latency percentile aggregation for `scripts/measure_search_latency.py`
(plan §P1.7, PRD §6.6) -- pure function so the percentile math is testable
without a live server, mirroring `eval/metrics/retrieval.py`'s split between
pure scoring functions and the store/HTTP code that feeds them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LatencyPercentiles:
    p50_ms: float
    p95_ms: float
    num_requests: int


def compute_latency_percentiles(latencies_ms: Sequence[float]) -> LatencyPercentiles:
    """Raises `ValueError` on an empty input rather than a misleading 0.0
    (CLAUDE.md §7 -- missing data must fail loudly), matching
    `eval/metrics/retrieval.py`'s `_require_nonempty` convention."""
    if not latencies_ms:
        raise ValueError("compute_latency_percentiles: latencies_ms is empty -- nothing was measured")
    return LatencyPercentiles(
        p50_ms=float(np.percentile(latencies_ms, 50)),
        p95_ms=float(np.percentile(latencies_ms, 95)),
        num_requests=len(latencies_ms),
    )
