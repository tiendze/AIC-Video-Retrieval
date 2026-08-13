"""CLI: benchmark `POST /api/search` latency against a LIVE server (plan
§P1.7, PRD §6.6) -- fills the ⚠️ p50/p95 rows for the `semantic`/`ocr` modes
with real numbers. The `DANTE`/`QUEST`/`VQA` rows of that same table stay
⚠️: those endpoints don't exist yet (P2/P3), so there is nothing to benchmark.

Measures the FULL client-observed HTTP round trip (send -> server receives
-> server responds -> client receives) via a real `httpx` request, matching
PRD §6.6's "đo từ lúc nhận request đến lúc trả response" from the operator's
actual experience -- not `eval/run.py`'s `score_channel`, which times a
narrower in-process `SearchService` call for a different purpose entirely
(retrieval-quality scoring, not latency).

Reuses `configs/eval/aic_eval_v1.json`'s captions as realistic query text
(same corpus queries `eval/run.py` scores against) -- no separate query set
to keep in sync.

Needs a live server (CLAUDE.md §10 -- project owner runs, not this agent):
  docker compose up -d
  ./.venv/Scripts/python.exe -m uvicorn app.main:app --port 8000
  # in another terminal:
  ./.venv/Scripts/python.exe scripts/measure_search_latency.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import httpx
import yaml

from eval.metrics.latency import LatencyPercentiles, compute_latency_percentiles

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_EVAL_SET_PATH = _REPO_ROOT / "configs" / "eval" / "aic_eval_v1.json"
_SEARCH_CONFIG_PATH = _REPO_ROOT / "configs" / "search" / "search.yaml"
# PRD §6.6's latency table only has real rows to fill for the two modes that
# exist today -- DANTE/QUEST/VQA are P2/P3, not built.
_MODES: tuple[str, ...] = ("semantic", "ocr")
# Discarded, not counted: the first request(s) after server startup can be
# slower for reasons unrelated to steady-state serving (OS file-cache warmup,
# connection setup) -- same reasoning as embed_clip_baseline.py's note on the
# first video of a fresh process.
_WARMUP_REQUESTS = 3


def _load_queries(eval_set_path: Path, limit: int | None) -> list[str]:
    records = json.loads(eval_set_path.read_text(encoding="utf-8"))
    captions = [record["caption"] for record in records]
    return captions if limit is None else captions[:limit]


def _benchmark_mode(
    client: httpx.Client, base_url: str, mode: str, queries: Sequence[str], top_k: int
) -> list[float]:
    """Sends one `POST /api/search` per query, sequentially (one operator
    searching at a time, not a concurrent-load test -- PRD §6.6 measures the
    experience of a single person triaging results, not throughput under
    load). Raises on a non-2xx response instead of silently including a
    server-error's latency in the percentiles (CLAUDE.md §7)."""
    latencies_ms: list[float] = []
    for query in queries:
        start = time.perf_counter()
        response = client.post(
            f"{base_url}/api/search", json={"query": query, "mode": mode, "top_k": top_k}
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        response.raise_for_status()
        latencies_ms.append(elapsed_ms)
    return latencies_ms


def _render_report(results: dict[str, LatencyPercentiles]) -> str:
    lines = ["| mode | p50 | p95 | n |", "|---|---|---|---|"]
    for mode, stats in results.items():
        lines.append(
            f"| {mode} | {stats.p50_ms:.0f} ms | {stats.p95_ms:.0f} ms | {stats.num_requests} |"
        )
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="benchmark only the first N eval captions per mode (default: all)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="overrides configs/search/search.yaml's default_top_k",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()
    config = yaml.safe_load(_SEARCH_CONFIG_PATH.read_text(encoding="utf-8"))
    top_k = args.top_k or config["default_top_k"]

    queries = _load_queries(_EVAL_SET_PATH, args.limit)
    logger.info("Loaded %d quer(ies) from %s", len(queries), _EVAL_SET_PATH)

    results: dict[str, LatencyPercentiles] = {}
    with httpx.Client(timeout=30.0) as client:
        for mode in _MODES:
            logger.info("Warming up %s (%d request(s), discarded)...", mode, _WARMUP_REQUESTS)
            _benchmark_mode(client, args.base_url, mode, queries[:_WARMUP_REQUESTS], top_k)

            logger.info("Benchmarking %s (%d request(s))...", mode, len(queries))
            latencies_ms = _benchmark_mode(client, args.base_url, mode, queries, top_k)
            stats = compute_latency_percentiles(latencies_ms)
            results[mode] = stats
            logger.info(
                "%s: p50=%.1fms p95=%.1fms (n=%d)",
                mode,
                stats.p50_ms,
                stats.p95_ms,
                stats.num_requests,
            )

    print()
    print(_render_report(results))


if __name__ == "__main__":
    main()
