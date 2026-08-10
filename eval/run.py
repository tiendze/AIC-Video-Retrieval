"""CLI: eval/run.py -- the SINGLE evaluation entry point (plan §P0.9,
CLAUDE.md §6 "eval/run.py là entry point duy nhất của evaluation"). Turns
the system from "chạy được" into "đo được": scores `SearchService`'s
`semantic`/`ocr` channels (P0.8) against the self-generated ground truth
(`aic_eval_v1.json`, P0.4') and writes a report.

Pipeline (one query = one ground-truth caption -> one shot):
  1. Load ground truth, keep only `video_id`s in `pilot_available_v1.txt`
     (CLAUDE.md §8: only videos actually ingest-"complete" count).
  2. For each channel (`semantic`, `ocr`) and each baseline (`random`,
     `bm25`), run every query's caption through `search(caption, top_k)` and
     apply the point-in-segment hit rule (PRD §6.2) at BOTH video-level and
     segment-level -- a video-level-high/segment-level-low split diagnoses a
     keyframe-sampling bug, not an embedding bug.
  3. Aggregate Recall@{1,5,20,50,100} / MRR / mAP / nDCG@10 / median rank /
     AIC Final Score, plus Recall@20 broken down by `author` and `batch`.
  4. Run the label-leakage test (caption <-> ocr_text overlap,
     eval/metrics/leakage.py) -- required by plan §P0.9, not optional.
  5. Write `reports/<timestamp>_<git_sha>/{metrics.json,report.md,
     per_query.csv}`.

Needs live Milvus/Elasticsearch/MongoDB (docker compose) and a loaded
BEiT-3 checkpoint (GPU) -- per CLAUDE.md §10, prepared here for the project
owner to run themselves, not run automatically by this agent:

  docker compose up -d
  python -m eval.run --config configs/eval/aic_pilot.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import yaml

from app.config import Settings
from app.encoders.beit3 import BEiT3Encoder
from app.services.search_service import SearchService
from app.stores.es_client import EsStore
from app.stores.milvus_client import MilvusStore
from app.stores.mongo_client import MongoStore
from datasets.aic2026 import AIC2026Adapter
from datasets.base import GroundTruthSegment
from eval.baselines.bm25 import BM25Baseline
from eval.baselines.random import KeyframeRecord, RandomBaseline
from eval.metrics.leakage import LeakageRecord, LeakageReport, compute_leakage_report
from eval.metrics.retrieval import (
    GroundTruthQuery,
    RetrievedHit,
    aic_final_score,
    first_hit_rank,
    mean_average_precision,
    mean_reciprocal_rank,
    median_rank,
    ndcg_at_k,
    recall_at_k,
    recall_at_k_by_group,
)

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_BEIT3_CONFIG_PATH = _REPO_ROOT / "configs" / "model" / "beit3.yaml"
_MILVUS_CONFIG_PATH = _REPO_ROOT / "configs" / "stores" / "milvus.yaml"
_ES_OCR_INDEX_NAME = "ocr_keyframes"
_NDCG_K = 10  # PRD §6.3: nDCG@10 specifically
_AUTHOR_BATCH_BREAKDOWN_K = 20  # PRD §6.3: "Recall@20 tách theo author và batch"


@dataclass(frozen=True)
class EvalRunConfig:
    """Mirrors `configs/eval/aic_pilot.yaml` -- see that file for the
    rationale behind each default."""

    split: str = "all"
    dataset_root: str = "data"
    eval_set_path: str = "configs/eval/aic_eval_v1.json"
    pilot_available_path: str = "configs/eval/pilot_available_v1.txt"
    top_ks: tuple[int, ...] = (1, 5, 20, 50, 100)
    modes: tuple[str, ...] = ("semantic", "ocr")
    random_baseline_seed: int = 20260810
    leakage_jaccard_threshold: float = 0.5
    vietnamese_stopwords: tuple[str, ...] = ()
    reports_dir: str = "reports"


@dataclass(frozen=True)
class ChannelMetrics:
    channel: str
    level: str  # "video" | "segment"
    recall_at_k: dict[int, float]
    mrr: float
    map: float
    ndcg_at_10: float
    median_rank: float | None
    aic_final_score: float
    num_queries: int
    num_found: int


@dataclass(frozen=True)
class PerQueryResult:
    video_id: str
    order_index: int
    caption: str
    batch: str
    author: str
    channel: str
    rank_video: int | None
    rank_segment: int | None


class ChannelSearcher(Protocol):
    async def search(self, query: str, top_k: int) -> Sequence[Any]: ...


# --------------------------------------------------------------------------
# Ground truth loading / filtering
# --------------------------------------------------------------------------


def load_pilot_available(path: Path) -> frozenset[str]:
    """The frozen denominator every metric report must cite (CLAUDE.md §8) --
    only `video_id`s that reached at least "embedded" in `scripts/run_ingest.py`.
    Raises, does not fall back to an empty set: scoring against un-ingested
    videos silently deflates every Recall for reasons unrelated to retrieval
    quality (CLAUDE.md §7 "thiếu dữ liệu phải fail rõ ràng")."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run scripts/run_ingest.py first (it writes this "
            "denominator list); eval/run.py refuses to score against un-ingested videos."
        )
    lines = path.read_text(encoding="utf-8").splitlines()
    return frozenset(line.strip() for line in lines if line.strip())


def filter_ground_truth_to_available(
    segments: Sequence[GroundTruthSegment], available_video_ids: frozenset[str]
) -> list[GroundTruthSegment]:
    return [s for s in segments if s.video_id in available_video_ids]


def _batch_of(video_id: str) -> str:
    return video_id.split("_V", 1)[0]


def build_author_map(adapter: AIC2026Adapter, split: str) -> dict[str, str]:
    return {v.video_id: (v.category or "unknown") for v in adapter.iter_videos(split)}


# --------------------------------------------------------------------------
# Scoring -- the shared core loop for real channels AND baselines, since
# SearchService / RandomBaseline / BM25Baseline all expose the same async
# `search(query, top_k) -> hits` shape.
# --------------------------------------------------------------------------


def _hit_to_retrieved(hit: Any) -> RetrievedHit:
    """Normalizes the three hit shapes this module drives (SearchHit
    dataclass, KeyframeRecord dataclass, raw ES dict) to the one shape
    `eval.metrics.retrieval` needs."""
    if isinstance(hit, dict):
        return RetrievedHit(video_id=hit["video_id"], timestamp=hit["timestamp"])
    return RetrievedHit(video_id=hit.video_id, timestamp=hit.timestamp)


async def score_channel(
    searcher: ChannelSearcher, queries: Sequence[GroundTruthSegment], top_k: int
) -> list[tuple[GroundTruthSegment, int | None, int | None]]:
    """Runs every query's caption through `searcher.search()`, returns
    `(ground_truth, rank_video, rank_segment)` triples."""
    results: list[tuple[GroundTruthSegment, int | None, int | None]] = []
    for gt in queries:
        raw_hits = await searcher.search(gt.caption, top_k)
        hits = [_hit_to_retrieved(h) for h in raw_hits]
        gt_query = GroundTruthQuery(
            video_id=gt.video_id, start_sec=gt.start_sec, end_sec=gt.end_sec
        )
        rank_video = first_hit_rank(hits, gt_query, level="video")
        rank_segment = first_hit_rank(hits, gt_query, level="segment")
        results.append((gt, rank_video, rank_segment))
    return results


def compute_channel_metrics(
    channel: str, level: str, ranks: Sequence[int | None], top_ks: Sequence[int]
) -> ChannelMetrics:
    return ChannelMetrics(
        channel=channel,
        level=level,
        recall_at_k={k: recall_at_k(ranks, k) for k in top_ks},
        mrr=mean_reciprocal_rank(ranks),
        map=mean_average_precision(ranks),
        ndcg_at_10=ndcg_at_k(ranks, _NDCG_K),
        median_rank=median_rank(ranks),
        aic_final_score=aic_final_score(ranks, tuple(top_ks)),
        num_queries=len(ranks),
        num_found=sum(1 for r in ranks if r is not None),
    )


def aggregate_channel(
    channel: str,
    scored: Sequence[tuple[GroundTruthSegment, int | None, int | None]],
    top_ks: Sequence[int],
    author_map: Mapping[str, str],
    breakdown_k: int,
) -> tuple[ChannelMetrics, ChannelMetrics, dict[str, float], dict[str, float]]:
    """Returns `(video_level, segment_level, recall_by_author, recall_by_batch)`.
    The author/batch breakdown is computed at the SEGMENT level (the
    stricter, more diagnostic level) and at `breakdown_k` (PRD §6.3's
    Recall@20 default)."""
    video_ranks = [r[1] for r in scored]
    segment_ranks = [r[2] for r in scored]
    video_metrics = compute_channel_metrics(channel, "video", video_ranks, top_ks)
    segment_metrics = compute_channel_metrics(channel, "segment", segment_ranks, top_ks)

    ranks_with_author = [(r[2], author_map.get(r[0].video_id, "unknown")) for r in scored]
    ranks_with_batch = [(r[2], _batch_of(r[0].video_id)) for r in scored]
    by_author = recall_at_k_by_group(ranks_with_author, breakdown_k)
    by_batch = recall_at_k_by_group(ranks_with_batch, breakdown_k)
    return video_metrics, segment_metrics, by_author, by_batch


# --------------------------------------------------------------------------
# Label-leakage test (caption <-> ocr_text) -- required P0.9 output
# --------------------------------------------------------------------------


def keyframes_in_window(
    keyframe_docs: Sequence[Mapping[str, Any]], start_sec: float, end_sec: float
) -> list[str]:
    """`keyframe_id`s whose `timestamp` falls in `[start_sec, end_sec]` --
    the point-in-segment rule applied to one video's own keyframe list."""
    return [doc["keyframe_id"] for doc in keyframe_docs if start_sec <= doc["timestamp"] <= end_sec]


class _KeyframeSource(Protocol):
    async def get_keyframes_for_video(self, video_id: str) -> list[dict[str, Any]]: ...


class _OcrTextSource(Protocol):
    async def get_by_ids(self, keyframe_ids: Sequence[str]) -> dict[str, str]: ...


async def build_leakage_records(
    ground_truth: Sequence[GroundTruthSegment],
    mongo_store: _KeyframeSource,
    es_store: _OcrTextSource,
) -> list[LeakageRecord]:
    """Joins each ground-truth query's segment window to the OCR text of the
    keyframes inside it. Caches `get_keyframes_for_video` per `video_id` --
    KIS queries from the same video (or a TRAKE sequence) would otherwise
    re-fetch the same keyframe list once per query."""
    records: list[LeakageRecord] = []
    keyframes_cache: dict[str, list[dict[str, Any]]] = {}
    for gt in ground_truth:
        if gt.video_id not in keyframes_cache:
            keyframes_cache[gt.video_id] = await mongo_store.get_keyframes_for_video(gt.video_id)
        keyframe_ids = keyframes_in_window(keyframes_cache[gt.video_id], gt.start_sec, gt.end_sec)
        ocr_by_id = await es_store.get_by_ids(keyframe_ids)
        records.append(
            LeakageRecord(
                video_id=gt.video_id,
                order_index=gt.order_index,
                caption=gt.caption,
                ocr_texts=list(ocr_by_id.values()),
            )
        )
    return records


# --------------------------------------------------------------------------
# Random baseline pool
# --------------------------------------------------------------------------


class _ScrollableMongoStore(Protocol):
    def scroll_all_keyframes(self): ...


async def build_keyframe_pool(mongo_store: _ScrollableMongoStore) -> list[KeyframeRecord]:
    pool: list[KeyframeRecord] = []
    async for doc in mongo_store.scroll_all_keyframes():
        pool.append(
            KeyframeRecord(
                keyframe_id=doc["keyframe_id"], video_id=doc["video_id"], timestamp=doc["timestamp"]
            )
        )
    return pool


# --------------------------------------------------------------------------
# Report writing
# --------------------------------------------------------------------------


def _channel_metrics_to_dict(m: ChannelMetrics) -> dict[str, Any]:
    return {
        "channel": m.channel,
        "level": m.level,
        "recall_at_k": {str(k): v for k, v in m.recall_at_k.items()},
        "mrr": m.mrr,
        "map": m.map,
        "ndcg_at_10": m.ndcg_at_10,
        "median_rank": m.median_rank,
        "aic_final_score": m.aic_final_score,
        "num_queries": m.num_queries,
        "num_found": m.num_found,
    }


def _render_report_md(
    *,
    git_sha: str,
    eval_set_version: str,
    eval_set_hash: str,
    num_video_ids: int,
    num_queries: int,
    seed: int,
    channel_metrics: Sequence[ChannelMetrics],
    author_breakdown: Mapping[str, Mapping[str, float]],
    batch_breakdown: Mapping[str, Mapping[str, float]],
    leakage_report: LeakageReport | None,
) -> str:
    lines = [
        "# AIC Video Retrieval -- eval/run.py report",
        "",
        f"- git_sha: {git_sha}",
        f"- eval_set_version: {eval_set_version} (sha256={eval_set_hash})",
        f"- num_video_ids (pilot_available): {num_video_ids}",
        f"- num_queries: {num_queries}",
        f"- seed: {seed}",
        "",
        "## Channel metrics",
        "",
        "| channel | level | AIC Final Score | MRR | mAP | nDCG@10 | median_rank | found/total |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for m in channel_metrics:
        lines.append(
            f"| {m.channel} | {m.level} | {m.aic_final_score:.4f} | {m.mrr:.4f} | "
            f"{m.map:.4f} | {m.ndcg_at_10:.4f} | "
            f"{'n/a' if m.median_rank is None else f'{m.median_rank:.1f}'} | "
            f"{m.num_found}/{m.num_queries} |"
        )

    lines += ["", "## Recall@20 by author / batch (segment-level)", ""]
    for channel, by_author in author_breakdown.items():
        lines.append(f"- **{channel}** by author: {dict(sorted(by_author.items()))}")
    for channel, by_batch in batch_breakdown.items():
        lines.append(f"- **{channel}** by batch: {dict(sorted(by_batch.items()))}")

    lines += ["", "## Label-leakage test (caption <-> ocr_text)", ""]
    if leakage_report is None:
        lines.append("not_measured (no OCR text available for the evaluated ground truth)")
    else:
        lines += [
            f"- total: {leakage_report.total}",
            f"- mean_overlap: {leakage_report.mean_overlap:.4f}",
            f"- median_overlap: {leakage_report.median_overlap:.4f}",
            f"- high_overlap_rate (>= {leakage_report.threshold}): "
            f"{leakage_report.high_overlap_rate:.4f} "
            f"({leakage_report.high_overlap_count}/{leakage_report.total})",
        ]
    return "\n".join(lines) + "\n"


def _write_per_query_csv(path: Path, per_query: Sequence[PerQueryResult]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "video_id",
                "order_index",
                "caption",
                "batch",
                "author",
                "channel",
                "rank_video",
                "rank_segment",
            ]
        )
        for r in per_query:
            writer.writerow(
                [
                    r.video_id,
                    r.order_index,
                    r.caption,
                    r.batch,
                    r.author,
                    r.channel,
                    r.rank_video,
                    r.rank_segment,
                ]
            )


def write_report(
    output_dir: Path,
    *,
    git_sha: str,
    eval_set_version: str,
    eval_set_hash: str,
    num_video_ids: int,
    num_queries: int,
    seed: int,
    channel_metrics: Sequence[ChannelMetrics],
    author_breakdown: Mapping[str, Mapping[str, float]],
    batch_breakdown: Mapping[str, Mapping[str, float]],
    leakage_report: LeakageReport | None,
    per_query: Sequence[PerQueryResult],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_doc = {
        "git_sha": git_sha,
        "eval_set_version": eval_set_version,
        "eval_set_hash": eval_set_hash,
        "num_video_ids": num_video_ids,
        "num_queries": num_queries,
        "seed": seed,
        "channels": [_channel_metrics_to_dict(m) for m in channel_metrics],
        "author_breakdown": {k: dict(v) for k, v in author_breakdown.items()},
        "batch_breakdown": {k: dict(v) for k, v in batch_breakdown.items()},
        "leakage_report": asdict(leakage_report) if leakage_report is not None else None,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics_doc, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "report.md").write_text(
        _render_report_md(
            git_sha=git_sha,
            eval_set_version=eval_set_version,
            eval_set_hash=eval_set_hash,
            num_video_ids=num_video_ids,
            num_queries=num_queries,
            seed=seed,
            channel_metrics=channel_metrics,
            author_breakdown=author_breakdown,
            batch_breakdown=batch_breakdown,
            leakage_report=leakage_report,
        ),
        encoding="utf-8",
    )
    _write_per_query_csv(output_dir / "per_query.csv", per_query)


# --------------------------------------------------------------------------
# CLI wiring -- mirrors scripts/run_ingest.py's own config-loading pattern
# --------------------------------------------------------------------------


def _git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, OSError, subprocess.TimeoutExpired):
        return "unknown"


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_config(path: Path) -> EvalRunConfig:
    raw = _load_yaml(path)
    raw["top_ks"] = tuple(raw.get("top_ks", EvalRunConfig().top_ks))
    raw["modes"] = tuple(raw.get("modes", EvalRunConfig().modes))
    raw["vietnamese_stopwords"] = tuple(raw.get("vietnamese_stopwords", ()))
    return EvalRunConfig(**raw)


def _build_encoder(device: str) -> BEiT3Encoder:
    config = _load_yaml(_BEIT3_CONFIG_PATH)
    return BEiT3Encoder(
        checkpoint_path=str(_REPO_ROOT / config["checkpoint_path"]),
        sentencepiece_model_path=str(_REPO_ROOT / config["sentencepiece_model_path"]),
        device=device,
        batch_size=config["batch_size"],
        model_id=config["model_id"],
    )


def _build_milvus_store(settings: Settings, encoder: BEiT3Encoder) -> MilvusStore:
    config = _load_yaml(_MILVUS_CONFIG_PATH)
    return MilvusStore(
        uri=settings.milvus_uri,
        token=settings.milvus_token.get_secret_value(),
        model_id=encoder.model_id,
        embedding_dim=encoder.embedding_dim,
        hnsw_m=config["hnsw_m"],
        hnsw_ef_construction=config["hnsw_ef_construction"],
        search_ef=config["search_ef"],
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config", type=Path, default=_REPO_ROOT / "configs" / "eval" / "aic_pilot.yaml"
    )
    parser.add_argument("--device", default="cuda", help="'cuda' or 'cpu'")
    return parser.parse_args()


class _SemanticSearcher:
    """Adapts `SearchService.search(query, mode, top_k)` to the shared
    `search(query, top_k)` shape `score_channel` drives."""

    def __init__(self, search_service: SearchService, mode: str) -> None:
        self._search_service = search_service
        self._mode = mode

    async def search(self, query: str, top_k: int):
        return await self._search_service.search(query, mode=self._mode, top_k=top_k)


async def _main() -> int:
    args = _parse_args()
    config = _load_config(args.config)
    settings = Settings()
    top_k = max(config.top_ks)

    adapter = AIC2026Adapter(
        dataset_root=_REPO_ROOT / config.dataset_root,
        eval_set_path=_REPO_ROOT / config.eval_set_path,
    )
    ground_truth_raw = adapter.load_ground_truth(config.split)
    available = load_pilot_available(_REPO_ROOT / config.pilot_available_path)
    ground_truth = filter_ground_truth_to_available(ground_truth_raw, available)
    logger.info(
        "Loaded %d ground-truth record(s), %d after filtering to %d available video_id(s)",
        len(ground_truth_raw),
        len(ground_truth),
        len(available),
    )
    author_map = build_author_map(adapter, config.split)

    encoder = _build_encoder(args.device)
    mongo = MongoStore(uri=settings.mongo_uri)
    milvus = _build_milvus_store(settings, encoder)
    es = EsStore(
        url=settings.elasticsearch_url,
        index_name=_ES_OCR_INDEX_NAME,
        api_key=settings.elastic_api_key.get_secret_value() if settings.elastic_api_key else None,
    )
    search_service = SearchService(
        encoder=encoder, milvus_store=milvus, es_store=es, mongo_store=mongo
    )

    channel_metrics: list[ChannelMetrics] = []
    author_breakdown: dict[str, dict[str, float]] = {}
    batch_breakdown: dict[str, dict[str, float]] = {}
    per_query: list[PerQueryResult] = []

    async def _run_and_collect(channel_name: str, searcher: ChannelSearcher) -> None:
        scored = await score_channel(searcher, ground_truth, top_k)
        video_m, segment_m, by_author, by_batch = aggregate_channel(
            channel_name, scored, config.top_ks, author_map, _AUTHOR_BATCH_BREAKDOWN_K
        )
        channel_metrics.extend([video_m, segment_m])
        author_breakdown[channel_name] = by_author
        batch_breakdown[channel_name] = by_batch
        for gt, rank_video, rank_segment in scored:
            per_query.append(
                PerQueryResult(
                    video_id=gt.video_id,
                    order_index=gt.order_index,
                    caption=gt.caption,
                    batch=_batch_of(gt.video_id),
                    author=author_map.get(gt.video_id, "unknown"),
                    channel=channel_name,
                    rank_video=rank_video,
                    rank_segment=rank_segment,
                )
            )

    for mode in config.modes:
        await _run_and_collect(mode, _SemanticSearcher(search_service, mode))

    keyframe_pool = await build_keyframe_pool(mongo)
    await _run_and_collect("random", RandomBaseline(keyframe_pool, config.random_baseline_seed))
    await _run_and_collect("bm25", BM25Baseline(es))

    leakage_report: LeakageReport | None = None
    leakage_records = await build_leakage_records(ground_truth, mongo, es)
    stopwords = frozenset(config.vietnamese_stopwords)
    if any(r.ocr_texts for r in leakage_records):
        leakage_report = compute_leakage_report(
            leakage_records, stopwords=stopwords, threshold=config.leakage_jaccard_threshold
        )
    else:
        logger.warning(
            "build_leakage_records: no OCR text found for any ground-truth keyframe -- "
            "leakage report skipped (see report.md 'not_measured')"
        )

    eval_set_hash_path = (_REPO_ROOT / config.eval_set_path).with_name(
        (_REPO_ROOT / config.eval_set_path).name + ".sha256"
    )
    eval_set_hash = (
        eval_set_hash_path.read_text(encoding="utf-8").strip()
        if eval_set_hash_path.exists()
        else "unknown"
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    git_sha = _git_sha()
    output_dir = _REPO_ROOT / config.reports_dir / f"{timestamp}_{git_sha}"
    write_report(
        output_dir,
        git_sha=git_sha,
        eval_set_version="v1",
        eval_set_hash=eval_set_hash,
        num_video_ids=len(available),
        num_queries=len(ground_truth),
        seed=config.random_baseline_seed,
        channel_metrics=channel_metrics,
        author_breakdown=author_breakdown,
        batch_breakdown=batch_breakdown,
        leakage_report=leakage_report,
        per_query=per_query,
    )
    logger.info("Wrote report to %s", output_dir)

    await mongo.close()
    await es.close()
    return 0


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    sys.exit(asyncio.run(_main()))


if __name__ == "__main__":
    main()
