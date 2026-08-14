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
  5. Optionally score `semantic@en` (VI->EN sidecar, `query_translation_path`),
     `semantic_live_en` (real production QueryTranslator wired into a second
     SearchService, `translate_semantic_live`), and/or `clip` (P1.4 mandatory
     baseline -- CLIP ViT-B/32 in its OWN Milvus collection, requires running
     `scripts/embed_clip_baseline.py` first, `score_clip_baseline`) -- all off
     by default.
  6. Write `reports/<timestamp>_<git_sha>/{metrics.json,report.md,
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
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import yaml

from app.config import Settings
from app.encoders.beit3 import BEiT3Encoder
from app.services.query_translator import LocalNmtTranslator
from app.services.search_service import SearchService
from app.services.trake_service import TrakeService
from app.stores.es_client import EsStore
from app.stores.milvus_client import MilvusStore
from app.stores.mongo_client import MongoStore
from datasets.aic2026 import AIC2026Adapter
from datasets.base import GroundTruthSegment, batch_of
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
from eval.query_translation import build_query_transform, load_translations
from eval.trake_run import TrakeGroupMetrics, render_trake_report_md, run_trake_eval

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_BEIT3_CONFIG_PATH = _REPO_ROOT / "configs" / "model" / "beit3.yaml"
_CLIP_CONFIG_PATH = _REPO_ROOT / "configs" / "model" / "clip.yaml"
_QUERY_TRANSLATOR_CONFIG_PATH = _REPO_ROOT / "configs" / "model" / "query_translator.yaml"
_MILVUS_CONFIG_PATH = _REPO_ROOT / "configs" / "stores" / "milvus.yaml"
_ES_OCR_INDEX_NAME = "ocr_keyframes"
_NDCG_K = 10  # PRD §6.3: nDCG@10 specifically
_AUTHOR_BATCH_BREAKDOWN_K = 20  # PRD §6.3: "Recall@20 tách theo author và batch"
_LIVE_TRANSLATED_CHANNEL = "semantic_live_en"


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
    # Optional VI->EN query sidecar (eval/query_translation.py). When set, each
    # mode in `query_translation_modes` is ALSO scored with English queries and
    # reported as a separate `<mode>@en` channel -- the Vietnamese channels
    # still run unchanged, so the two sit side by side in one report. Empty
    # string = feature off, no extra channels, no behaviour change.
    query_translation_path: str = ""
    # `ocr` is deliberately NOT translated by default: the indexed OCR text is
    # Vietnamese, so an English query there would measure translation loss
    # rather than anything about the retrieval channel.
    query_translation_modes: tuple[str, ...] = ("semantic",)
    # When True, scores an EXTRA `semantic_live_en` channel through a SECOND
    # SearchService wired with the real production QueryTranslator
    # (configs/model/query_translator.yaml) -- the exact code path a live
    # user query goes through (SearchService._search_semantic translating
    # internally). This is stronger evidence than `semantic@en` (which
    # translates via a precomputed sidecar OUTSIDE SearchService entirely):
    # it exercises the actual DI wiring app/main.py uses in production.
    # False by default -- loads a second translation model, only needed when
    # deliberately verifying the production path (plan §P0.9).
    translate_semantic_live: bool = False
    # P1.4 mandatory baseline (PRD §6.3): scores an EXTRA `clip` channel
    # through a SECOND SearchService wired with ClipEncoder (CLIP ViT-B/32,
    # OpenAI weights) against its OWN Milvus collection
    # (keyframes_clip_vit_b32_openai). False by default: that collection only
    # exists after `python scripts/embed_clip_baseline.py` has been run
    # (CLAUDE.md §10 -- resource-heavy, prepared for the project owner to run
    # themselves) -- flip to True once it has, otherwise this channel is
    # silently skipped rather than erroring on a missing collection.
    score_clip_baseline: bool = False
    # P2.3 (PRD §6.4): scores an EXTRA TRAKE/DANTE section appended to the
    # SAME report -- one N-consecutive-shot sequence per N in
    # trake_sequence_lengths is one query, run through TrakeService and
    # scored with eval.metrics.trake (Sequence Exact-Match, Kendall's tau,
    # etc). False by default: it drives TrakeService's own per-video-in-
    # corpus DANTE loop (architecture.md §3.5), a separate cost from the
    # KIS/OCR channels above.
    score_trake: bool = False
    trake_sequence_lengths: tuple[int, ...] = (3, 5, 7)
    trake_top_k: int = 10
    # None = use configs/search/dante.yaml's configured lambda_penalty
    # (same "None means default" contract as TrakeService.search()).
    trake_lambda_penalty: float | None = None


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
    # Defaults to `caption` so existing callers keep their meaning; differs
    # only on channels whose query text was transformed (e.g. `semantic@en`).
    query_sent: str = ""


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
    searcher: ChannelSearcher,
    queries: Sequence[GroundTruthSegment],
    top_k: int,
    query_transform: Callable[[GroundTruthSegment], str] | None = None,
) -> list[tuple[GroundTruthSegment, int | None, int | None]]:
    """Runs every query through `searcher.search()`, returns
    `(ground_truth, rank_video, rank_segment)` triples.

    `query_transform` rewrites the TEXT sent to the searcher (e.g. the VI->EN
    sidecar) while the ground truth used to score the hits stays untouched --
    that asymmetry is the whole point: it isolates the query side without
    editing a single label (CLAUDE.md §8). Default is the caption verbatim.
    """
    results: list[tuple[GroundTruthSegment, int | None, int | None]] = []
    for gt in queries:
        query_text = gt.caption if query_transform is None else query_transform(gt)
        raw_hits = await searcher.search(query_text, top_k)
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
    ranks_with_batch = [(r[2], batch_of(r[0].video_id)) for r in scored]
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
        "| channel | level | AIC Final Score | MRR | mAP | nDCG@10 | Rank-in-Grid | found/total |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for m in channel_metrics:
        lines.append(
            f"| {m.channel} | {m.level} | {m.aic_final_score:.4f} | {m.mrr:.4f} | "
            f"{m.map:.4f} | {m.ndcg_at_10:.4f} | "
            f"{'n/a' if m.median_rank is None else f'{m.median_rank:.1f}'} | "
            f"{m.num_found}/{m.num_queries} |"
        )
    lines += [
        "",
        "`Rank-in-Grid` (PRD §6.6) is the position of the correct hit in the "
        "grid the operator has to scroll to -- the `Rank-in-Grid` column "
        "above is the MEDIAN of that value across the eval set (same "
        "computation as `median_rank`, just named per PRD §6.6); per-query "
        "values are in `per_query.csv`'s `rank_video`/`rank_segment` columns.",
    ]

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
                "query_sent",
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
                    r.query_sent,
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
    trake_groups: Sequence[TrakeGroupMetrics] | None = None,
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
        # None (not "trake" key omitted) when score_trake was off -- a
        # reader of metrics.json can tell "not run" from "ran, empty" the
        # same way leakage_report already distinguishes the two.
        "trake": [asdict(g) for g in trake_groups] if trake_groups is not None else None,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics_doc, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report_md = _render_report_md(
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
    )
    if trake_groups is not None:
        # Appended to the SAME report.md a KIS/OCR run already writes (plan
        # §P2.3) -- one report per eval/run.py invocation, not a second file
        # that could drift out of sync with which git_sha/eval_set it came
        # from.
        report_md += "\n" + render_trake_report_md(trake_groups)
    (output_dir / "report.md").write_text(report_md, encoding="utf-8")
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
    raw["query_translation_modes"] = tuple(
        raw.get("query_translation_modes", EvalRunConfig().query_translation_modes)
    )
    raw["trake_sequence_lengths"] = tuple(
        raw.get("trake_sequence_lengths", EvalRunConfig().trake_sequence_lengths)
    )
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


def _prepare_ground_truth(
    config: EvalRunConfig, adapter: AIC2026Adapter
) -> tuple[list[GroundTruthSegment], frozenset[str]]:
    ground_truth_raw = adapter.load_ground_truth(config.split)
    available = load_pilot_available(_REPO_ROOT / config.pilot_available_path)
    ground_truth = filter_ground_truth_to_available(ground_truth_raw, available)
    logger.info(
        "Loaded %d ground-truth record(s), %d after filtering to %d available video_id(s)",
        len(ground_truth_raw),
        len(ground_truth),
        len(available),
    )
    return ground_truth, available


def _build_stores(
    settings: Settings, encoder: BEiT3Encoder
) -> tuple[MongoStore, MilvusStore, EsStore, SearchService]:
    mongo = MongoStore(uri=settings.mongo_uri)
    milvus = _build_milvus_store(settings, encoder)
    es = EsStore(
        url=settings.elasticsearch_url,
        index_name=_ES_OCR_INDEX_NAME,
        api_key=settings.elastic_api_key.get_secret_value() if settings.elastic_api_key else None,
    )
    search_service = SearchService(
        encoder=encoder,
        milvus_store=milvus,
        es_store=es,
        mongo_store=mongo,
    )
    return mongo, milvus, es, search_service


def _build_query_translator() -> LocalNmtTranslator:
    """Reads the SAME model config app/main.py uses for the production
    SearchService, so this diagnostic channel and a live query are
    translated identically."""
    config = _load_yaml(_QUERY_TRANSLATOR_CONFIG_PATH)
    return LocalNmtTranslator(
        model_id=config["model_id"],
        device=config["device"],
        source_prefix=config["source_prefix"],
        target_prefix=config["target_prefix"],
        max_new_tokens=config["max_new_tokens"],
    )


def _build_live_translated_search_service(
    encoder: BEiT3Encoder, milvus: MilvusStore, es: EsStore, mongo: MongoStore
) -> SearchService:
    """A SECOND SearchService sharing the same stores/encoder as the main one,
    but wired with the real production QueryTranslator -- exercises the exact
    code path a live user query goes through (SearchService._search_semantic
    translating internally), unlike `semantic@en` which translates via a
    precomputed sidecar entirely outside SearchService."""
    return SearchService(
        encoder=encoder,
        milvus_store=milvus,
        es_store=es,
        mongo_store=mongo,
        query_translator=_build_query_translator(),
    )


def _build_clip_search_service(settings: Settings, es: EsStore, mongo: MongoStore) -> SearchService:
    """P1.4 baseline (PRD §6.3): a THIRD SearchService, own encoder (CLIP
    ViT-B/32) + own Milvus collection (`scripts/embed_clip_baseline.py`
    writes `keyframes_clip_vit_b32_openai`, never mixed with BEiT-3's
    collection -- CLAUDE.md §3). Shares `es`/`mongo` with the main
    SearchService since OCR text and keyframe metadata don't depend on which
    vision encoder produced the image vectors.

    `open_clip` is only imported here, not at module level -- routine
    semantic/ocr reports must not require the `baseline-clip` extra to be
    installed (`score_clip_baseline: false` is the default precisely so this
    function is never called unless the project owner opted in).

    Wired WITH the production QueryTranslator, same as
    `_build_live_translated_search_service`: CLIP's text tower is
    English-trained same as BEiT-3's, so scoring it on raw Vietnamese
    captions would measure the VI->EN gap again, not CLIP's visual quality --
    the whole point of this baseline is a fair image-encoder comparison
    against the ALREADY-translated `semantic_live_en` number.
    """
    from app.encoders.clip_baseline import ClipEncoder

    config = _load_yaml(_CLIP_CONFIG_PATH)
    encoder = ClipEncoder(
        model_name=config["model_name"],
        pretrained_tag=config["pretrained_tag"],
        device=config["device"],
        batch_size=config["batch_size"],
        model_id=config["model_id"],
        preprocess_workers=config["preprocess_workers"],
    )
    milvus_config = _load_yaml(_MILVUS_CONFIG_PATH)
    milvus = MilvusStore(
        uri=settings.milvus_uri,
        token=settings.milvus_token.get_secret_value(),
        model_id=encoder.model_id,
        embedding_dim=encoder.embedding_dim,
        hnsw_m=milvus_config["hnsw_m"],
        hnsw_ef_construction=milvus_config["hnsw_ef_construction"],
        search_ef=milvus_config["search_ef"],
    )
    return SearchService(
        encoder=encoder,
        milvus_store=milvus,
        es_store=es,
        mongo_store=mongo,
        query_translator=_build_query_translator(),
    )


class _ResultAccumulator:
    """Collects channel metrics + per-query rows across every channel/baseline
    run in one `eval.run` pass -- replaces a closure over `_main()`'s locals
    so each channel run stays a single call, not an inline nested function."""

    def __init__(
        self,
        ground_truth: Sequence[GroundTruthSegment],
        top_k: int,
        top_ks: Sequence[int],
        author_map: Mapping[str, str],
        breakdown_k: int,
    ) -> None:
        self._ground_truth = ground_truth
        self._top_k = top_k
        self._top_ks = top_ks
        self._author_map = author_map
        self._breakdown_k = breakdown_k
        self.channel_metrics: list[ChannelMetrics] = []
        self.author_breakdown: dict[str, dict[str, float]] = {}
        self.batch_breakdown: dict[str, dict[str, float]] = {}
        self.per_query: list[PerQueryResult] = []

    async def run_and_collect(
        self,
        channel_name: str,
        searcher: ChannelSearcher,
        query_transform: Callable[[GroundTruthSegment], str] | None = None,
    ) -> None:
        scored = await score_channel(searcher, self._ground_truth, self._top_k, query_transform)
        video_m, segment_m, by_author, by_batch = aggregate_channel(
            channel_name, scored, self._top_ks, self._author_map, self._breakdown_k
        )
        self.channel_metrics.extend([video_m, segment_m])
        self.author_breakdown[channel_name] = by_author
        self.batch_breakdown[channel_name] = by_batch
        for gt, rank_video, rank_segment in scored:
            self.per_query.append(
                PerQueryResult(
                    video_id=gt.video_id,
                    order_index=gt.order_index,
                    caption=gt.caption,
                    batch=batch_of(gt.video_id),
                    author=self._author_map.get(gt.video_id, "unknown"),
                    channel=channel_name,
                    rank_video=rank_video,
                    rank_segment=rank_segment,
                    # The text actually sent to the searcher. Without this a
                    # `semantic@en` row is indistinguishable from a `semantic`
                    # row in the CSV, and the run stops being auditable.
                    query_sent=gt.caption if query_transform is None else query_transform(gt),
                )
            )


_TRANSLATED_CHANNEL_SUFFIX = "@en"


async def _collect_translated_channels(
    config: EvalRunConfig,
    accumulator: _ResultAccumulator,
    search_service: SearchService,
    ground_truth: Sequence[GroundTruthSegment],
) -> None:
    """Scores `<mode>@en` for every mode in `query_translation_modes`.

    No-op when `query_translation_path` is empty, so an unconfigured run
    behaves exactly as before. When it IS configured, a bad sidecar raises out
    of here rather than being skipped with a warning: a report silently
    missing its English channel is the one outcome that would waste the whole
    diagnostic (CLAUDE.md §7).
    """
    if not config.query_translation_path:
        return

    translations = load_translations(_REPO_ROOT / config.query_translation_path)
    transform = build_query_transform(ground_truth, translations)
    logger.info(
        "Loaded %d translated quer(ies) from %s -- scoring %s channel(s)",
        len(translations),
        config.query_translation_path,
        ", ".join(f"{m}{_TRANSLATED_CHANNEL_SUFFIX}" for m in config.query_translation_modes),
    )
    for mode in config.query_translation_modes:
        await accumulator.run_and_collect(
            f"{mode}{_TRANSLATED_CHANNEL_SUFFIX}",
            _SemanticSearcher(search_service, mode),
            transform,
        )


async def _build_leakage_report_or_none(
    ground_truth: Sequence[GroundTruthSegment],
    mongo: MongoStore,
    es: EsStore,
    stopwords: frozenset[str],
    threshold: float,
) -> LeakageReport | None:
    leakage_records = await build_leakage_records(ground_truth, mongo, es)
    if not any(r.ocr_texts for r in leakage_records):
        logger.warning(
            "build_leakage_records: no OCR text found for any ground-truth keyframe -- "
            "leakage report skipped (see report.md 'not_measured')"
        )
        return None
    return compute_leakage_report(leakage_records, stopwords=stopwords, threshold=threshold)


def _resolve_eval_set_hash(eval_set_path: Path) -> str:
    hash_path = eval_set_path.with_name(eval_set_path.name + ".sha256")
    if not hash_path.exists():
        return "unknown"
    return hash_path.read_text(encoding="utf-8").strip()


async def _run_all_channels(
    accumulator: _ResultAccumulator,
    config: EvalRunConfig,
    mongo: MongoStore,
    es: EsStore,
    search_service: SearchService,
    ground_truth: Sequence[GroundTruthSegment],
    live_translated_search_service: SearchService | None = None,
    clip_search_service: SearchService | None = None,
) -> None:
    for mode in config.modes:
        await accumulator.run_and_collect(mode, _SemanticSearcher(search_service, mode))

    await _collect_translated_channels(config, accumulator, search_service, ground_truth)

    if live_translated_search_service is not None:
        # Raw Vietnamese caption in, no query_transform -- the SECOND
        # SearchService translates internally, same as a real user query.
        await accumulator.run_and_collect(
            _LIVE_TRANSLATED_CHANNEL,
            _SemanticSearcher(live_translated_search_service, "semantic"),
        )

    if clip_search_service is not None:
        # Same raw-Vietnamese-in pattern as live_translated_search_service --
        # ClipEncoder's SearchService translates internally too (P1.4 baseline).
        await accumulator.run_and_collect(
            "clip", _SemanticSearcher(clip_search_service, "semantic")
        )

    keyframe_pool = await build_keyframe_pool(mongo)
    await accumulator.run_and_collect(
        "random", RandomBaseline(keyframe_pool, config.random_baseline_seed)
    )
    await accumulator.run_and_collect("bm25", BM25Baseline(es))


def _build_trake_service(
    encoder: BEiT3Encoder, milvus: MilvusStore, mongo: MongoStore
) -> TrakeService:
    """P2.3 (PRD §6.4): reuses the SAME encoder/Milvus/Mongo instances the
    KIS/OCR channels above already built -- TRAKE's per-video DP needs the
    identical vector space (CLAUDE.md §3), not a second encoder.

    ALWAYS wired with a QueryTranslator, independent of
    `translate_semantic_live` (that flag only controls an EXTRA diagnostic
    KIS channel) -- an untranslated TRAKE run just re-measures the known
    VI/EN mismatch (chủ dự án 2026-08-14: video_level_recall_at_k=0.0000
    across every query before this was wired in), not a useful baseline."""
    return TrakeService(
        encoder=encoder,
        milvus_store=milvus,
        mongo_store=mongo,
        query_translator=_build_query_translator(),
    )


def _write_final_report(
    config: EvalRunConfig,
    ground_truth: Sequence[GroundTruthSegment],
    available: frozenset[str],
    accumulator: _ResultAccumulator,
    leakage_report: LeakageReport | None,
    trake_groups: Sequence[TrakeGroupMetrics] | None = None,
) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    git_sha = _git_sha()
    output_dir = _REPO_ROOT / config.reports_dir / f"{timestamp}_{git_sha}"
    write_report(
        output_dir,
        git_sha=git_sha,
        eval_set_version="v1",
        eval_set_hash=_resolve_eval_set_hash(_REPO_ROOT / config.eval_set_path),
        num_video_ids=len(available),
        num_queries=len(ground_truth),
        seed=config.random_baseline_seed,
        channel_metrics=accumulator.channel_metrics,
        author_breakdown=accumulator.author_breakdown,
        batch_breakdown=accumulator.batch_breakdown,
        leakage_report=leakage_report,
        per_query=accumulator.per_query,
        trake_groups=trake_groups,
    )
    return output_dir


async def _main() -> int:
    args = _parse_args()
    config = _load_config(args.config)
    settings = Settings()
    top_k = max(config.top_ks)

    adapter = AIC2026Adapter(
        dataset_root=_REPO_ROOT / config.dataset_root,
        eval_set_path=_REPO_ROOT / config.eval_set_path,
    )
    ground_truth, available = _prepare_ground_truth(config, adapter)
    author_map = build_author_map(adapter, config.split)

    encoder = _build_encoder(args.device)
    mongo, milvus, es, search_service = _build_stores(settings, encoder)
    try:
        live_translated_search_service = (
            _build_live_translated_search_service(encoder, milvus, es, mongo)
            if config.translate_semantic_live
            else None
        )
        clip_search_service = (
            _build_clip_search_service(settings, es, mongo) if config.score_clip_baseline else None
        )

        accumulator = _ResultAccumulator(
            ground_truth, top_k, config.top_ks, author_map, _AUTHOR_BATCH_BREAKDOWN_K
        )
        await _run_all_channels(
            accumulator,
            config,
            mongo,
            es,
            search_service,
            ground_truth,
            live_translated_search_service,
            clip_search_service,
        )

        leakage_report = await _build_leakage_report_or_none(
            ground_truth,
            mongo,
            es,
            frozenset(config.vietnamese_stopwords),
            config.leakage_jaccard_threshold,
        )

        trake_groups = None
        if config.score_trake:
            # Every sequence_length is validated before any expensive
            # search runs (run_trake_eval's own fail-fast check), but this
            # try/finally is the second line of defense: ANY exception past
            # this point (not just an unextractable N) must still close
            # mongo/es (regression 2026-08-14 -- a real run left both
            # connections open after an uncaught ValueError propagated
            # straight out of _main()).
            trake_service = _build_trake_service(encoder, milvus, mongo)
            trake_groups = await run_trake_eval(
                trake_service,
                ground_truth,
                config.trake_sequence_lengths,
                config.trake_top_k,
                config.trake_lambda_penalty,
            )

        output_dir = _write_final_report(
            config, ground_truth, available, accumulator, leakage_report, trake_groups
        )
        logger.info("Wrote report to %s", output_dir)
    finally:
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
