"""CLI: transcribe the AIC 2026 dataset's spoken dialogue (P0.11a TEXT/ASR
branch) and index it to Elasticsearch at segment granularity.

Reads audio straight from each video's `.mp4` via faster-whisper (PyAV,
bundled) -- no Mongo/Milvus keyframe data needed, so this can run whether
or not `scripts/run_ingest.py` has finished for a given video (P0.11 is
off the P0 critical path; it only competes for GPU, not for ingest data).

Per video: transcribe -> drop hallucinated segments (silence/noise/short
filler, thresholds in configs/ingest/asr.yaml) -> bulk-index the rest to ES
-> record `asr_runs.asr_status` in MongoDB, keyed by `video_id`. A video
already at `asr_status="done"` is skipped on a re-run (idempotent, mirrors
scripts/run_ingest.py's resume behavior).

WARNING: GPU-heavy (large-v3 Whisper decode) -- per CLAUDE.md section 10,
prepared here for you to run yourself, not run automatically. Do not run
alongside scripts/run_ingest.py while that script still holds TransNetV2 +
BEiT-3 on the same GPU -- sequential only on 8GB VRAM.

Prerequisites:
  docker compose up -d      # MongoDB + Elasticsearch (Milvus not needed here)
  pip install -e ".[asr]"   # extends [ingest] with faster-whisper

Usage:
  python scripts/run_asr.py --batches L21 --limit 1   # eyeball one transcript first
  python scripts/run_asr.py --batches L21             # full batch (29 video)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import yaml

from app.config import Settings
from app.stores.es_client import AsrEsStore
from app.stores.mongo_client import AsrRunDocument, MongoStore
from datasets.aic2026 import AIC2026Adapter
from pipelines.asr import STATUS_DONE, STATUS_FAILED, AsrConfig, FasterWhisperClient, run_asr
from pipelines.video_io import read_video_info

# Windows consoles often default to cp1252, which cannot encode the
# Vietnamese text this project logs (video titles, transcript text) -- same
# fix as scripts/run_ingest.py.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_ASR_CONFIG_PATH = _REPO_ROOT / "configs" / "ingest" / "asr.yaml"
_ES_ASR_INDEX_NAME = "asr_segments"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--split", default="pilot", choices=["pilot", "all"])
    parser.add_argument(
        "--batches",
        type=str,
        default=None,
        help="Comma-separated batch prefixes to narrow --split to, e.g. 'L21' or 'L21,L22'.",
    )
    parser.add_argument("--dataset-root", type=Path, default=_REPO_ROOT / "data")
    parser.add_argument(
        "--eval-set-path", type=Path, default=_REPO_ROOT / "configs" / "eval" / "aic_eval_v1.json"
    )
    parser.add_argument("--device", default="cuda", help="'cuda' or 'cpu'")
    parser.add_argument("--limit", type=int, default=None, help="process only the first N videos")
    return parser.parse_args()


def _load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _batch_of(video_id: str) -> str:
    return video_id.split("_V", 1)[0]


async def _main() -> int:
    args = _parse_args()
    settings = Settings()

    adapter = AIC2026Adapter(dataset_root=args.dataset_root, eval_set_path=args.eval_set_path)
    asr_config_dict = _load_yaml(_ASR_CONFIG_PATH)
    asr_config_dict["device"] = args.device  # --device overrides asr.yaml's own value
    asr_config = AsrConfig(**asr_config_dict)
    asr_client = FasterWhisperClient(config=asr_config)  # built once, reused for every video

    mongo = MongoStore(uri=settings.mongo_uri)
    await mongo.ensure_indexes()
    es = AsrEsStore(
        url=settings.elasticsearch_url,
        index_name=_ES_ASR_INDEX_NAME,
        api_key=settings.elastic_api_key.get_secret_value() if settings.elastic_api_key else None,
    )
    await es.ensure_index()

    videos = list(adapter.iter_videos(args.split))
    if args.batches:
        wanted_batches = {b.strip() for b in args.batches.split(",")}
        videos = [v for v in videos if _batch_of(v.video_id) in wanted_batches]
        if not videos:
            logger.warning("--batches=%s matched no video in split=%s", args.batches, args.split)
    if args.limit is not None:
        videos = videos[: args.limit]
    logger.info(
        "Transcribing %d video(s) (split=%s%s)",
        len(videos),
        args.split,
        f", batches={args.batches}" if args.batches else "",
    )

    completed: list[str] = []
    skipped: list[str] = []
    failed: list[str] = []
    total_kept = 0
    total_dropped_no_speech = 0
    total_dropped_low_confidence = 0
    total_dropped_short_text = 0

    for video in videos:
        existing = await mongo.get_asr_run(video.video_id)
        if existing is not None and existing.get("asr_status") == STATUS_DONE:
            skipped.append(video.video_id)
            continue

        try:
            video_info = read_video_info(video.source_path)
            segments = asr_client.transcribe(video.source_path)
            summary = await run_asr(
                video.video_id,
                segments,
                es=es,
                fps=video_info.fps,
                frame_count=video_info.frame_count,
                config=asr_config,
            )
            await mongo.upsert_asr_run(
                AsrRunDocument(
                    video_id=video.video_id,
                    asr_status=STATUS_DONE,
                    num_segments=summary.kept_segments,
                    model_id=asr_config.model_size,
                    fps=video_info.fps,
                )
            )
            logger.info(
                "ASR video_id=%s: total_segments=%d kept=%d "
                "dropped(no_speech=%d, low_confidence=%d, short_text=%d)",
                video.video_id,
                summary.total_segments,
                summary.kept_segments,
                summary.dropped_no_speech,
                summary.dropped_low_confidence,
                summary.dropped_short_text,
            )
            completed.append(video.video_id)
            total_kept += summary.kept_segments
            total_dropped_no_speech += summary.dropped_no_speech
            total_dropped_low_confidence += summary.dropped_low_confidence
            total_dropped_short_text += summary.dropped_short_text
        except Exception as exc:
            # Deliberately broad -- same rationale as run_ingest.py's outer
            # boundary: faster-whisper/ctranslate2 failures (CUDA OOM, a
            # corrupt/unreadable .mp4 PyAV can't decode) aren't normalized to
            # a narrow exception type, and one bad video must not abort the
            # rest of the batch (CLAUDE.md §4/§7).
            logger.error("ASR failed for video_id=%s: %s", video.video_id, exc)
            await mongo.upsert_asr_run(
                AsrRunDocument(
                    video_id=video.video_id,
                    asr_status=STATUS_FAILED,
                    num_segments=0,
                    model_id=asr_config.model_size,
                    fps=0.0,
                )
            )
            failed.append(video.video_id)

    logger.info(
        "Done: %d completed, %d skipped (already done), %d failed",
        len(completed),
        len(skipped),
        len(failed),
    )
    if failed:
        logger.warning("Failed video_id(s): %s", failed)
    logger.info(
        "ASR hallucination-filter evidence (P0.11 Bước 1): kept=%d "
        "dropped(no_speech=%d, low_confidence=%d, short_text=%d)",
        total_kept,
        total_dropped_no_speech,
        total_dropped_low_confidence,
        total_dropped_short_text,
    )

    await mongo.close()
    await es.close()
    return 1 if failed else 0


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    sys.exit(asyncio.run(_main()))


if __name__ == "__main__":
    main()
