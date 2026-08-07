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

After the loop, the whole `asr_segments` index is dumped to one shareable
JSON file (--output-json) -- read back from Elasticsearch, not from this
run's in-memory results, so previously-transcribed videos skipped by the
resume check are still included. Re-running with every video already `done`
therefore rebuilds the JSON without loading Whisper or touching the GPU.

WARNING: GPU-heavy (Whisper decode, checkpoint set by configs/ingest/asr.yaml's
model_id) -- per CLAUDE.md section 10, prepared here for you to run yourself,
not run automatically. Do not run alongside scripts/run_ingest.py while that
script still holds TransNetV2 + BEiT-3 on the same GPU -- sequential only on
8GB VRAM.

A video already `asr_status="done"` under the SAME model_id is skipped on a
re-run; changing model_id in asr.yaml (or passing --force) re-transcribes it
even if a prior run under a different model finished (see should_transcribe
in pipelines/asr.py).

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
import json
import logging
import sys
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import yaml

from app.config import Settings
from app.stores.es_client import AsrEsStore
from app.stores.mongo_client import AsrRunDocument, MongoStore
from datasets.aic2026 import AIC2026Adapter
from pipelines.asr import (
    STATUS_DONE,
    STATUS_FAILED,
    AsrConfig,
    FasterWhisperClient,
    VideoTiming,
    format_duration,
    run_asr,
    should_transcribe,
    summarize_timings,
)
from pipelines.export_asr import build_asr_export, build_shot_aligned_export
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
_DEFAULT_OUTPUT_JSON = _REPO_ROOT / "data" / "asr_transcripts.json"
_DEFAULT_SHOTS_JSON = _REPO_ROOT / "data" / "shots.json"
_DEFAULT_OUTPUT_SHOT_JSON = _REPO_ROOT / "data" / "asr_by_shot.json"


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
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="process only the first N videos; --limit 0 transcribes nothing and "
        "only rebuilds --output-json from Elasticsearch",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=_DEFAULT_OUTPUT_JSON,
        help="where to dump the shareable transcript JSON (rebuilt from Elasticsearch)",
    )
    parser.add_argument(
        "--shots-json",
        type=Path,
        default=_DEFAULT_SHOTS_JSON,
        help="shot boundaries to lay the transcript over (team-provided file)",
    )
    parser.add_argument(
        "--output-shot-json",
        type=Path,
        default=_DEFAULT_OUTPUT_SHOT_JSON,
        help="where to dump the shot-aligned transcript JSON",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-transcribe even videos already done under the same model_id "
        "(normally only a model_id change or a prior failure triggers a re-run)",
    )
    return parser.parse_args()


def _load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _batch_of(video_id: str) -> str:
    return video_id.split("_V", 1)[0]


def _log_timing_summary(timings: Sequence[VideoTiming]) -> None:
    """Reports what this run cost, and what it implies for the rest of the
    corpus. Covers only videos transcribed *in this run* — ones skipped as
    already `done` carry no timing, so a resumed run reports its own slice
    rather than a cumulative total."""
    summary = summarize_timings(timings)
    if summary.num_videos == 0:
        logger.info("No video transcribed in this run — no timing to report")
        return

    logger.info(
        "ASR timing: %d video in %s (mean %s/video) for %s of audio — %.1fx realtime",
        summary.num_videos,
        format_duration(summary.total_wall_sec),
        format_duration(summary.mean_wall_sec),
        format_duration(summary.total_audio_sec),
        summary.speed_ratio,
    )
    slowest = max(timings, key=lambda timing: timing.wall_sec)
    logger.info(
        "Slowest video: %s at %s for %s of audio",
        slowest.video_id,
        format_duration(slowest.wall_sec),
        format_duration(slowest.audio_sec),
    )


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(document, f, ensure_ascii=False, indent=2)


async def _dump_json_artifacts(es: AsrEsStore, args: argparse.Namespace, model_id: str) -> bool:
    """Writes both shareable files and reports whether it fully succeeded.

    Rebuilds from the whole ES index rather than this run's in-memory results,
    so videos skipped by the resume check are still included:

      --output-json       flat, one row per segment  (retrieval's own view)
      --output-shot-json  the same speech laid over shots.json's shot cuts

    The flat file is written first, so a missing/misshapen shots.json costs the
    shot-aligned file only. Both `ValueError` cases mean "the data isn't there"
    (empty index, or no shared video_id) and must be visible, not swallowed —
    any Elasticsearch transport error still propagates (CLAUDE.md §7).
    """
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    await es.refresh()
    documents = [document async for document in es.scroll_all()]

    try:
        export = build_asr_export(documents, model_id=model_id, generated_at=generated_at)
    except ValueError as exc:
        logger.error("Transcript JSON not written: %s", exc)
        return False

    _write_json(args.output_json, export)
    logger.info(
        "Wrote transcript JSON: %s (%d video, %d segment)",
        args.output_json,
        export["meta"]["num_videos"],
        export["meta"]["num_segments"],
    )

    try:
        shots_document = _load_json(args.shots_json)
        aligned = build_shot_aligned_export(export, shots_document, generated_at=generated_at)
    except FileNotFoundError:
        logger.error("Shot-aligned JSON not written: shots file not found: %s", args.shots_json)
        return False
    except ValueError as exc:
        logger.error("Shot-aligned JSON not written: %s", exc)
        return False

    _write_json(args.output_shot_json, aligned)
    logger.info(
        "Wrote shot-aligned JSON: %s (%d shot, %d with speech, %d transcript row)",
        args.output_shot_json,
        aligned["meta"]["num_shots"],
        aligned["meta"]["num_shots_with_speech"],
        aligned["meta"]["num_transcript_rows"],
    )
    if aligned["meta"]["videos_missing_shots"]:
        logger.warning(
            "%d video have ASR but no entry in %s (kept in %s only): %s",
            len(aligned["meta"]["videos_missing_shots"]),
            args.shots_json,
            args.output_json,
            aligned["meta"]["videos_missing_shots"],
        )
    return True


async def _main() -> int:
    args = _parse_args()
    settings = Settings()

    adapter = AIC2026Adapter(dataset_root=args.dataset_root, eval_set_path=args.eval_set_path)
    asr_config_dict = _load_yaml(_ASR_CONFIG_PATH)
    asr_config_dict["device"] = args.device  # --device overrides asr.yaml's own value
    asr_config = AsrConfig(**asr_config_dict)
    # Built once on first real need, then reused for every video. Lazy so an
    # export-only re-run (every video already `done`) neither downloads the
    # checkpoint nor claims VRAM.
    asr_client: FasterWhisperClient | None = None

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
    timings: list[VideoTiming] = []
    total_kept = 0
    total_dropped_no_speech = 0
    total_dropped_low_confidence = 0
    total_dropped_short_text = 0

    for video in videos:
        existing = await mongo.get_asr_run(video.video_id)
        if not should_transcribe(existing, model_id=asr_config.model_id, force=args.force):
            skipped.append(video.video_id)
            continue

        if asr_client is None:
            # Outside the try on purpose: a missing faster-whisper install or a
            # failed checkpoint download must surface as itself, not be
            # recorded as every video failing to transcribe.
            asr_client = FasterWhisperClient(config=asr_config)

        started = time.perf_counter()
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
                    model_id=asr_config.model_id,
                    fps=video_info.fps,
                )
            )
            timing = VideoTiming(
                video_id=video.video_id,
                wall_sec=time.perf_counter() - started,
                # From the decoder, not media-info's `length`, which is rounded
                # to whole seconds and would skew the speed ratio.
                audio_sec=video_info.frame_count / video_info.fps,
            )
            timings.append(timing)
            logger.info(
                "ASR video_id=%s: total_segments=%d kept=%d "
                "dropped(no_speech=%d, low_confidence=%d, short_text=%d) "
                "in %s for %s of audio (%.1fx realtime)",
                video.video_id,
                summary.total_segments,
                summary.kept_segments,
                summary.dropped_no_speech,
                summary.dropped_low_confidence,
                summary.dropped_short_text,
                format_duration(timing.wall_sec),
                format_duration(timing.audio_sec),
                timing.audio_sec / timing.wall_sec if timing.wall_sec else 0.0,
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
                    model_id=asr_config.model_id,
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
    _log_timing_summary(timings)

    export_ok = await _dump_json_artifacts(es, args, asr_config.model_id)

    await mongo.close()
    await es.close()
    return 1 if failed or not export_ok else 0


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    sys.exit(asyncio.run(_main()))


if __name__ == "__main__":
    main()
