"""CLI: import a Kaggle-produced asr_out/ artifact (kaggle_asr_transcribe.ipynb)
into local Elasticsearch/MongoDB (.claude/plans/kaggle-asr-transcribe.plan.md,
Nhip A T3).

Kaggle only decodes Whisper and writes RAW (unfiltered) JSONL -- no
hallucination filter, no frame-index window, no ES/Mongo write (plan Q2).
This script is where all of that finally happens, using the SAME `run_asr()`
that `scripts/run_asr.py` calls for the local-decode path -- no second
filtering code path (pipelines/import_asr_out.py docstring).

Per video: verify `fps`/`frame_count` Kaggle exported against a fresh local
read of the SAME `.mp4` (F1 -- the two machines' copies are not provably
identical) -> re-cut batched-mode segments via word timestamps if configured
(configs/ingest/asr_offline.yaml's resegment_min_pause_sec) -> filter +
index through `run_asr()` -> record `asr_runs.asr_status` in MongoDB.

WARNING: touches Elasticsearch + MongoDB. This is an owner-operated command;
it is prepared here for the project owner to run, never for automatic execution.

Routine import prerequisites:
  MongoDB + Elasticsearch are already running and the artifact/model identity
  checks below are expected to pass. A routine import does not require reset.

Routine import:
  python scripts/import_asr_out.py --asr-out-dir data/asr_out
  python scripts/import_asr_out.py --asr-out-dir data/asr_out --skip-checkpoint-parity
  python scripts/import_asr_out.py --asr-out-dir data/asr_out --skip-checkpoint-parity \
    --trust-artifact-video-info
  python scripts/import_asr_out.py --asr-out-dir data/asr_out --limit-videos 1  # eyeball one first

The owner-only `--reset` flag only bypasses the different-model_id guard. It
does not delete Elasticsearch, MongoDB, or Milvus data. Before using it, the
owner must separately review the destructive replacement procedure and verify
explicit backup and restore inputs.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import yaml

from app.config import Settings
from app.stores.es_client import AsrEsStore
from app.stores.mongo_client import AsrRunDocument, MongoStore
from datasets.aic2026 import AIC2026Adapter
from pipelines.asr import STATUS_DONE, STATUS_FAILED, AsrConfig, run_asr
from pipelines.import_asr_out import (
    AsrArtifactError,
    build_segments_for_video,
    discover_video_ids,
    load_manifest,
    load_raw_segment_rows,
    load_run_index,
    verify_manifest,
    verify_video_info_matches,
)
from pipelines.video_io import read_video_info

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_ASR_CONFIG_PATH = _REPO_ROOT / "configs" / "ingest" / "asr.yaml"
_ASR_OFFLINE_CONFIG_PATH = _REPO_ROOT / "configs" / "ingest" / "asr_offline.yaml"
_ES_ASR_INDEX_NAME = "asr_segments"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--asr-out-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=_REPO_ROOT / "data")
    parser.add_argument(
        "--eval-set-path", type=Path, default=_REPO_ROOT / "configs" / "eval" / "aic_eval_v1.json"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help=(
            "owner-only: bypass the different-model_id guard after a separately reviewed "
            "replacement with verified backup/restore inputs; does not delete old documents"
        ),
    )
    parser.add_argument(
        "--limit-videos", type=int, default=0, help="0 = no limit; >0 to eyeball first"
    )
    parser.add_argument(
        "--skip-checkpoint-parity",
        action="store_true",
        help=(
            "trust the asr_out manifest without reading local model.bin/vocabulary.json; "
            "still requires manifest model_id to match configs/ingest/asr.yaml"
        ),
    )
    parser.add_argument(
        "--trust-artifact-video-info",
        action="store_true",
        help=(
            "do not require local .mp4 files; use fps/frame_count from runs.jsonl instead "
            "of cross-checking against local video metadata"
        ),
    )
    return parser.parse_args(argv)


def _load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


async def _existing_model_ids(mongo: MongoStore) -> set[str]:
    return {doc["model_id"] async for doc in mongo.asr_runs.find({}, {"model_id": 1})}


async def _main() -> int:
    args = _parse_args()
    settings = Settings()

    asr_config = AsrConfig(**_load_yaml(_ASR_CONFIG_PATH))
    offline_config = _load_yaml(_ASR_OFFLINE_CONFIG_PATH)

    manifest_path = args.asr_out_dir / "manifest.json"
    manifest = load_manifest(manifest_path)
    # model_id IS the relative checkpoint path (asr.yaml's own docstring:
    # "a local path e.g. models/PhoWhisper-large-ct2") -- join straight onto
    # the repo root, no need to reconstruct it from just the last segment.
    checkpoint_dir = _REPO_ROOT / asr_config.model_id
    try:
        verify_manifest(
            manifest,
            asr_config,
            checkpoint_dir,
            skip_checkpoint_parity=args.skip_checkpoint_parity,
        )
    except AsrArtifactError as exc:
        logger.error("Artifact rejected -- refusing to import: %s", exc)
        return 1
    if args.skip_checkpoint_parity:
        logger.warning(
            "Manifest checkpoint parity skipped: trusting artifact model_id=%s without local "
            "model.bin/vocabulary.json SHA verification",
            manifest["model_id"],
        )
    else:
        logger.info("Manifest verified: model_id=%s matches local checkpoint", manifest["model_id"])

    mongo = MongoStore(uri=settings.mongo_uri)
    await mongo.ensure_indexes()
    es = AsrEsStore(
        url=settings.elasticsearch_url,
        index_name=_ES_ASR_INDEX_NAME,
        api_key=settings.elastic_api_key.get_secret_value() if settings.elastic_api_key else None,
    )
    await es.ensure_index()

    if not args.reset:
        existing = await _existing_model_ids(mongo)
        stale = existing - {asr_config.model_id}
        if stale:
            logger.error(
                "asr_runs already holds model_id(s)=%s, importing model_id=%s would mix "
                "two models (R2/F5) -- refusing routine import. The owner must separately "
                "review the destructive replacement, verify backup/restore inputs, clear "
                "the old stores, and only then rerun with --reset; --reset alone deletes "
                "nothing",
                sorted(stale),
                asr_config.model_id,
            )
            await mongo.close()
            await es.close()
            return 1

    videos_by_id = {}
    if args.trust_artifact_video_info:
        logger.warning(
            "Local video info verification skipped: using fps/frame_count from runs.jsonl"
        )
    else:
        adapter = AIC2026Adapter(dataset_root=args.dataset_root, eval_set_path=args.eval_set_path)
        videos_by_id = {v.video_id: v for v in adapter.iter_videos("all")}

    segments_dir = args.asr_out_dir / "segments"
    run_index = load_run_index(args.asr_out_dir / "runs.jsonl")
    video_ids = discover_video_ids(segments_dir)
    if args.limit_videos > 0:
        video_ids = video_ids[: args.limit_videos]
    logger.info("Importing %d video(s) from %s", len(video_ids), args.asr_out_dir)

    imported: list[str] = []
    skipped: list[str] = []
    failed: list[str] = []
    total_kept = 0

    for video_id in video_ids:
        run_row = run_index.get(video_id)
        if run_row is None or run_row.get("status") != "ok":
            logger.warning(
                "%s: no ok row in runs.jsonl (status=%s) -- skipping",
                video_id,
                run_row and run_row.get("status"),
            )
            skipped.append(video_id)
            continue

        try:
            if args.trust_artifact_video_info:
                fps = float(run_row["fps"])
                frame_count = int(run_row["frame_count"])
            else:
                video_ref = videos_by_id.get(video_id)
                if video_ref is None:
                    raise AsrArtifactError(
                        f"{video_id}: not found via AIC2026Adapter.iter_videos('all')"
                    )
                local_video_info = read_video_info(video_ref.source_path)
                verify_video_info_matches(
                    video_id,
                    local_video_info,
                    kaggle_fps=run_row["fps"],
                    kaggle_frame_count=run_row["frame_count"],
                    fps_tolerance=offline_config["fps_tolerance"],
                    frame_count_tolerance=offline_config["frame_count_tolerance"],
                )
                fps = local_video_info.fps
                frame_count = local_video_info.frame_count
            raw_rows = load_raw_segment_rows(segments_dir / f"{video_id}.jsonl")
            segments = build_segments_for_video(raw_rows, offline_config["resegment_min_pause_sec"])
            summary = await run_asr(
                video_id,
                segments,
                es=es,
                fps=fps,
                frame_count=frame_count,
                config=asr_config,
            )
            await mongo.upsert_asr_run(
                AsrRunDocument(
                    video_id=video_id,
                    asr_status=STATUS_DONE,
                    num_segments=summary.kept_segments,
                    model_id=asr_config.model_id,
                    fps=fps,
                )
            )
            imported.append(video_id)
            total_kept += summary.kept_segments
            logger.info(
                "%s: %d raw -> %d kept (dropped no_speech=%d low_confidence=%d short_text=%d)",
                video_id,
                summary.total_segments,
                summary.kept_segments,
                summary.dropped_no_speech,
                summary.dropped_low_confidence,
                summary.dropped_short_text,
            )
        except (AsrArtifactError, OSError, ValueError) as exc:
            # Same boundary rationale as run_asr.py: one bad video (corrupt
            # .mp4, cross-check failure) must not abort the rest of the batch.
            logger.error("%s: import failed: %s", video_id, exc)
            await mongo.upsert_asr_run(
                AsrRunDocument(
                    video_id=video_id,
                    asr_status=STATUS_FAILED,
                    num_segments=0,
                    model_id=asr_config.model_id,
                    fps=0.0,
                )
            )
            failed.append(video_id)

    logger.info(
        "Done: %d imported, %d skipped (no ok row), %d failed. Total kept segments: %d",
        len(imported),
        len(skipped),
        len(failed),
        total_kept,
    )
    if failed:
        logger.warning("Failed video_id(s): %s", failed)

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
