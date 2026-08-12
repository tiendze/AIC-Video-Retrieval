"""CLI: import the team's already-run OCR output (`data/ocr_results.jsonl`)
into Elasticsearch (`ocr_keyframes` index) -- no Gemini call, just a bulk
import of text that was already extracted offline.

For every `video_id` in the file: reads that video's already-imported
MongoDB keyframe docs (via `scripts/import_embed_out.py`), builds a
`frame_idx -> (keyframe_id, timestamp)` index, and joins each OCR record to
it by `frame_id`. A video with zero MongoDB keyframes is skipped outright
(not yet imported -- report and move on, don't guess); a record whose
`frame_id` doesn't match any keyframe of an otherwise-imported video is
counted and logged, never silently dropped (CLAUDE.md §7).

WARNING: per CLAUDE.md §10, this writes to a live Elasticsearch index and
reads from a live MongoDB. Prepared here for you to run yourself, not run
automatically.

Prerequisites:
  docker compose up -d                        # MongoDB + Elasticsearch
  python scripts/import_embed_out.py --reset  # must run FIRST -- this
                                                # script only reads Mongo,
                                                # it never writes keyframes

Usage:
  python scripts/import_ocr_results.py
  python scripts/import_ocr_results.py --limit-videos 1   # smoke test
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from app.config import Settings
from app.stores.es_client import EsStore
from app.stores.mongo_client import MongoStore
from pipelines.import_ocr_results import (
    build_keyframe_index,
    build_ocr_documents_for_video,
    load_ocr_results,
)

# Windows consoles often default to cp1252, which cannot encode the
# Vietnamese OCR text this script logs -- same fix as run_ingest.py.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_ES_OCR_INDEX_NAME = "ocr_keyframes"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--ocr-jsonl", type=Path, default=_REPO_ROOT / "data" / "ocr_results.jsonl"
    )
    parser.add_argument(
        "--limit-videos", type=int, default=None, help="Import only the first N video_ids"
    )
    return parser.parse_args()


async def _main() -> int:
    args = _parse_args()
    settings = Settings()

    mongo = MongoStore(uri=settings.mongo_uri)
    es = EsStore(
        url=settings.elasticsearch_url,
        index_name=_ES_OCR_INDEX_NAME,
        api_key=settings.elastic_api_key.get_secret_value() if settings.elastic_api_key else None,
    )
    await es.ensure_index()

    grouped = load_ocr_results(args.ocr_jsonl)
    video_ids = sorted(grouped)
    if args.limit_videos is not None:
        video_ids = video_ids[: args.limit_videos]
    logger.info("Importing OCR for %d video(s) from %s", len(video_ids), args.ocr_jsonl)

    total_matched = 0
    total_unmatched = 0
    skipped_video_ids: list[str] = []
    unmatched_by_video: dict[str, int] = {}

    for video_id in video_ids:
        mongo_docs = await mongo.get_keyframes_for_video(video_id)
        if not mongo_docs:
            skipped_video_ids.append(video_id)
            continue
        keyframe_index = build_keyframe_index(mongo_docs)
        docs, unmatched = build_ocr_documents_for_video(video_id, grouped[video_id], keyframe_index)
        if docs:
            await es.bulk_index(docs)
        total_matched += len(docs)
        if unmatched:
            total_unmatched += len(unmatched)
            unmatched_by_video[video_id] = len(unmatched)

    logger.info(
        "Done: %d record(s) indexed, %d unmatched frame_id(s) across %d video(s), "
        "%d video(s) skipped (not yet in MongoDB -- run scripts/import_embed_out.py first)",
        total_matched,
        total_unmatched,
        len(unmatched_by_video),
        len(skipped_video_ids),
    )
    if skipped_video_ids:
        logger.warning("Skipped video_id(s) (not in MongoDB): %s", skipped_video_ids)
    if unmatched_by_video:
        logger.warning("Unmatched frame_id count by video_id: %s", unmatched_by_video)

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
