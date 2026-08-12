"""CLI: warm the on-demand thumbnail cache `app/services/thumbnail_service.py`
serves from `GET /api/keyframes/{id}/thumbnail` (plan §P1.6 backend gap 1.3).

The imported Kaggle keyframe set has no pre-made thumbnail
(`pipelines/import_embed_out.py` sets `thumbnail_path == image_path`), so the
live corpus is 221,903 full-res JPEGs (~40 GB) with nothing pre-resized.
`ThumbnailService` resizes lazily on first request; this script resizes
everything up front instead, so the frontend's first real session doesn't
pay per-image resize latency for the whole corpus.

Calls `app.services.thumbnail_service.resize_and_save()` and
`cache_path_for()` directly -- the exact functions the live API uses --
so there is exactly one definition of what a "thumbnail" is (plan §P1.6:
no second code path to drift from the on-demand one).

Resumable: skips any keyframe whose cache file already exists.

Resizing runs on a `ThreadPoolExecutor` (`thumbnail_build_workers` in
configs/search/search.yaml, default 8), not a sequential loop -- PIL
decode/resize/encode is CPU-bound and releases the GIL for the bulk of its
work, the same reasoning already measured for `ClipEncoder`'s
`preprocess_workers` (app/encoders/clip_baseline.py's docstring: batch_size
alone barely moved throughput because image decode, not GPU/compute, was
the bottleneck there). No GPU or model is involved here at all, so this is
pure I/O+CPU parallelism.

WARNING: resource-heavy (reads/resizes up to 221,903 images, ~40 GB source).
Per CLAUDE.md §10, prepared here for the project owner to run themselves,
not run automatically by this agent.

Usage:
  # time a pilot slice FIRST (CLAUDE.md §10.4 -- do not guess throughput):
  python scripts/build_thumbnails.py --limit-videos 5
  python scripts/build_thumbnails.py                    # full corpus
  python scripts/build_thumbnails.py --workers 16        # override config default
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

from app.config import Settings
from app.services.thumbnail_service import cache_path_for, resize_and_save
from app.stores.mongo_client import MongoStore

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SEARCH_CONFIG_PATH = _REPO_ROOT / "configs" / "search" / "search.yaml"
_PROGRESS_LOG_INTERVAL = 500  # log every N completed thumbnails, not every one


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--limit-videos",
        type=int,
        default=None,
        help="process only the first N videos -- for timing a pilot slice "
        "before committing to the full corpus (CLAUDE.md §10.4)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="overrides configs/search/search.yaml's thumbnail_build_workers",
    )
    return parser.parse_args()


def _load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


async def _group_keyframes_by_video(mongo: MongoStore) -> dict[str, list[dict]]:
    by_video: dict[str, list[dict]] = defaultdict(list)
    async for doc in mongo.scroll_all_keyframes():
        by_video[doc["video_id"]].append(doc)
    return by_video


def _partition_pending(
    video_ids: list[str], by_video: dict[str, list[dict]], cache_dir: Path
) -> tuple[list[tuple[dict, Path]], int]:
    """Splits keyframes into (doc, cache_path) pairs still needing a
    resize vs. an already-cached count -- the existence check itself is
    cheap stat() I/O, done up front so the thread pool below only ever
    submits real resize work."""
    pending: list[tuple[dict, Path]] = []
    skipped_count = 0
    for video_id in video_ids:
        for doc in by_video[video_id]:
            cache_path = cache_path_for(cache_dir, doc["keyframe_id"])
            if cache_path.exists():
                skipped_count += 1
            else:
                pending.append((doc, cache_path))
    return pending, skipped_count


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()
    settings = Settings()
    config = _load_yaml(_SEARCH_CONFIG_PATH)
    cache_dir = _REPO_ROOT / config["thumbnail_cache_dir"]
    max_px = config["thumbnail_max_px"]
    jpeg_quality = config["thumbnail_jpeg_quality"]
    workers = args.workers or config["thumbnail_build_workers"]

    mongo = MongoStore(uri=settings.mongo_uri)
    by_video = await _group_keyframes_by_video(mongo)
    await mongo.close()

    video_ids = list(by_video)
    if args.limit_videos is not None:
        video_ids = video_ids[: args.limit_videos]

    pending, skipped_count = _partition_pending(video_ids, by_video, cache_dir)
    logger.info(
        "Warming thumbnail cache: %d video(s), %d to resize, %d already cached, "
        "%d worker(s) -> %s",
        len(video_ids),
        len(pending),
        skipped_count,
        workers,
        cache_dir,
    )

    start = time.monotonic()
    resized_count = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                resize_and_save, Path(doc["image_path"]), cache_path, max_px, jpeg_quality
            )
            for doc, cache_path in pending
        ]
        for future in as_completed(futures):
            future.result()  # surfaces the first exception instead of swallowing it
            resized_count += 1
            if resized_count % _PROGRESS_LOG_INTERVAL == 0:
                logger.info("[%d/%d] resized", resized_count, len(pending))

    elapsed = time.monotonic() - start
    logger.info(
        "Done. %d resized, %d already cached, in %.1fs (%.1f thumbnails/s).",
        resized_count,
        skipped_count,
        elapsed,
        resized_count / elapsed if elapsed > 0 else 0.0,
    )


if __name__ == "__main__":
    asyncio.run(main())
