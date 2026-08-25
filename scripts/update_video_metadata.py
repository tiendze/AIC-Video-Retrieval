"""Metadata-only refresh for the MongoDB `videos` collection.

Populates/repairs the organiser metadata fields (`watch_url`, `title`,
`publish_date`, `channel_id`, `author`, `duration`) on videos that are
ALREADY ingested, WITHOUT re-running the full embed import.

Why this exists: `scripts/import_embed_out.py` is the only code path that
writes these fields, and it skips any video already `ingest_status=complete`
(`is_video_already_imported`). So once a corpus is ingested, there is no
built-in way to backfill metadata short of a destructive `--reset` +
re-ingest of every keyframe -- which is impossible here anyway when
`data/embed_out/` is no longer on disk. This script closes that gap.

Safety contract:
  * Reads the CURRENT video_id list straight from MongoDB (the ground truth of
    what is ingested), then loads `media-info/<video_id>.json` by matching
    filename -- it never touches `data/embed_out/` and never needs the raw
    `.mp4` (so it works on a machine that has neither).
  * Only ever `$set`s the metadata fields listed above. It NEVER writes
    `s_v`/`e_v`/`global_index`/`fps`/`num_keyframes`/`ingest_status`/`batch`
    or any keyframe row -- the fields DANTE/search/TRAKE depend on are left
    exactly as ingested (CLAUDE.md §3).
  * Dry-run by default: prints what WOULD change and exits without writing.
    Pass --apply to actually update.

Usage:
  python scripts/update_video_metadata.py                 # dry-run report
  python scripts/update_video_metadata.py --apply         # write the changes
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

from pymongo.errors import PyMongoError

from app.config import Settings
from app.stores.mongo_client import MongoStore

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent

# media-info key -> MongoDB videos field. Mirrors scripts/import_embed_out.py's
# VideoRef mapping (lines 445-450) so a refreshed doc matches what a fresh
# ingest would have written.
_STRING_FIELD_MAP = {
    "title": "title",
    "publish_date": "publish_date",
    "watch_url": "watch_url",
    "channel_id": "channel_id",
    "author": "author",
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=_REPO_ROOT / "data" / "metadata" / "media-info",
        help="Directory of <video_id>.json media-info files",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually write updates. Without it the script is a read-only dry run.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="0 = all videos; >0 to process only the first N (report/eyeball).",
    )
    return parser.parse_args(argv)


def _load_media_info(metadata_dir: Path, video_id: str) -> dict[str, Any] | None:
    path = metadata_dir / f"{video_id}.json"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _build_update(metadata: dict[str, Any]) -> dict[str, Any]:
    """The `$set` payload for one video: only fields the media-info actually
    carries a non-empty value for, so an update never clobbers an existing
    good value with an empty string."""
    update: dict[str, Any] = {}
    for source_key, mongo_field in _STRING_FIELD_MAP.items():
        value = metadata.get(source_key)
        if value:
            update[mongo_field] = str(value)
    length = metadata.get("length")
    if length is not None:
        try:
            update["duration"] = int(float(length))
        except (TypeError, ValueError):
            pass  # a non-numeric length is metadata noise, not worth failing on
    return update


def _needs_change(existing: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    """Subset of `update` whose values differ from what MongoDB already holds
    -- lets the report distinguish 'already correct' from 'will change'."""
    return {k: v for k, v in update.items() if existing.get(k) != v}


async def _run(args: argparse.Namespace) -> int:
    settings = Settings()
    mongo = MongoStore(uri=settings.mongo_uri)

    video_ids: list[str] = []
    existing_by_id: dict[str, dict[str, Any]] = {}
    async for doc in mongo.scroll_all_videos():
        video_id = doc["video_id"]
        video_ids.append(video_id)
        existing_by_id[video_id] = doc
    video_ids.sort()
    if args.limit > 0:
        video_ids = video_ids[: args.limit]

    logger.info("MongoDB holds %d video(s); metadata dir=%s", len(video_ids), args.metadata_dir)

    missing_file: list[str] = []
    to_change: list[tuple[str, dict[str, Any]]] = []
    already_ok = 0

    for video_id in video_ids:
        metadata = _load_media_info(args.metadata_dir, video_id)
        if metadata is None:
            missing_file.append(video_id)
            continue
        update = _build_update(metadata)
        changes = _needs_change(existing_by_id[video_id], update)
        if changes:
            to_change.append((video_id, changes))
        else:
            already_ok += 1

    # Sample so the operator can eyeball correctness before --apply.
    logger.info("--- sample of pending changes (up to 5) ---")
    for video_id, changes in to_change[:5]:
        before = {k: existing_by_id[video_id].get(k) for k in changes}
        logger.info("%s: %s -> %s", video_id, before, changes)

    logger.info(
        "Summary: %d already correct, %d to change, %d missing media-info file",
        already_ok,
        len(to_change),
        len(missing_file),
    )
    if missing_file:
        logger.warning("No media-info file for %d video(s), e.g. %s", len(missing_file), missing_file[:10])

    if not args.apply:
        logger.info("DRY RUN -- no writes. Re-run with --apply to persist the %d change(s).", len(to_change))
        await mongo.close()
        return 0

    updated = 0
    for video_id, changes in to_change:
        try:
            result = await mongo.videos.update_one({"video_id": video_id}, {"$set": changes})
            updated += result.modified_count
        except PyMongoError as exc:  # log which video, then keep going
            logger.error("%s: metadata update failed: %s", video_id, exc)
    logger.info("APPLIED: %d video document(s) updated.", updated)

    await mongo.close()
    return 0


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sys.exit(asyncio.run(_run(_parse_args())))


if __name__ == "__main__":
    main()
