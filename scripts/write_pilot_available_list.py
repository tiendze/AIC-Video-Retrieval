"""CLI: regenerate `configs/eval/pilot_available_v1.txt` from MongoDB's
live `ingest_status` (2026-08-10 follow-up to plan §P0.9) -- see
`pipelines/write_pilot_available_list.py` for why this is needed (the
Kaggle-offline-embed import path never wrote to this file, so it was stuck
at 29/873 video_ids).

Run this BEFORE `python -m eval.run --config configs/eval/aic_pilot.yaml` --
otherwise P0.9 scores against a stale, mostly-wrong denominator.

WARNING: reads a live MongoDB. Per CLAUDE.md §10, prepared here for the
project owner to run, not run automatically by this agent.

Usage:
  python scripts/write_pilot_available_list.py
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

from app.config import Settings
from app.stores.mongo_client import MongoStore
from pipelines.write_pilot_available_list import available_video_ids, write_pilot_available_file

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_OUT_PATH = _REPO_ROOT / "configs" / "eval" / "pilot_available_v1.txt"


async def _main() -> int:
    settings = Settings()
    mongo = MongoStore(uri=settings.mongo_uri)
    try:
        video_docs = [doc async for doc in mongo.scroll_all_videos()]
    finally:
        await mongo.close()

    video_ids = available_video_ids(video_docs)
    digest = write_pilot_available_file(video_ids, _OUT_PATH)
    logger.info(
        "Wrote %d video_id(s) (of %d total video doc(s) in MongoDB) to %s (sha256=%s)",
        len(video_ids),
        len(video_docs),
        _OUT_PATH,
        digest,
    )
    return 0


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    sys.exit(asyncio.run(_main()))


if __name__ == "__main__":
    main()
