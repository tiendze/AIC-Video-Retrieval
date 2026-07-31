"""MongoStore — the only layer that talks to MongoDB directly (architecture.md §4.3).

Uses `AsyncMongoClient` (native async, PyMongo >=4.9) since callers are async
FastAPI endpoints / async ingest tasks — a sync `MongoClient` inside `async def`
would block the event loop (`.claude/rules/ecc/python/fastapi.md`).

Unique indexes on `keyframe_id` / `video_id` are the idempotency mechanism
(CLAUDE.md T1): re-running ingest on an already-processed video raises a
duplicate-key error per row instead of creating a second record.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from pymongo import AsyncMongoClient, InsertOne
from pymongo.errors import BulkWriteError

logger = logging.getLogger(__name__)

_DUPLICATE_KEY_ERROR_CODE = 11000


@dataclass(frozen=True)
class KeyframeDocument:
    keyframe_id: str
    video_id: str
    shot_id: str
    frame_idx: int
    timestamp: float
    global_index: int
    image_path: str
    thumbnail_path: str


@dataclass(frozen=True)
class VideoDocument:
    video_id: str
    s_v: int
    e_v: int
    duration: int
    fps: float
    num_keyframes: int
    ingest_status: str
    author: str
    batch: str
    title: str
    publish_date: str
    watch_url: str
    channel_id: str


class MongoStore:
    """Metadata store for the `keyframes` and `videos` collections."""

    def __init__(self, uri: str, database_name: str = "aic_video_retrieval") -> None:
        self._client: AsyncMongoClient = AsyncMongoClient(uri)
        self._db = self._client[database_name]
        self.keyframes = self._db["keyframes"]
        self.videos = self._db["videos"]

    async def ensure_indexes(self) -> None:
        await self.keyframes.create_index("keyframe_id", unique=True)
        await self.videos.create_index("video_id", unique=True)

    async def bulk_insert_keyframes(self, docs: Sequence[KeyframeDocument]) -> int:
        """Insert keyframe documents. Returns the number of newly inserted rows.

        A previously-ingested `keyframe_id` raises a duplicate-key write error,
        which is swallowed here (idempotent re-run) — any other write error
        re-raises, since that is a real failure, not an expected duplicate.
        """
        if not docs:
            return 0
        operations = [InsertOne(document=_keyframe_document_to_dict(d)) for d in docs]
        try:
            result = await self.keyframes.bulk_write(operations, ordered=False)
            return result.inserted_count
        except BulkWriteError as exc:
            write_errors = exc.details["writeErrors"]
            real_errors = [e for e in write_errors if e["code"] != _DUPLICATE_KEY_ERROR_CODE]
            if real_errors:
                failed_ids = [docs[e["index"]].keyframe_id for e in real_errors]
                logger.error(
                    "bulk_insert_keyframes: non-duplicate write error(s) for keyframe_ids=%s",
                    failed_ids,
                )
                raise
            duplicate_ids = [docs[e["index"]].keyframe_id for e in write_errors]
            logger.info(
                "bulk_insert_keyframes: skipped %d duplicate keyframe_id(s)=%s",
                len(duplicate_ids),
                duplicate_ids,
            )
            return exc.details["nInserted"]

    async def upsert_video(self, doc: VideoDocument) -> None:
        """Insert-or-replace a video's metadata/ingest_status, keyed by `video_id`."""
        await self.videos.replace_one(
            {"video_id": doc.video_id}, _video_document_to_dict(doc), upsert=True
        )

    async def get_video(self, video_id: str) -> dict[str, Any] | None:
        return await self.videos.find_one({"video_id": video_id})

    async def close(self) -> None:
        await self._client.close()


def _keyframe_document_to_dict(doc: KeyframeDocument) -> dict[str, Any]:
    return {
        "keyframe_id": doc.keyframe_id,
        "video_id": doc.video_id,
        "shot_id": doc.shot_id,
        "frame_idx": doc.frame_idx,
        "timestamp": doc.timestamp,
        "global_index": doc.global_index,
        "image_path": doc.image_path,
        "thumbnail_path": doc.thumbnail_path,
    }


def _video_document_to_dict(doc: VideoDocument) -> dict[str, Any]:
    return {
        "video_id": doc.video_id,
        "s_v": doc.s_v,
        "e_v": doc.e_v,
        "duration": doc.duration,
        "fps": doc.fps,
        "num_keyframes": doc.num_keyframes,
        "ingest_status": doc.ingest_status,
        "author": doc.author,
        "batch": doc.batch,
        "title": doc.title,
        "publish_date": doc.publish_date,
        "watch_url": doc.watch_url,
        "channel_id": doc.channel_id,
    }
