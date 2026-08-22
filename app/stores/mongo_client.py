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
import re
import unicodedata
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any

from pymongo import AsyncMongoClient, InsertOne
from pymongo.errors import BulkWriteError, PyMongoError

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
class AsrRunDocument:
    """One video's ASR transcription run (P0.11a) — deliberately its own
    collection, not a field on `VideoDocument`: `upsert_video()` is a full
    `replace_one` (see below), so an ASR field added there would be
    silently overwritten every time ingest re-runs for that video."""

    video_id: str
    asr_status: str
    num_segments: int
    model_id: str
    fps: float


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
        self.asr_runs = self._db["asr_runs"]

    async def ping(self) -> bool:
        """Connectivity check for `/health` (plan §P0.8). A store being
        unreachable is expected degraded-mode data (CLAUDE.md §4), not a
        bug -- returns False instead of letting `PyMongoError` propagate."""
        try:
            await self._client.admin.command("ping")
            return True
        except PyMongoError:
            return False

    async def ensure_indexes(self) -> None:
        await self.keyframes.create_index("keyframe_id", unique=True)
        await self.videos.create_index("video_id", unique=True)
        await self.asr_runs.create_index("video_id", unique=True)

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

    async def get_keyframes_for_video(self, video_id: str) -> list[dict[str, Any]]:
        """All persisted keyframe rows for one video, ordered by `global_index`
        — used to resume the TEXT/OCR branch for a video whose VISION branch
        already finished in an earlier run, without redoing shot detection."""
        cursor = self.keyframes.find({"video_id": video_id}).sort("global_index", 1)
        return [doc async for doc in cursor]

    async def get_keyframes_by_ids(self, keyframe_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
        """Batch lookup by primary key, keyed by `keyframe_id` in the result --
        used by SearchService to join `frame_idx`/`thumbnail_path` onto search
        hits. Elasticsearch OCR hits carry no `frame_idx` at all (CLAUDE.md §3
        only requires it in Milvus/MongoDB), so OCR-mode results need this
        lookup just to be submittable."""
        if not keyframe_ids:
            return {}
        cursor = self.keyframes.find({"keyframe_id": {"$in": list(keyframe_ids)}})
        docs = [doc async for doc in cursor]
        return {doc["keyframe_id"]: doc for doc in docs}

    async def scroll_all_videos(self) -> AsyncIterator[dict[str, Any]]:
        """Every persisted video row (with its `ingest_status`), unfiltered --
        the live-state source `scripts/write_pilot_available_list.py`
        (2026-08-10 follow-up) reads to regenerate `pilot_available_v1.txt`.
        Needed because the Kaggle-offline-embed import path
        (`scripts/import_embed_out.py`) writes `ingest_status` straight to
        this collection and never touches that file -- only
        `scripts/run_ingest.py`'s own local runs did."""
        cursor = self.videos.find({})
        async for doc in cursor:
            yield doc

    async def search_videos_by_title(self, query_text: str, top_k: int) -> list[dict[str, Any]]:
        """Lexical title search over existing video metadata.

        The project has no title embedding/index today, and the video table is
        small enough to score in process. This keeps Title independent from OCR
        while avoiding a new retrieval subsystem.
        """
        query_tokens = _title_tokens(query_text)
        if not query_tokens or top_k <= 0:
            return []

        scored: list[tuple[float, str, dict[str, Any]]] = []
        async for doc in self.scroll_all_videos():
            title = str(doc.get("title", ""))
            normalized_title = _normalize_title(title)
            if not normalized_title:
                continue
            token_hits = sum(1 for token in query_tokens if token in normalized_title)
            if token_hits == 0:
                continue
            exact_bonus = 1.0 if _normalize_title(query_text) in normalized_title else 0.0
            score = token_hits / len(query_tokens) + exact_bonus
            scored.append((score, str(doc.get("video_id", "")), doc))

        scored.sort(key=lambda item: (-item[0], item[1]))
        return [{**doc, "score": score} for score, _, doc in scored[:top_k]]

    async def scroll_all_keyframes(self) -> AsyncIterator[dict[str, Any]]:
        """Every persisted keyframe row across the whole corpus, unfiltered
        -- the sampling pool `eval/baselines/random.py` (plan §P0.9) draws
        from. Deliberately different from `get_keyframes_for_video`/
        `get_keyframes_by_ids`: the random baseline needs to sample from
        every ingested keyframe, not one video or one id list."""
        cursor = self.keyframes.find({})
        async for doc in cursor:
            yield doc

    async def max_keyframe_global_index(self) -> int | None:
        """Highest `global_index` actually claimed by a written keyframe row
        — the ingest orchestrator's true recovery point for the next
        `global_index` range. Deliberately reads `keyframes`, not
        `VideoDocument.e_v`: `e_v` is only written once a video's ingest
        fully finishes, so recovering from it would miss `global_index`
        values already claimed by keyframes written just before a crash,
        risking the next video reusing (and colliding with) that range
        (CLAUDE.md T1: resume must not duplicate or collide global_index)."""
        doc = await self.keyframes.find_one(sort=[("global_index", -1)])
        return doc["global_index"] if doc else None

    async def upsert_asr_run(self, doc: AsrRunDocument) -> None:
        """Insert-or-replace one video's ASR run record, keyed by `video_id`
        — the resume/idempotency mechanism for `scripts/run_asr.py` (mirrors
        `upsert_video`)."""
        await self.asr_runs.replace_one(
            {"video_id": doc.video_id}, _asr_run_document_to_dict(doc), upsert=True
        )

    async def get_asr_run(self, video_id: str) -> dict[str, Any] | None:
        return await self.asr_runs.find_one({"video_id": video_id})

    async def close(self) -> None:
        await self._client.close()


def _asr_run_document_to_dict(doc: AsrRunDocument) -> dict[str, Any]:
    return {
        "video_id": doc.video_id,
        "asr_status": doc.asr_status,
        "num_segments": doc.num_segments,
        "model_id": doc.model_id,
        "fps": doc.fps,
    }


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


def _normalize_title(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text).lower()
    without_marks = "".join(
        char for char in normalized if unicodedata.category(char) != "Mn"
    ).replace("đ", "d")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", without_marks)).strip()


def _title_tokens(text: str) -> list[str]:
    return [token for token in _normalize_title(text).split() if token]
