"""Integration tests for MongoStore — requires the docker-compose stack (P0.2) running.

docker compose up -d
pytest -m integration tests/integration/test_mongo_client.py -v
"""

from __future__ import annotations

import dataclasses
import os
import uuid
from collections.abc import AsyncIterator

import pytest
from pymongo import AsyncMongoClient
from pymongo.errors import BulkWriteError

from app.stores.mongo_client import KeyframeDocument, MongoStore, VideoDocument

pytestmark = pytest.mark.integration

# See test_milvus_client.py for why this is overridable via env var.
_MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")


@pytest.fixture()
async def store() -> AsyncIterator[MongoStore]:
    database_name = f"test_{uuid.uuid4().hex[:8]}"
    created = MongoStore(uri=_MONGO_URI, database_name=database_name)
    await created.ensure_indexes()
    yield created

    cleanup_client: AsyncMongoClient = AsyncMongoClient(_MONGO_URI)
    await cleanup_client.drop_database(database_name)
    await cleanup_client.close()
    await created.close()


def _keyframe(keyframe_id: str, video_id: str = "video1") -> KeyframeDocument:
    return KeyframeDocument(
        keyframe_id=keyframe_id,
        video_id=video_id,
        shot_id=f"{video_id}_shot0",
        frame_idx=0,
        timestamp=0.0,
        global_index=0,
        image_path=f"/data/{keyframe_id}.jpg",
        thumbnail_path=f"/data/{keyframe_id}_thumb.jpg",
    )


def _video(video_id: str = "video1", ingest_status: str = "pending") -> VideoDocument:
    return VideoDocument(
        video_id=video_id,
        s_v=0,
        e_v=3,
        duration=120,
        fps=25.0,
        num_keyframes=4,
        ingest_status=ingest_status,
        author="channel_a",
        batch="L21",
        title="Sample title",
        publish_date="01/01/2026",
        watch_url="https://example.test/video1",
        channel_id="channel_a_id",
    )


async def test_bulk_insert_keyframes_inserts_new_documents(store: MongoStore) -> None:
    inserted = await store.bulk_insert_keyframes([_keyframe("video1_0"), _keyframe("video1_1")])

    assert inserted == 2
    assert await store.keyframes.count_documents({}) == 2


async def test_duplicate_keyframe_id_does_not_create_second_record(store: MongoStore) -> None:
    await store.bulk_insert_keyframes([_keyframe("video1_0")])

    inserted_on_rerun = await store.bulk_insert_keyframes([_keyframe("video1_0")])

    assert inserted_on_rerun == 0
    assert await store.keyframes.count_documents({"keyframe_id": "video1_0"}) == 1


async def test_duplicate_is_skipped_without_blocking_other_rows_in_same_batch(
    store: MongoStore,
) -> None:
    await store.bulk_insert_keyframes([_keyframe("video1_0")])

    inserted = await store.bulk_insert_keyframes([_keyframe("video1_0"), _keyframe("video1_1")])

    assert inserted == 1
    assert await store.keyframes.count_documents({}) == 2


async def test_non_duplicate_write_error_reraises_and_does_not_block_valid_rows(
    store: MongoStore,
) -> None:
    """A real write failure (not a duplicate key) must re-raise and must not
    silently drop other, valid rows in the same `ordered=False` batch."""
    await store.keyframes.database.command(
        "collMod",
        "keyframes",
        validator={
            "$jsonSchema": {
                "bsonType": "object",
                "properties": {"frame_idx": {"bsonType": "int"}},
            }
        },
        validationAction="error",
    )
    valid_doc = _keyframe("video1_valid")
    invalid_doc = dataclasses.replace(_keyframe("video1_invalid"), frame_idx="not-an-int")

    with pytest.raises(BulkWriteError):
        await store.bulk_insert_keyframes([valid_doc, invalid_doc])

    assert await store.keyframes.count_documents({"keyframe_id": "video1_valid"}) == 1
    assert await store.keyframes.count_documents({"keyframe_id": "video1_invalid"}) == 0


async def test_upsert_video_then_get_video_returns_latest_ingest_status(
    store: MongoStore,
) -> None:
    await store.upsert_video(_video(ingest_status="pending"))
    await store.upsert_video(_video(ingest_status="complete"))

    video = await store.get_video("video1")

    assert video is not None
    assert video["ingest_status"] == "complete"
    assert await store.videos.count_documents({}) == 1
