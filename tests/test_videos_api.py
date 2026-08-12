"""RED reproducer for `GET /api/videos/{id}` and `/api/videos/{id}/keyframes`
(plan §P1.6 backend gap 1.4) -- thin wrappers over `MongoStore.get_video()` /
`get_keyframes_for_video()` (both already exist, `app/stores/mongo_client.py`).
`watch_url` here is what the frontend's "Play YouTube" button needs; the
keyframe list backs the neighboring-keyframes panel in the detail view.

A Fake `MongoStore` is injected via `app.dependency_overrides` -- no Docker
needed here; `MongoStore`'s own behavior is covered by tests/test_mongo_client.py.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import videos as videos_api
from app.api.errors import register_error_handling
from app.dependencies import get_mongo_store


class _FakeMongoStore:
    def __init__(
        self,
        videos_by_id: dict[str, dict[str, Any]] | None = None,
        keyframes_by_video: dict[str, list[dict[str, Any]]] | None = None,
    ) -> None:
        self._videos_by_id = videos_by_id or {}
        self._keyframes_by_video = keyframes_by_video or {}
        self.get_video_calls: list[str] = []
        self.get_keyframes_for_video_calls: list[str] = []

    async def get_video(self, video_id: str) -> dict[str, Any] | None:
        self.get_video_calls.append(video_id)
        return self._videos_by_id.get(video_id)

    async def get_keyframes_for_video(self, video_id: str) -> list[dict[str, Any]]:
        self.get_keyframes_for_video_calls.append(video_id)
        return self._keyframes_by_video.get(video_id, [])


def _client(fake_store: _FakeMongoStore) -> TestClient:
    app = FastAPI()
    register_error_handling(app)
    app.include_router(videos_api.router)
    app.dependency_overrides[get_mongo_store] = lambda: fake_store
    return TestClient(app, raise_server_exceptions=False)


_VIDEO_DOC = {
    "video_id": "L21_V001",
    "s_v": 0,
    "e_v": 669,
    "duration": 600,
    "fps": 25.0,
    "num_keyframes": 670,
    "ingest_status": "complete",
    "author": "VTV",
    "batch": "L21",
    "title": "Ban tin thoi su",
    "publish_date": "01/01/2026",
    "watch_url": "https://youtube.com/watch?v=abc123",
    "channel_id": "UCxyz",
}


def test_get_video_returns_watch_url_and_metadata() -> None:
    store = _FakeMongoStore(videos_by_id={"L21_V001": _VIDEO_DOC})
    client = _client(store)

    response = client.get("/api/videos/L21_V001")

    assert response.status_code == 200
    body = response.json()
    assert body["video_id"] == "L21_V001"
    assert body["watch_url"] == "https://youtube.com/watch?v=abc123"
    assert body["s_v"] == 0
    assert body["e_v"] == 669
    assert store.get_video_calls == ["L21_V001"]


def test_get_video_returns_404_for_unknown_video_id() -> None:
    store = _FakeMongoStore(videos_by_id={})
    client = _client(store)

    response = client.get("/api/videos/L99_V999")

    assert response.status_code == 404
    body = response.json()
    assert body["error_code"] == "video_not_found"
    assert "request_id" in body


def test_get_video_keyframes_returns_thumbnail_and_image_urls() -> None:
    store = _FakeMongoStore(
        videos_by_id={"L21_V001": _VIDEO_DOC},
        keyframes_by_video={
            "L21_V001": [
                {
                    "keyframe_id": "L21_V001_f10",
                    "frame_idx": 10,
                    "timestamp": 0.4,
                    "global_index": 0,
                }
            ]
        },
    )
    client = _client(store)

    response = client.get("/api/videos/L21_V001/keyframes")

    assert response.status_code == 200
    keyframes = response.json()["keyframes"]
    assert keyframes == [
        {
            "keyframe_id": "L21_V001_f10",
            "frame_idx": 10,
            "timestamp": 0.4,
            "global_index": 0,
            "thumbnail_url": "/api/keyframes/L21_V001_f10/thumbnail",
            "image_url": "/api/keyframes/L21_V001_f10/image",
        }
    ]


def test_get_video_keyframes_returns_404_for_unknown_video_id() -> None:
    store = _FakeMongoStore(videos_by_id={})
    client = _client(store)

    response = client.get("/api/videos/L99_V999/keyframes")

    assert response.status_code == 404
    assert response.json()["error_code"] == "video_not_found"
    # Never fetches keyframes for a video that doesn't exist (CLAUDE.md §7:
    # fail clearly, don't quietly return an empty list that looks like a
    # video with zero keyframes).
    assert store.get_keyframes_for_video_calls == []
