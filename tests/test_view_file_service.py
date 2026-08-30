from __future__ import annotations

from typing import Any

from app.services.view_file_service import (
    TIMESTAMP_UNAVAILABLE,
    YOUTUBE_NOT_AVAILABLE,
    ViewFileService,
    youtube_video_id_from_url,
)


class _FakeMongoStore:
    def __init__(
        self,
        *,
        videos_by_id: dict[str, dict[str, Any]],
        keyframes_by_video: dict[str, list[dict[str, Any]]],
    ) -> None:
        self._videos_by_id = videos_by_id
        self._keyframes_by_video = keyframes_by_video

    async def get_video(self, video_id: str) -> dict[str, Any] | None:
        return self._videos_by_id.get(video_id)

    async def get_keyframes_for_video(self, video_id: str) -> list[dict[str, Any]]:
        return self._keyframes_by_video.get(video_id, [])


def _keyframe(frame_idx: int) -> dict[str, Any]:
    return {
        "keyframe_id": f"L21_V001_f{frame_idx}",
        "video_id": "L21_V001",
        "frame_idx": frame_idx,
        "timestamp": frame_idx / 25,
        "global_index": frame_idx,
    }


async def test_frame_resolution_uses_exact_local_keyframe_only() -> None:
    store = _FakeMongoStore(
        videos_by_id={
            "L21_V001": {
                "video_id": "L21_V001",
                "fps": 25,
                "watch_url": "https://youtube.com/watch?v=abc123",
            }
        },
        keyframes_by_video={"L21_V001": [_keyframe(frame_idx) for frame_idx in [0, 10, 20, 30]]},
    )
    service = ViewFileService(mongo_store=store)

    frame = await service.frame_resolution("L21_V001", 10)

    assert frame.video_id == "L21_V001"
    assert frame.frame_idx == 10
    assert frame.source == "local"
    assert frame.thumbnail_url == "/api/keyframes/L21_V001_f10/thumbnail"
    assert frame.image_url == "/api/keyframes/L21_V001_f10/image"


async def test_frame_resolution_uses_youtube_when_exact_keyframe_is_missing() -> None:
    store = _FakeMongoStore(
        videos_by_id={
            "L21_V001": {
                "video_id": "L21_V001",
                "fps": 25,
                "watch_url": "https://youtube.com/watch?v=abc123",
            }
        },
        keyframes_by_video={"L21_V001": [_keyframe(frame_idx) for frame_idx in [20, 40]]},
    )
    service = ViewFileService(mongo_store=store)

    frame = await service.frame_resolution("L21_V001", 10)

    assert frame.source == "youtube"
    assert frame.youtube_url == "https://youtube.com/watch?v=abc123"
    assert frame.timestamp_seconds == 0.4
    assert frame.thumbnail_url is None


async def test_frame_resolution_marks_missing_youtube_unavailable() -> None:
    store = _FakeMongoStore(
        videos_by_id={
            "L21_V001": {
                "video_id": "L21_V001",
                "fps": 25,
                "watch_url": "https://example.test/watch?v=abc123",
            }
        },
        keyframes_by_video={"L21_V001": [_keyframe(frame_idx) for frame_idx in [20, 40]]},
    )
    service = ViewFileService(mongo_store=store)

    frame = await service.frame_resolution("L21_V001", 10)

    assert frame.source == "unavailable"
    assert frame.error_message == YOUTUBE_NOT_AVAILABLE


async def test_frame_resolution_keeps_youtube_state_when_fps_is_missing() -> None:
    store = _FakeMongoStore(
        videos_by_id={
            "L21_V001": {
                "video_id": "L21_V001",
                "fps": 0,
                "watch_url": "https://youtu.be/abc123",
            }
        },
        keyframes_by_video={"L21_V001": []},
    )
    service = ViewFileService(mongo_store=store)

    frame = await service.frame_resolution("L21_V001", 10)

    assert frame.source == "youtube"
    assert frame.timestamp_seconds is None
    assert frame.error_message == TIMESTAMP_UNAVAILABLE


def test_youtube_video_id_from_url_accepts_common_youtube_formats() -> None:
    assert youtube_video_id_from_url("https://youtube.com/watch?v=abc123") == "abc123"
    assert youtube_video_id_from_url("https://youtu.be/abc123?t=5") == "abc123"
    assert youtube_video_id_from_url("https://www.youtube.com/embed/abc123") == "abc123"
    assert youtube_video_id_from_url("https://example.test/watch?v=abc123") is None
