"""Exact frame resolver for the View File workspace.

The frontend owns CSV parsing and layout. This service only answers one
question for each CSV row: does the exact `(video_id, frame_idx)` keyframe
exist locally, or should the UI fall back to a lightweight YouTube/unavailable
state?
"""

from __future__ import annotations

import re
from typing import Any, Protocol
from urllib.parse import parse_qs, urlparse

from app.schemas.response.view_file import ViewFileFrameResolutionResponse

YOUTUBE_NOT_AVAILABLE = "Youtube Not Available"
TIMESTAMP_UNAVAILABLE = "Unable to resolve video timestamp"

_VIDEO_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


class ViewFileServiceError(Exception):
    def __init__(self, detail: str, error_code: str) -> None:
        super().__init__(detail)
        self.detail = detail
        self.error_code = error_code


class _MongoViewFileStore(Protocol):
    async def get_video(self, video_id: str) -> dict[str, Any] | None: ...

    async def get_keyframes_for_video(self, video_id: str) -> list[dict[str, Any]]: ...


def _frame_idx_of(doc: dict[str, Any]) -> int:
    return int(doc["frame_idx"])


def _timestamp_for(frame_idx: int, fps: float | None) -> float | None:
    if fps is None or fps <= 0:
        return None
    return frame_idx / fps


def _coerce_fps(value: Any) -> float | None:
    try:
        fps = float(value)
    except (TypeError, ValueError):
        return None
    return fps if fps > 0 else None


def youtube_video_id_from_url(watch_url: str) -> str | None:
    try:
        parsed = urlparse(watch_url)
    except ValueError:
        return None

    host = parsed.hostname or ""
    host = host.lower().removeprefix("www.").removeprefix("m.")
    if host == "youtu.be":
        return parsed.path.strip("/").split("/")[0] or None

    if host == "youtube-nocookie.com" or host == "youtube.com" or host.endswith(".youtube.com"):
        watch_id = parse_qs(parsed.query).get("v", [None])[0]
        if watch_id:
            return watch_id
        path_parts = [part for part in parsed.path.split("/") if part]
        if len(path_parts) >= 2 and path_parts[0] in {"embed", "shorts", "live"}:
            return path_parts[1]
    return None


def _valid_youtube_url(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    watch_url = value.strip()
    if not watch_url:
        return None
    return watch_url if youtube_video_id_from_url(watch_url) else None


class ViewFileService:
    def __init__(self, mongo_store: _MongoViewFileStore) -> None:
        self._mongo_store = mongo_store

    async def frame_resolution(
        self, video_id: str, frame_idx: int
    ) -> ViewFileFrameResolutionResponse:
        self._validate_request(video_id, frame_idx)

        keyframes = await self._mongo_store.get_keyframes_for_video(video_id)
        video = await self._mongo_store.get_video(video_id)
        fps = _coerce_fps(video.get("fps")) if video is not None else None
        exact_keyframe = next(
            (doc for doc in keyframes if _frame_idx_of(doc) == frame_idx),
            None,
        )
        if exact_keyframe is not None:
            return self._local_keyframe_response(video_id, exact_keyframe, fps)

        youtube_url = _valid_youtube_url(video.get("watch_url")) if video is not None else None
        timestamp = _timestamp_for(frame_idx, fps)
        if youtube_url is not None:
            return ViewFileFrameResolutionResponse(
                video_id=video_id,
                frame_idx=frame_idx,
                timestamp_seconds=timestamp,
                source="youtube",
                youtube_url=youtube_url,
                fps=fps,
                error_message=TIMESTAMP_UNAVAILABLE if timestamp is None else None,
            )

        return ViewFileFrameResolutionResponse(
            video_id=video_id,
            frame_idx=frame_idx,
            timestamp_seconds=timestamp,
            source="unavailable",
            fps=fps,
            error_message=YOUTUBE_NOT_AVAILABLE,
        )

    def _validate_request(self, video_id: str, frame_idx: int) -> None:
        if not _VIDEO_ID_PATTERN.match(video_id):
            raise ViewFileServiceError(
                detail=f"Invalid video_id: {video_id!r}", error_code="invalid_video_id"
            )
        if frame_idx < 0:
            raise ViewFileServiceError(
                detail="frame_idx must be non-negative", error_code="invalid_frame_idx"
            )

    def _local_keyframe_response(
        self, video_id: str, doc: dict[str, Any], fps: float | None
    ) -> ViewFileFrameResolutionResponse:
        frame_idx = _frame_idx_of(doc)
        timestamp = doc.get("timestamp")
        timestamp_seconds = (
            float(timestamp) if timestamp is not None else _timestamp_for(frame_idx, fps)
        )
        keyframe_id = str(doc["keyframe_id"])
        return ViewFileFrameResolutionResponse(
            video_id=video_id,
            frame_idx=frame_idx,
            timestamp_seconds=timestamp_seconds,
            source="local",
            keyframe_id=keyframe_id,
            thumbnail_url=f"/api/keyframes/{keyframe_id}/thumbnail",
            image_url=f"/api/keyframes/{keyframe_id}/image",
            fps=fps,
        )
