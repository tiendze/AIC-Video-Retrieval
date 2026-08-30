from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

ViewFileFrameSource = Literal["local", "youtube", "unavailable"]


class ViewFileFrameResolutionResponse(BaseModel):
    video_id: str
    frame_idx: int
    timestamp_seconds: float | None
    source: ViewFileFrameSource
    keyframe_id: str | None = None
    thumbnail_url: str | None = None
    image_url: str | None = None
    youtube_url: str | None = None
    fps: float | None = None
    error_message: str | None = None
