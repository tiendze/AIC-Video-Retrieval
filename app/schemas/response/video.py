"""Response shape for `GET /api/videos/{id}` and `/api/videos/{id}/keyframes`
(plan §P1.6 backend gap 1.4) -- `watch_url` backs the frontend's "Play
YouTube" button, `s_v`/`e_v` are the DANTE-required global_index bounds
(CLAUDE.md §3), and the keyframe list's `thumbnail_url`/`image_url` let the
frontend fetch images without knowing the on-disk path convention."""

from __future__ import annotations

from pydantic import BaseModel


class VideoResponse(BaseModel):
    video_id: str
    watch_url: str
    title: str
    author: str
    publish_date: str
    duration: int
    fps: float
    num_keyframes: int
    s_v: int
    e_v: int


class VideoKeyframeResponse(BaseModel):
    keyframe_id: str
    frame_idx: int
    timestamp: float
    global_index: int
    thumbnail_url: str
    image_url: str


class VideoKeyframesResponse(BaseModel):
    keyframes: list[VideoKeyframeResponse]
