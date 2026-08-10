"""Response shape for `POST /api/search` (plan §P0.8) -- every field the
P0.8 verify step requires: score, video_id, timestamp, keyframe_id,
thumbnail_path, plus frame_idx (CLAUDE.md §3: mandatory on every keyframe
result)."""

from __future__ import annotations

from pydantic import BaseModel


class SearchHitResponse(BaseModel):
    keyframe_id: str
    video_id: str
    score: float
    frame_idx: int
    timestamp: float
    thumbnail_path: str


class SearchResponse(BaseModel):
    results: list[SearchHitResponse]
