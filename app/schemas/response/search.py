"""Response shape for `POST /api/search` (plan §P0.8) -- every field the
P0.8 verify step requires: score, video_id, timestamp, keyframe_id,
thumbnail_path, plus frame_idx (CLAUDE.md §3: mandatory on every keyframe
result).

`thumbnail_url`/`image_url` (plan §P1.6) are added on top of the original
`thumbnail_path` -- the frontend fetches images through
`GET /api/keyframes/{id}/thumbnail`/`/image`, never by reading a raw
filesystem path (which may not even exist client-side, and differs between
the two ingest paths' directory conventions). `thumbnail_path` is kept as-is
for existing callers/tests/debugging."""

from __future__ import annotations

from pydantic import BaseModel


class SearchHitResponse(BaseModel):
    keyframe_id: str
    video_id: str
    score: float
    frame_idx: int
    timestamp: float
    thumbnail_path: str
    thumbnail_url: str
    image_url: str


class SearchResponse(BaseModel):
    results: list[SearchHitResponse]
