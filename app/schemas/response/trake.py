"""Response shape for `POST /api/trake` (plan §P2.2). `thumbnail_url`/
`image_url` mirror `app/schemas/response/search.py`'s convention: the
frontend fetches images through `GET /api/keyframes/{id}/thumbnail`/`/image`,
never a raw filesystem path."""

from __future__ import annotations

from pydantic import BaseModel


class TrakeKeyframeResponse(BaseModel):
    keyframe_id: str
    frame_idx: int
    timestamp: float
    thumbnail_url: str
    image_url: str


class TrakeVideoResultResponse(BaseModel):
    video_id: str
    score: float
    # Exactly len(events) entries, strictly time-ordered (CLAUDE.md §4's
    # DANTE ordering invariant). One video may appear multiple times when
    # multiple sequences clear the request threshold.
    keyframes: list[TrakeKeyframeResponse]


class TrakeResponse(BaseModel):
    results: list[TrakeVideoResultResponse]
