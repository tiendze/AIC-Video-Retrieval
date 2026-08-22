"""Response shapes for QUEST branch 2 (plan §P2.5, PRD §F5). `candidates`
carry only `candidate_id`/`thumbnail_url`/`title` -- the real image URL
never leaves the server (SSRF guard, app/services/quest_service.py). The
`status` discriminator mirrors `QuestService`'s own fallback semantics:
"candidates"/"fallback" for image-candidates, "ok"/"fallback" for
image-search."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

class ImageCandidateResponse(BaseModel):
    candidate_id: str
    thumbnail_url: str
    title: str


class QuestSearchHitResponse(BaseModel):
    keyframe_id: str
    video_id: str
    score: float
    frame_idx: int
    timestamp: float
    thumbnail_path: str
    thumbnail_url: str
    image_url: str


class ImageCandidatesResponse(BaseModel):
    status: Literal["candidates", "fallback"]
    candidate_set_id: str | None
    candidates: list[ImageCandidateResponse]
    fallback_hits: list[QuestSearchHitResponse]


class ImageSearchResponse(BaseModel):
    status: Literal["ok", "fallback"]
    hits: list[QuestSearchHitResponse]


class QueryRewriteResponse(BaseModel):
    status: Literal["rewritten", "fallback"]
    original_query: str
    rewritten_query: str | None
