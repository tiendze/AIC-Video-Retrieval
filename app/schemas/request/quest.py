"""Request shapes for QUEST branch 2 (plan §P2.5, PRD §F5) --
`POST /api/quest/image-candidates` and `POST /api/quest/image-search`.
`top_k` bounds reuse `app.schemas.request.search`'s `DEFAULT_TOP_K`/
`MAX_TOP_K` (configs/search/search.yaml) -- both endpoints ultimately
return `SearchHit` results through the same `SearchService`, so there is no
separate quest-specific top_k bound (CLAUDE.md §6: no magic numbers)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.schemas.request.search import DEFAULT_TOP_K, MAX_TOP_K


class ImageCandidatesRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=MAX_TOP_K)


class ImageSearchRequest(BaseModel):
    candidate_set_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=MAX_TOP_K)
