"""Request shape for `POST /api/qa` (plan §P3.2, PRD §F7). `top_k` bounds
reuse `app.schemas.request.search`'s `DEFAULT_TOP_K`/`MAX_TOP_K`
(configs/search/search.yaml) -- the QA grid is the same `SearchHit` grid
`/api/search` returns, so there is no separate QA-specific bound
(CLAUDE.md §6: no magic numbers).

`top_k` sizes the retrieved GRID, not how many keyframes reach the model --
that second cap is `max_keyframes_to_model` in `configs/qa/vqa.yaml`, since
sending 100 images to Gemini for one question is never right.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.schemas.request.search import DEFAULT_TOP_K, MAX_TOP_K


class QaRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=MAX_TOP_K)
