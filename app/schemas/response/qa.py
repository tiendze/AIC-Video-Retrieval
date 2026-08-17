"""Response shape for `POST /api/qa` (plan §P3.2, PRD §F7, architecture.md
§7). Mirrors `QaResult`'s three-way `status` discriminator:

- `answered` -- the model answered AND every cited keyframe passed the
  grounding gate.
- `insufficient_evidence` -- the model abstained, cited nothing, or cited a
  keyframe it was never shown (gate rejection). `answer` carries the
  abstention text so the UI has something to render.
- `unavailable` -- Gemini was unreachable. `answer` is null, `notice`
  explains it in Vietnamese, and `keyframes` is still populated.

`keyframes` is non-empty in every status where retrieval found anything --
the operator always gets a grid to work from (US-5), which is the whole
point of the degraded path.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from app.schemas.response.search import SearchHitResponse


class QaResponse(BaseModel):
    status: Literal["answered", "insufficient_evidence", "unavailable"]
    answer: str | None
    confidence: float | None
    evidence_keyframe_ids: list[str]
    reasoning: str | None
    keyframes: list[SearchHitResponse]
    notice: str | None
