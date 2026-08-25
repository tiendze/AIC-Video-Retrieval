"""Response shape for `POST /api/trake` (candidate-chaining pipeline).

`video_id` / `score` / `keyframes` are unchanged from the DANTE-era shape so the
frontend's submission export (`frontend/src/utils/submissionExport.ts`) keeps
working; everything else is additive and optional.

The extra fields exist so an operator being timed can see WHY a sequence ranks
where it does without reading server logs: `min_score` is the weak link the
ranking is actually built on, `coverage` is how many events cleared the bar, and
per-keyframe `semantic_z` / `kis_rank` separate "the encoder liked this frame"
from "KIS retrieved this frame".

`thumbnail_url` / `image_url` mirror `app/schemas/response/search.py`: the
frontend fetches images through `GET /api/keyframes/{id}/thumbnail` and
`/image`, never a raw filesystem path.
"""

from __future__ import annotations

from pydantic import BaseModel


class TrakeKeyframeResponse(BaseModel):
    keyframe_id: str
    frame_idx: int
    timestamp: float
    thumbnail_url: str
    image_url: str
    # Per-event diagnostics. `score` is what the chain optimized:
    # semantic_z + kis_evidence_weight * kis_evidence.
    score: float | None = None
    semantic_z: float | None = None
    kis_rank: int | None = None


class TrakeVideoResultResponse(BaseModel):
    video_id: str
    score: float
    # Exactly len(events) entries, strictly time-ordered. One video may appear
    # multiple times when several distinct sequences survive sequence-NMS.
    keyframes: list[TrakeKeyframeResponse]
    # Weak-link score: TRAKE is graded per event, so this is the number that
    # predicts whether the whole sequence is right, not `score`'s blend.
    min_score: float | None = None
    mean_score: float | None = None
    coverage: float | None = None
    # Video-level rank score (B8) -- shared by every sequence of this video.
    video_score: float | None = None


class TrakeEventDiagnosticsResponse(BaseModel):
    event_index: int
    kis_candidate_count: int
    kis_video_count: int
    has_text: bool


class TrakeResponse(BaseModel):
    """`results` is already in submission order: round-robin across videos
    (B9), not a flat sort by sequence score.

    `sequences_by_video` is a separate inspection view: for each video that
    produced chains, its own top sequences (before the submission cap), keyed by
    `video_id`. The UI's per-video "Details" button expands into this; these
    rows never consume submission slots."""

    results: list[TrakeVideoResultResponse]
    events: list[TrakeEventDiagnosticsResponse] = []
    candidate_video_count: int = 0
    chained_video_count: int = 0
    dropped_video_count: int = 0
    sequences_by_video: dict[str, list[TrakeVideoResultResponse]] = {}
