"""`POST /api/quest/image-candidates` and `POST /api/quest/image-search`
(plan §P2.5, PRD §F5) -- QUEST branch 2: resolve an Out-of-Knowledge query
by letting the operator pick a web sample image, then running
image-to-image search on it. Thin routers: validate the request, delegate
to `QuestService`, map `QuestServiceError` onto the unified
`{detail, error_code, request_id}` schema
(.claude/rules/ecc/python/fastapi.md, CLAUDE.md §7). No business logic
here -- see app/services/quest_service.py for the SSRF guard, ddgs call,
and fallback semantics.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.errors import APIError
from app.dependencies import get_quest_service
from app.schemas.request.quest import ImageCandidatesRequest, ImageSearchRequest
from app.schemas.response.quest import (
    ImageCandidateResponse,
    ImageCandidatesResponse,
    ImageSearchResponse,
)
from app.schemas.response.search import SearchHitResponse
from app.services.quest_service import QuestService, QuestServiceError
from app.services.search_service import SearchHit

router = APIRouter()


def _to_hit_response(hit: SearchHit) -> SearchHitResponse:
    return SearchHitResponse(
        keyframe_id=hit.keyframe_id,
        video_id=hit.video_id,
        score=hit.score,
        frame_idx=hit.frame_idx,
        timestamp=hit.timestamp,
        thumbnail_path=hit.thumbnail_path,
        thumbnail_url=f"/api/keyframes/{hit.keyframe_id}/thumbnail",
        image_url=f"/api/keyframes/{hit.keyframe_id}/image",
    )


@router.post("/api/quest/image-candidates", response_model=ImageCandidatesResponse)
async def image_candidates(
    body: ImageCandidatesRequest, service: QuestService = Depends(get_quest_service)
) -> ImageCandidatesResponse:
    result = await service.find_image_candidates(body.query, body.top_k)
    return ImageCandidatesResponse(
        status=result.status,
        candidate_set_id=result.candidate_set_id,
        candidates=[
            ImageCandidateResponse(
                candidate_id=c.candidate_id, thumbnail_url=c.thumbnail_url, title=c.title
            )
            for c in result.candidates
        ],
        fallback_hits=[_to_hit_response(hit) for hit in result.fallback_hits],
    )


@router.post("/api/quest/image-search", response_model=ImageSearchResponse)
async def image_search(
    body: ImageSearchRequest, service: QuestService = Depends(get_quest_service)
) -> ImageSearchResponse:
    try:
        result = await service.search_by_candidate(
            body.candidate_set_id, body.candidate_id, body.top_k
        )
    except QuestServiceError as exc:
        raise APIError(
            detail=str(exc), error_code=exc.error_code, status_code=exc.status_code
        ) from exc

    return ImageSearchResponse(
        status=result.status, hits=[_to_hit_response(hit) for hit in result.hits]
    )
