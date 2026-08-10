"""`POST /api/search` (plan §P0.8, architecture.md §7) -- thin router:
validate the request, delegate to `SearchService`, map its errors onto the
unified `{detail, error_code, request_id}` schema
(.claude/rules/ecc/python/fastapi.md, CLAUDE.md §7). No business logic here.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.errors import APIError
from app.dependencies import get_search_service
from app.schemas.request.search import SearchRequest
from app.schemas.response.search import SearchHitResponse, SearchResponse
from app.services.search_service import SearchService, SearchServiceError

router = APIRouter()


@router.post("/api/search", response_model=SearchResponse)
async def search(
    body: SearchRequest, service: SearchService = Depends(get_search_service)
) -> SearchResponse:
    try:
        hits = await service.search(body.query, mode=body.mode, top_k=body.top_k)
    except SearchServiceError as exc:
        raise APIError(detail=str(exc), error_code="invalid_search_mode", status_code=400) from exc

    return SearchResponse(
        results=[
            SearchHitResponse(
                keyframe_id=hit.keyframe_id,
                video_id=hit.video_id,
                score=hit.score,
                frame_idx=hit.frame_idx,
                timestamp=hit.timestamp,
                thumbnail_path=hit.thumbnail_path,
            )
            for hit in hits
        ]
    )
