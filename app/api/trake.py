"""`POST /api/trake` (plan §P2.2, architecture.md §3.5) -- thin router:
validate the request, delegate to `TrakeService`, map its `TrakeVideoResult`
list onto the response schema. No DANTE math here -- that lives in
`app/services/dante_service.py` / `app/services/trake_service.py`.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.errors import APIError
from app.dependencies import get_trake_service
from app.schemas.request.trake import TrakeRequest
from app.schemas.response.trake import (
    TrakeKeyframeResponse,
    TrakeResponse,
    TrakeVideoResultResponse,
)
from app.services.trake_embedding_cache import TrakeEmbeddingCacheError
from app.services.trake_service import TrakeService, TrakeVideoResult

router = APIRouter()


def _to_response(results: list[TrakeVideoResult]) -> TrakeResponse:
    return TrakeResponse(
        results=[
            TrakeVideoResultResponse(
                video_id=result.video_id,
                score=result.score,
                keyframes=[
                    TrakeKeyframeResponse(
                        keyframe_id=kf.keyframe_id,
                        frame_idx=kf.frame_idx,
                        timestamp=kf.timestamp,
                        thumbnail_url=f"/api/keyframes/{kf.keyframe_id}/thumbnail",
                        image_url=f"/api/keyframes/{kf.keyframe_id}/image",
                    )
                    for kf in result.keyframes
                ],
            )
            for result in results
        ]
    )


@router.post("/api/trake", response_model=TrakeResponse)
async def trake(
    body: TrakeRequest, service: TrakeService = Depends(get_trake_service)
) -> TrakeResponse:
    try:
        results = await service.search(
            body.events,
            top_k=body.top_k,
            lambda_penalty=body.lambda_penalty,
            threshold=body.threshold,
        )
    except TrakeEmbeddingCacheError as exc:
        raise APIError(
            detail=str(exc), error_code=exc.error_code, status_code=exc.status_code
        ) from exc
    return _to_response(results)
