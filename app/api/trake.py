"""`POST /api/trake` -- thin router: validate, delegate to `TrakeService`, map
its result onto the response schema. No pipeline logic here; the chaining math
lives in `app/services/trake_chain.py` and the orchestration in
`app/services/trake_service.py`.
"""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Depends

from app.api.errors import APIError
from app.dependencies import get_trake_service
from app.schemas.request.trake import TrakeEventRequest, TrakeRequest
from app.schemas.response.trake import (
    TrakeEventDiagnosticsResponse,
    TrakeKeyframeResponse,
    TrakeResponse,
    TrakeVideoResultResponse,
)
from app.services.trake_chain import ChainCandidate, TrakeChainError
from app.services.trake_embedding_cache import TrakeEmbeddingCacheError
from app.services.trake_service import (
    TrakeEventQuery,
    TrakeSearchResult,
    TrakeService,
    TrakeServiceError,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _to_event_query(event: TrakeEventRequest) -> TrakeEventQuery:
    return TrakeEventQuery(
        text_queries=tuple(event.text_queries),
        text_top_k=event.text_top_k,
        local_phrases=tuple(event.local_phrases),
        ocr_query=event.ocr_query,
        ocr_top_k=event.ocr_top_k,
        title_query=event.title_query,
        title_top_k=event.title_top_k,
        asr_queries=tuple(event.asr_queries),
        asr_match_mode=event.asr_match_mode,
        asr_video_top_k=event.asr_video_top_k,
    )


def _to_result_response(
    candidate: ChainCandidate, video_score: float | None
) -> TrakeVideoResultResponse:
    return TrakeVideoResultResponse(
        video_id=candidate.video_id,
        score=candidate.score,
        min_score=candidate.min_score,
        mean_score=candidate.mean_score,
        coverage=candidate.coverage,
        video_score=video_score,
        keyframes=[
            TrakeKeyframeResponse(
                keyframe_id=frame.keyframe_id,
                frame_idx=frame.frame_idx,
                timestamp=frame.timestamp,
                thumbnail_url=f"/api/keyframes/{frame.keyframe_id}/thumbnail",
                image_url=f"/api/keyframes/{frame.keyframe_id}/image",
                score=frame.score,
                semantic_z=frame.semantic_z,
                kis_rank=frame.kis_rank,
                local_score=frame.local_score,
                local_z=frame.local_z,
            )
            for frame in candidate.frames
        ],
    )


def _to_response(result: TrakeSearchResult) -> TrakeResponse:
    return TrakeResponse(
        results=[
            _to_result_response(
                candidate,
                (
                    result.video_scores[candidate.video_id].score
                    if candidate.video_id in result.video_scores
                    else None
                ),
            )
            for candidate in result.candidates
        ],
        events=[
            TrakeEventDiagnosticsResponse(
                event_index=event.event_index,
                kis_candidate_count=event.kis_candidate_count,
                kis_video_count=event.kis_video_count,
                has_text=event.has_text,
                text_truncated=event.text_truncated,
            )
            for event in result.events
        ],
        candidate_video_count=result.candidate_video_count,
        chained_video_count=result.chained_video_count,
        dropped_video_count=result.dropped_video_count,
        local_rerank_video_count=result.local_rerank_video_count,
        local_missing_frame_count=result.local_missing_frame_count,
        sequences_by_video={
            video_id: [
                _to_result_response(
                    candidate,
                    (
                        result.video_scores[video_id].score
                        if video_id in result.video_scores
                        else None
                    ),
                )
                for candidate in sequences
            ]
            for video_id, sequences in result.sequences_by_video.items()
        },
    )


@router.post("/api/trake", response_model=TrakeResponse)
async def trake(
    body: TrakeRequest, service: TrakeService = Depends(get_trake_service)
) -> TrakeResponse:
    start = time.perf_counter()
    try:
        result = await service.search(
            [_to_event_query(event) for event in body.events],
            top_k=body.top_k,
            threshold=body.threshold,
        )
    except (TrakeServiceError, TrakeChainError) as exc:
        raise APIError(
            detail=str(exc), error_code=exc.error_code, status_code=exc.status_code
        ) from exc
    except TrakeEmbeddingCacheError as exc:
        raise APIError(
            detail=str(exc), error_code=exc.error_code, status_code=exc.status_code
        ) from exc

    latency_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "trake request completed: latency_ms=%.1f events=%d num_results=%d "
        "candidate_videos=%d dropped_videos=%d",
        latency_ms,
        len(body.events),
        len(result.candidates),
        result.candidate_video_count,
        result.dropped_video_count,
    )
    return _to_response(result)
