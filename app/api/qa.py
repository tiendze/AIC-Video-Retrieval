"""`POST /api/qa` (plan §P3.2, PRD §F7, architecture.md §3.6/§7) -- VQA
answering grounded in keyframes the system actually retrieved.

Thin router: validate, delegate to `QaService`, shape the response
(.claude/rules/ecc/python/fastapi.md). There is deliberately no
`try/except` here for the VQA branch -- `QaService` already converts a
Gemini outage into a `status="unavailable"` result carrying the keyframe
grid, so this endpoint answers **200 with a grid + notice** rather than a
5xx (plan §P3.2, US-5). A retrieval failure is a different matter and does
propagate to the shared error handler.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.dependencies import get_qa_service
from app.schemas.request.qa import QaRequest
from app.schemas.response.qa import QaResponse
from app.schemas.response.search import SearchHitResponse
from app.services.search_service import SearchHit
from app.services.vqa_service import QaService

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


@router.post("/api/qa", response_model=QaResponse)
async def answer_question(
    body: QaRequest, service: QaService = Depends(get_qa_service)
) -> QaResponse:
    result = await service.answer_question(body.question, body.top_k)
    return QaResponse(
        status=result.status,
        answer=result.answer,
        confidence=result.confidence,
        evidence_keyframe_ids=list(result.evidence_keyframe_ids),
        reasoning=result.reasoning,
        keyframes=[_to_hit_response(hit) for hit in result.keyframes],
        notice=result.notice,
    )
