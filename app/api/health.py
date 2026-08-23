"""`GET /health` (plan §P0.8, architecture.md §7) -- checks connectivity to
all three stores at once. Thin router: validate + delegate to
`HealthService`, no business logic here (.claude/rules/ecc/python/fastapi.md).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.dependencies import get_health_service
from app.schemas.response.health import HealthResponse
from app.services.health_service import HealthService

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health(service: HealthService = Depends(get_health_service)) -> HealthResponse:
    report = await service.check()
    return HealthResponse(
        status=report.status,
        milvus_ok=report.milvus_ok,
        elasticsearch_ok=report.elasticsearch_ok,
        mongodb_ok=report.mongodb_ok,
        asr_status=report.asr_status,
        asr_ready=report.asr_ready,
        asr_segment_count=report.asr_segment_count,
    )
