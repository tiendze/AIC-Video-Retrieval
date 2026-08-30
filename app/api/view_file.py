from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.errors import APIError
from app.dependencies import get_view_file_service
from app.schemas.response.view_file import ViewFileFrameResolutionResponse
from app.services.view_file_service import ViewFileService, ViewFileServiceError

router = APIRouter()


@router.get("/api/view-file/frame-resolution", response_model=ViewFileFrameResolutionResponse)
async def get_frame_resolution(
    video_id: str,
    frame_idx: int,
    service: ViewFileService = Depends(get_view_file_service),
) -> ViewFileFrameResolutionResponse:
    try:
        return await service.frame_resolution(video_id, frame_idx)
    except ViewFileServiceError as exc:
        raise APIError(
            detail=exc.detail, error_code=exc.error_code, status_code=400
        ) from exc
