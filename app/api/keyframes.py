"""`GET /api/keyframes/{id}/thumbnail` and `/image` (plan §P1.6 backend gap
1.2) -- thin router: delegate to `ThumbnailService`, map
`ThumbnailServiceError` onto the unified `{detail, error_code, request_id}`
schema (.claude/rules/ecc/python/fastapi.md, CLAUDE.md §7). No business
logic here.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse

from app.api.errors import APIError
from app.dependencies import get_thumbnail_service
from app.services.thumbnail_service import ThumbnailService, ThumbnailServiceError

router = APIRouter()


def _to_file_response(path, cache_max_age_seconds: int) -> FileResponse:
    return FileResponse(
        path,
        media_type="image/jpeg",
        headers={"Cache-Control": f"public, max-age={cache_max_age_seconds}, immutable"},
    )


@router.get("/api/keyframes/{keyframe_id}/thumbnail")
async def get_thumbnail(
    keyframe_id: str, service: ThumbnailService = Depends(get_thumbnail_service)
) -> FileResponse:
    try:
        path = await service.get_thumbnail_path(keyframe_id)
    except ThumbnailServiceError as exc:
        raise APIError(detail=exc.detail, error_code=exc.error_code, status_code=404) from exc
    return _to_file_response(path, service.cache_max_age_seconds)


@router.get("/api/keyframes/{keyframe_id}/image")
async def get_image(
    keyframe_id: str, service: ThumbnailService = Depends(get_thumbnail_service)
) -> FileResponse:
    try:
        path = await service.get_image_path(keyframe_id)
    except ThumbnailServiceError as exc:
        raise APIError(detail=exc.detail, error_code=exc.error_code, status_code=404) from exc
    return _to_file_response(path, service.cache_max_age_seconds)
