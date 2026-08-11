"""`POST /api/search` (plan §P0.8, architecture.md §7) and `POST
/api/search/image` (plan §P1.1) -- thin routers: validate the request,
delegate to `SearchService`, map its errors onto the unified
`{detail, error_code, request_id}` schema
(.claude/rules/ecc/python/fastapi.md, CLAUDE.md §7). No business logic here.

`/api/search/image` validates MIME type and size **before** decoding
(CLAUDE.md §7: uploaded images -- limit size, check MIME, never store
permanently) -- an unreadable/corrupt image is a 400, not a 500, since it's
an expected caller error, not a bug. The app itself never writes the upload
to disk; `register_upload_guard()` rejects an oversized body by its declared
Content-Length before Starlette's multipart parser buffers it, since
`python-multipart` has no per-request total-size cap of its own and would
otherwise spool an arbitrarily large file part into memory/a temp file
before this route's own `MAX_UPLOAD_SIZE_BYTES` check ever runs (found in
code review 2026-08-11: a well-behaved but huge upload would otherwise be
fully received first) -- this is a best-effort guard against a truthful
Content-Length, not a hard bound against a client that lies about it.
"""

from __future__ import annotations

import uuid
from io import BytesIO

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from app.api.errors import APIError
from app.dependencies import get_search_service
from app.schemas.request.search import (
    ALLOWED_UPLOAD_MIME_TYPES,
    DEFAULT_TOP_K,
    MAX_TOP_K,
    MAX_UPLOAD_SIZE_BYTES,
    SearchRequest,
)
from app.schemas.response.error import ErrorResponse
from app.schemas.response.search import SearchHitResponse, SearchResponse
from app.services.search_service import SearchHit, SearchService, SearchServiceError

router = APIRouter()

_UPLOAD_GUARDED_PATH = "/api/search/image"

# Must stay in sync with `ALLOWED_UPLOAD_MIME_TYPES` (configs/search/search.yaml):
# `Image.open()` detects format from magic bytes, independent of the
# caller-supplied `Content-Type` header, so the header alone cannot be
# trusted to mean the decoded bytes are actually one of these formats.
_MIME_TO_PIL_FORMAT = {
    "image/jpeg": "JPEG",
    "image/png": "PNG",
    "image/webp": "WEBP",
}


def _to_response(hits: list[SearchHit]) -> SearchResponse:
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


@router.post("/api/search", response_model=SearchResponse)
async def search(
    body: SearchRequest, service: SearchService = Depends(get_search_service)
) -> SearchResponse:
    try:
        hits = await service.search(body.query, mode=body.mode, top_k=body.top_k)
    except SearchServiceError as exc:
        raise APIError(detail=str(exc), error_code="invalid_search_mode", status_code=400) from exc

    return _to_response(hits)


@router.post("/api/search/image", response_model=SearchResponse)
async def search_by_image(
    file: UploadFile = File(...),
    top_k: int = Form(default=DEFAULT_TOP_K, ge=1, le=MAX_TOP_K),
    service: SearchService = Depends(get_search_service),
) -> SearchResponse:
    if file.content_type not in ALLOWED_UPLOAD_MIME_TYPES:
        raise APIError(
            detail=f"Unsupported image type: {file.content_type!r}",
            error_code="unsupported_media_type",
            status_code=415,
        )

    contents = await file.read()
    if len(contents) > MAX_UPLOAD_SIZE_BYTES:
        raise APIError(
            detail=f"Upload exceeds the {MAX_UPLOAD_SIZE_BYTES}-byte limit",
            error_code="upload_too_large",
            status_code=413,
        )

    try:
        image = Image.open(BytesIO(contents))
        # Force full decode now (truncated file -> OSError) so a bad upload
        # fails here with a 400, not mid-encode with a 500; also bounds
        # decompression-bomb-style images (huge declared dimensions in a
        # tiny file) that would otherwise blow up memory during encoding.
        image.load()
    except (OSError, Image.DecompressionBombError) as exc:
        raise APIError(
            detail="Uploaded file is not a readable image",
            error_code="invalid_image",
            status_code=400,
        ) from exc

    if image.format != _MIME_TO_PIL_FORMAT[file.content_type]:
        raise APIError(
            detail=f"Declared type {file.content_type!r} does not match the uploaded file's "
            f"actual format {image.format!r}",
            error_code="unsupported_media_type",
            status_code=415,
        )

    hits = await service.search_by_image(image, top_k=top_k)
    return _to_response(hits)


async def _content_length_guard(request: Request, call_next):
    if request.url.path == _UPLOAD_GUARDED_PATH:
        content_length = request.headers.get("content-length")
        if content_length is not None and int(content_length) > MAX_UPLOAD_SIZE_BYTES:
            request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
            body = ErrorResponse(
                detail=f"Upload exceeds the {MAX_UPLOAD_SIZE_BYTES}-byte limit",
                error_code="upload_too_large",
                request_id=request_id,
            )
            return JSONResponse(
                status_code=413,
                content=body.model_dump(),
                headers={"X-Request-ID": request_id},
            )
    return await call_next(request)


def register_upload_guard(app) -> None:
    """Rejects an oversized `/api/search/image` request by Content-Length
    before Starlette's multipart parser runs (see module docstring) -- must
    be registered on every app instance that mounts `router`, mirroring
    `register_error_handling()`."""
    app.middleware("http")(_content_length_guard)
