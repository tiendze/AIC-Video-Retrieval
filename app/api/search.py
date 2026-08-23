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

import logging
import time
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
    KisCandidateRequest,
    KisRerankRequest,
    SearchRequest,
)
from app.schemas.response.error import ErrorResponse
from app.schemas.response.search import (
    AbeModalityScoreResponse,
    AbePairScoreResponse,
    AbePatchHeatmapResponse,
    AbePatchHitResponse,
    AbeQueryScoreResponse,
    KisCandidateHitResponse,
    KisCandidateResponse,
    KisRerankResponse,
    KisSearchHitResponse,
    KisSearchResponse,
    OcrModalityScoreResponse,
    OcrQueryScoreResponse,
    SearchHitResponse,
    SearchModalitiesResponse,
    SearchResponse,
    Stage1OcrScoreResponse,
    Stage1RetrievalResponse,
    Stage1TextQueryScoreResponse,
    Stage1TextScoreResponse,
    Stage1TitleScoreResponse,
    TextModalityScoreResponse,
    TextQueryScoreResponse,
)
from app.services.kis_scoring import (
    AbeModalityScore,
    AbePair,
    KisQuery,
    OcrModalityScore,
    QueryType,
    Stage1RetrievalScore,
    TextModalityScore,
)
from app.services.search_service import (
    KisCandidatePool,
    SearchHit,
    SearchService,
    SearchServiceError,
)

logger = logging.getLogger(__name__)

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
                fused_image_score=hit.fused_image_score,
                fused_image_rank=hit.fused_image_rank,
                video_title=hit.video_title,
                frame_idx=hit.frame_idx,
                timestamp=hit.timestamp,
                thumbnail_path=hit.thumbnail_path,
                # Derived from keyframe_id alone (plan §P1.6) -- the frontend
                # never needs to know the on-disk path convention.
                thumbnail_url=f"/api/keyframes/{hit.keyframe_id}/thumbnail",
                image_url=f"/api/keyframes/{hit.keyframe_id}/image",
            )
            for hit in hits
        ]
    )


def _to_kis_response(hits: list[SearchHit]) -> KisSearchResponse:
    return KisSearchResponse(
        results=[
            KisSearchHitResponse(
                keyframe_id=hit.keyframe_id,
                video_id=hit.video_id,
                score=hit.score,
                fused_image_score=hit.fused_image_score,
                fused_image_rank=hit.fused_image_rank,
                video_title=hit.video_title,
                frame_idx=hit.frame_idx,
                timestamp=hit.timestamp,
                thumbnail_path=hit.thumbnail_path,
                thumbnail_url=f"/api/keyframes/{hit.keyframe_id}/thumbnail",
                image_url=f"/api/keyframes/{hit.keyframe_id}/image",
                final_score=hit.final_score if hit.final_score is not None else hit.score,
                active_modalities=hit.active_modalities
                if hit.active_modalities is not None
                else len(hit.modalities or {}),
                modalities=_to_modalities_response(hit.modalities) or SearchModalitiesResponse(),
            )
            for hit in hits
        ]
    )


def _to_kis_candidate_response(pool: KisCandidatePool) -> KisCandidateResponse:
    return KisCandidateResponse(
        candidate_ids=pool.candidate_ids,
        candidate_pool_size=pool.candidate_pool_size,
        text_candidate_count=pool.text_candidate_count,
        ocr_candidate_count=pool.ocr_candidate_count,
        title_candidate_count=pool.title_candidate_count,
        rrf_k=pool.rrf_k,
        results=[
            KisCandidateHitResponse(
                keyframe_id=hit.keyframe_id,
                video_id=hit.video_id,
                score=hit.score,
                fused_image_score=hit.fused_image_score,
                fused_image_rank=hit.fused_image_rank,
                video_title=hit.video_title,
                frame_idx=hit.frame_idx,
                timestamp=hit.timestamp,
                thumbnail_path=hit.thumbnail_path,
                thumbnail_url=f"/api/keyframes/{hit.keyframe_id}/thumbnail",
                image_url=f"/api/keyframes/{hit.keyframe_id}/image",
                stage1=_to_stage1_response(hit.stage1),
            )
            for hit in pool.results
        ],
    )


def _to_rerank_response(
    hits: list[SearchHit], candidate_pool_size: int, final_top_k: int
) -> KisRerankResponse:
    return KisRerankResponse(
        candidate_pool_size=candidate_pool_size,
        final_top_k=final_top_k,
        results=_to_kis_response(hits).results,
    )


def _to_stage1_response(stage1: Stage1RetrievalScore | None) -> Stage1RetrievalResponse:
    if stage1 is None:
        return Stage1RetrievalResponse(source_modalities=[])
    return Stage1RetrievalResponse(
        source_modalities=stage1.source_modalities,
        fused_image_score=stage1.fused_image_score,
        fused_image_rank=stage1.fused_image_rank,
        text=(
            Stage1TextScoreResponse(
                query=stage1.text.query,
                raw_cosine=stage1.text.raw_cosine,
                rank=stage1.text.rank,
            )
            if stage1.text is not None
            else None
        ),
        text_queries=(
            [
                Stage1TextQueryScoreResponse(
                    query_index=query.query_index,
                    query=query.query,
                    raw_cosine=query.raw_cosine,
                    original_rank=query.original_rank,
                    rank=query.rank,
                )
                for query in stage1.text_queries
            ]
            if stage1.text_queries
            else None
        ),
        ocr=(
            Stage1OcrScoreResponse(
                query=stage1.ocr.query,
                raw_bm25=stage1.ocr.raw_bm25,
                bm25_reference=stage1.ocr.bm25_reference,
                bm25_normalized=stage1.ocr.bm25_normalized,
                coverage=stage1.ocr.coverage,
                query_score=stage1.ocr.query_score,
                rank=stage1.ocr.rank,
            )
            if stage1.ocr is not None
            else None
        ),
        title=(
            Stage1TitleScoreResponse(
                query=stage1.title.query,
                video_title=stage1.title.video_title,
                title_score=stage1.title.title_score,
                rank=stage1.title.rank,
                match_type=stage1.title.match_type,
                matched_text=stage1.title.matched_text,
                matched_tokens=list(stage1.title.matched_tokens),
            )
            if stage1.title is not None
            else None
        ),
    )


def _to_modalities_response(
    modalities: dict[str, TextModalityScore | OcrModalityScore | AbeModalityScore] | None,
) -> SearchModalitiesResponse | None:
    if modalities is None:
        return None
    text = modalities.get("text")
    ocr = modalities.get("ocr")
    abe = modalities.get("abe")
    return SearchModalitiesResponse(
        text=(
            TextModalityScoreResponse(
                geometric_mean=text.geometric_mean,
                score=text.score,
                queries=[
                    TextQueryScoreResponse(
                        query_index=query.query_index,
                        query=query.query,
                        raw_cosine=query.raw_cosine,
                        rank=query.rank,
                        rank_percentile=query.rank_percentile,
                    )
                    for query in text.queries
                ],
            )
            if isinstance(text, TextModalityScore)
            else None
        ),
        ocr=(
            OcrModalityScoreResponse(
                geometric_mean=ocr.geometric_mean,
                score=ocr.score,
                queries=[
                    OcrQueryScoreResponse(
                        query_index=query.query_index,
                        query=query.query,
                        raw_bm25=query.raw_bm25,
                        bm25_reference=query.bm25_reference,
                        bm25_normalized=query.bm25_normalized,
                        coverage=query.coverage,
                        query_score=query.query_score,
                        rank=query.rank,
                        rank_percentile=query.rank_percentile,
                    )
                    for query in ocr.queries
                ],
            )
            if isinstance(ocr, OcrModalityScore)
            else None
        ),
        abe=(
            AbeModalityScoreResponse(
                geometric_mean=abe.geometric_mean,
                score=abe.score,
                queries=[
                    AbeQueryScoreResponse(
                        query_index=query.query_index,
                        pairs=[
                            AbePairScoreResponse(
                                attribute=pair.attribute,
                                object=pair.object,
                                positive_attributes=pair.positive_attributes,
                                negative_attributes=pair.negative_attributes,
                            )
                            for pair in query.pairs
                        ],
                        base_local=query.base_local,
                        refined_local=query.refined_local,
                        binding_delta=query.binding_delta,
                        local_score=query.local_score,
                        global_abe=query.global_abe,
                        raw_score=query.raw_score,
                        rank=query.rank,
                        rank_percentile=query.rank_percentile,
                        patch_heatmap=(
                            AbePatchHeatmapResponse(
                                patch_count=query.patch_heatmap.patch_count,
                                grid_width=query.patch_heatmap.grid_width,
                                grid_height=query.patch_heatmap.grid_height,
                                base_top_patches=[
                                    AbePatchHitResponse(
                                        label=patch.label,
                                        patch_index=patch.patch_index,
                                        score=patch.score,
                                    )
                                    for patch in query.patch_heatmap.base_top_patches
                                ],
                                refined_top_patches=[
                                    AbePatchHitResponse(
                                        label=patch.label,
                                        patch_index=patch.patch_index,
                                        score=patch.score,
                                    )
                                    for patch in query.patch_heatmap.refined_top_patches
                                ],
                            )
                            if query.patch_heatmap is not None
                            else None
                        ),
                    )
                    for query in abe.queries
                ],
            )
            if isinstance(abe, AbeModalityScore)
            else None
        ),
    )


@router.post("/api/search", response_model=SearchResponse, response_model_exclude_none=True)
async def search(
    body: SearchRequest, service: SearchService = Depends(get_search_service)
) -> SearchResponse:
    start = time.perf_counter()
    try:
        hits = await service.search(body.query, mode=body.mode, top_k=body.top_k)
    except SearchServiceError as exc:
        # plan §P1.8: exc carries its own error_code/status_code -- distinct
        # causes (unknown mode vs. Elasticsearch down for `ocr`, CLAUDE.md
        # §7) must not collapse onto one hardcoded response.
        raise APIError(
            detail=str(exc), error_code=exc.error_code, status_code=exc.status_code
        ) from exc

    # plan §P1.7, PRD §6.6: operational latency measurement -- p50/p95 per
    # mode are computed offline (scripts/measure_search_latency.py) from
    # request logs like this one. No `cache_hit` field: /api/search has no
    # result cache to report a hit/miss for (only thumbnails do, a different
    # endpoint), so the field would always read False and mislead a reader
    # into thinking a cache exists here.
    latency_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "search request completed: mode=%s latency_ms=%.1f num_results=%d",
        body.mode,
        latency_ms,
        len(hits),
    )
    return _to_response(hits)


@router.post(
    "/api/search/kis/candidates",
    response_model=KisCandidateResponse,
    response_model_exclude_none=True,
)
async def retrieve_kis_candidates(
    body: KisCandidateRequest, service: SearchService = Depends(get_search_service)
) -> KisCandidateResponse:
    start = time.perf_counter()
    try:
        pool = await service.retrieve_kis_candidates(
            text_query=body.text_query,
            text_queries=body.text_queries,
            text_top_k=body.text_top_k,
            ocr_query=body.ocr_query,
            ocr_top_k=body.ocr_top_k,
            title_query=body.title_query,
            title_top_k=body.title_top_k,
        )
    except SearchServiceError as exc:
        raise APIError(
            detail=str(exc), error_code=exc.error_code, status_code=exc.status_code
        ) from exc

    latency_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "search request completed: mode=kis_stage1 latency_ms=%.1f candidate_pool_size=%d "
        "text_candidates=%d ocr_candidates=%d title_candidates=%d",
        latency_ms,
        pool.candidate_pool_size,
        pool.text_candidate_count,
        pool.ocr_candidate_count,
        pool.title_candidate_count,
    )
    return _to_kis_candidate_response(pool)


@router.post(
    "/api/search/kis/rerank",
    response_model=KisRerankResponse,
    response_model_exclude_none=True,
)
async def rerank_kis_candidates(
    body: KisRerankRequest, service: SearchService = Depends(get_search_service)
) -> KisRerankResponse:
    start = time.perf_counter()
    subqueries = [
        KisQuery(
            query_index=index,
            type=QueryType(query.type),
            text=query.text or "",
            pairs=tuple(
                AbePair(attribute=pair.attribute, object=pair.object) for pair in query.pairs
            ),
        )
        for index, query in enumerate(body.subqueries)
    ]
    try:
        hits = await service.rerank_kis_candidates(
            candidate_ids=body.candidate_ids,
            queries=subqueries,
            final_top_k=body.final_top_k,
        )
    except SearchServiceError as exc:
        raise APIError(
            detail=str(exc), error_code=exc.error_code, status_code=exc.status_code
        ) from exc

    latency_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "search request completed: mode=kis_stage2 latency_ms=%.1f candidate_pool_size=%d "
        "num_results=%d",
        latency_ms,
        len(body.candidate_ids),
        len(hits),
    )
    return _to_rerank_response(hits, len(body.candidate_ids), body.final_top_k)


@router.post("/api/search/image", response_model=SearchResponse, response_model_exclude_none=True)
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
