"""`GET /api/videos/{id}` and `/api/videos/{id}/keyframes` (plan §P1.6
backend gap 1.4) -- thin wrappers over `MongoStore.get_video()` /
`get_keyframes_for_video()` (both already exist,
.claude/rules/ecc/python/fastapi.md: no business logic here). `watch_url`
backs the frontend's "Play YouTube" button; the keyframe list backs the
neighboring-keyframes panel in the detail view -- not in the paper, added
because a Textual KIS top-1 hit is often a few shots off from the actual
answer, and it reuses `get_keyframes_for_video()` as-is.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.errors import APIError
from app.dependencies import get_mongo_store
from app.schemas.response.video import VideoKeyframeResponse, VideoKeyframesResponse, VideoResponse
from app.stores.mongo_client import MongoStore

router = APIRouter()


async def _get_video_or_404(video_id: str, mongo_store: MongoStore) -> dict:
    doc = await mongo_store.get_video(video_id)
    if doc is None:
        raise APIError(
            detail=f"Unknown video_id: {video_id!r}",
            error_code="video_not_found",
            status_code=404,
        )
    return doc


@router.get("/api/videos/{video_id}", response_model=VideoResponse)
async def get_video(
    video_id: str, mongo_store: MongoStore = Depends(get_mongo_store)
) -> VideoResponse:
    doc = await _get_video_or_404(video_id, mongo_store)
    return VideoResponse(
        video_id=doc["video_id"],
        watch_url=doc["watch_url"],
        title=doc["title"],
        author=doc["author"],
        publish_date=doc["publish_date"],
        duration=doc["duration"],
        fps=doc["fps"],
        num_keyframes=doc["num_keyframes"],
        s_v=doc["s_v"],
        e_v=doc["e_v"],
    )


@router.get("/api/videos/{video_id}/keyframes", response_model=VideoKeyframesResponse)
async def get_video_keyframes(
    video_id: str, mongo_store: MongoStore = Depends(get_mongo_store)
) -> VideoKeyframesResponse:
    # Confirms the video itself exists before listing its keyframes -- an
    # unknown video_id must 404, not silently look like "a video with zero
    # keyframes" (CLAUDE.md §7: fail clearly, don't return empty quietly).
    await _get_video_or_404(video_id, mongo_store)
    keyframe_docs = await mongo_store.get_keyframes_for_video(video_id)
    return VideoKeyframesResponse(
        keyframes=[
            VideoKeyframeResponse(
                keyframe_id=doc["keyframe_id"],
                frame_idx=doc["frame_idx"],
                timestamp=doc["timestamp"],
                global_index=doc["global_index"],
                thumbnail_url=f"/api/keyframes/{doc['keyframe_id']}/thumbnail",
                image_url=f"/api/keyframes/{doc['keyframe_id']}/image",
            )
            for doc in keyframe_docs
        ]
    )
