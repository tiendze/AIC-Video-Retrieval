"""Serves keyframe thumbnails/images for the frontend (plan §P1.6 backend
gap 1.2). The imported Kaggle keyframe set has no pre-made thumbnail --
`pipelines/import_embed_out.py`'s `to_keyframe_document()` sets
`thumbnail_path == image_path` -- so the live corpus is 221,903 full-res
JPEGs (~40 GB). This resizes on demand and caches the result on disk, so a
result grid never makes the browser decode full-res images.

`resize_and_save()` is the single place that defines what a "thumbnail" is
-- `scripts/build_thumbnails.py` (batch cache warm) calls the exact same
function, so there is no second code path to drift from this one.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
from pathlib import Path
from typing import Protocol

from PIL import Image

# keyframe_id is never trusted to build a filesystem path directly -- only a
# plain token (letters/digits/underscore/hyphen) is accepted, so path
# traversal is structurally impossible rather than merely checked.
_KEYFRAME_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


class ThumbnailServiceError(Exception):
    """Raised for every caller-facing failure case -- the router maps all
    three (invalid id shape, unknown keyframe_id, missing source file) onto
    a 404 with a distinct `error_code` (CLAUDE.md §7 unified error schema)."""

    def __init__(self, detail: str, error_code: str) -> None:
        super().__init__(detail)
        self.detail = detail
        self.error_code = error_code


class _MongoKeyframeLookup(Protocol):
    async def get_keyframes_by_ids(self, keyframe_ids: list[str]) -> dict[str, dict]: ...


def cache_path_for(cache_dir: Path, keyframe_id: str) -> Path:
    """Shards into 256 subdirectories (sha1 first byte) -- 221k thumbnail
    files flat in one directory is its own problem on NTFS. Deliberately a
    separate namespace from the data's own `<batch>/<video_id>/...` layout
    (`pipelines/import_embed_out.py`), so this cache never becomes a second
    place that has to agree with the ingest path convention."""
    digest = hashlib.sha1(
        keyframe_id.encode("utf-8")
    ).hexdigest()  # noqa: S324 -- cache sharding, not security
    return cache_dir / digest[:2] / f"{keyframe_id}.jpg"


def resize_and_save(source_path: Path, dest_path: Path, max_px: int, jpeg_quality: int) -> None:
    """Blocking PIL resize -- always run via `asyncio.to_thread` from async
    callers. Shared verbatim by `ThumbnailService` (on-demand) and
    `scripts/build_thumbnails.py` (batch warm)."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path) as image:
        rgb_image = image.convert("RGB")
        rgb_image.thumbnail((max_px, max_px))
        rgb_image.save(dest_path, format="JPEG", quality=jpeg_quality)


def _validate_keyframe_id(keyframe_id: str) -> None:
    if not _KEYFRAME_ID_PATTERN.match(keyframe_id):
        raise ThumbnailServiceError(
            detail=f"Invalid keyframe_id: {keyframe_id!r}", error_code="invalid_keyframe_id"
        )


class ThumbnailService:
    def __init__(
        self,
        mongo_store: _MongoKeyframeLookup,
        cache_dir: Path,
        max_px: int,
        jpeg_quality: int,
        cache_max_age_seconds: int,
    ) -> None:
        self._mongo_store = mongo_store
        self._cache_dir = cache_dir
        self._max_px = max_px
        self._jpeg_quality = jpeg_quality
        self.cache_max_age_seconds = cache_max_age_seconds

    async def get_thumbnail_path(self, keyframe_id: str) -> Path:
        _validate_keyframe_id(keyframe_id)
        cache_path = cache_path_for(self._cache_dir, keyframe_id)
        if cache_path.exists():
            return cache_path  # cache hit -- zero Mongo round trips

        source_path = await self._resolve_image_path(keyframe_id)
        await asyncio.to_thread(
            resize_and_save, source_path, cache_path, self._max_px, self._jpeg_quality
        )
        return cache_path

    async def get_image_path(self, keyframe_id: str) -> Path:
        _validate_keyframe_id(keyframe_id)
        return await self._resolve_image_path(keyframe_id)

    async def _resolve_image_path(self, keyframe_id: str) -> Path:
        docs = await self._mongo_store.get_keyframes_by_ids([keyframe_id])
        doc = docs.get(keyframe_id)
        if doc is None:
            raise ThumbnailServiceError(
                detail=f"Unknown keyframe_id: {keyframe_id!r}", error_code="keyframe_not_found"
            )
        source_path = Path(doc["image_path"])
        if not source_path.exists():
            raise ThumbnailServiceError(
                detail=f"Source image missing on disk for {keyframe_id!r}",
                error_code="source_image_missing",
            )
        return source_path
