"""RED reproducer for `ThumbnailService` (plan §P1.6 backend gap 1.2) --
`GET /api/keyframes/{id}/thumbnail` needs on-demand resize + disk cache
because the imported Kaggle keyframe set has no pre-made thumbnail
(`pipelines/import_embed_out.py` sets `thumbnail_path == image_path`, so the
live corpus is 221,903 full-res JPEGs, ~40 GB total). A cache hit must never
touch MongoDB (CLAUDE.md §7 -- no unnecessary work on the hot path), and a
`keyframe_id` is never trusted to build a filesystem path directly (path
traversal must be structurally impossible, not merely checked).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from app.services.thumbnail_service import (
    ThumbnailService,
    ThumbnailServiceError,
    cache_path_for,
)


class _FakeMongoStore:
    def __init__(self, docs_by_id: dict[str, dict[str, Any]]) -> None:
        self._docs_by_id = docs_by_id
        self.get_keyframes_by_ids_calls: list[list[str]] = []

    async def get_keyframes_by_ids(self, keyframe_ids) -> dict[str, dict[str, Any]]:
        self.get_keyframes_by_ids_calls.append(list(keyframe_ids))
        return {kid: self._docs_by_id[kid] for kid in keyframe_ids if kid in self._docs_by_id}


def _make_source_image(path: Path, size: tuple[int, int] = (640, 480)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(10, 20, 30)).save(path, format="JPEG")


def _service(
    mongo_store: _FakeMongoStore, tmp_path: Path, max_px: int = 320, jpeg_quality: int = 80
) -> ThumbnailService:
    return ThumbnailService(
        mongo_store=mongo_store,
        cache_dir=tmp_path / "cache",
        max_px=max_px,
        jpeg_quality=jpeg_quality,
        cache_max_age_seconds=3600,
    )


@pytest.mark.asyncio
async def test_get_thumbnail_path_resizes_and_writes_cache_on_first_request(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source" / "L21_V001_f10.jpg"
    _make_source_image(source, size=(1920, 1080))
    mongo = _FakeMongoStore({"L21_V001_f10": {"image_path": str(source)}})
    service = _service(mongo, tmp_path, max_px=320)

    result_path = await service.get_thumbnail_path("L21_V001_f10")

    assert result_path == cache_path_for(tmp_path / "cache", "L21_V001_f10")
    assert result_path.exists()
    with Image.open(result_path) as thumb:
        assert max(thumb.size) <= 320


@pytest.mark.asyncio
async def test_get_thumbnail_path_serves_from_cache_without_touching_mongo_again(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source" / "L21_V001_f10.jpg"
    _make_source_image(source)
    mongo = _FakeMongoStore({"L21_V001_f10": {"image_path": str(source)}})
    service = _service(mongo, tmp_path)

    await service.get_thumbnail_path("L21_V001_f10")
    assert mongo.get_keyframes_by_ids_calls == [["L21_V001_f10"]]

    await service.get_thumbnail_path("L21_V001_f10")

    # A second call must be served entirely from the on-disk cache -- no
    # second Mongo round trip.
    assert mongo.get_keyframes_by_ids_calls == [["L21_V001_f10"]]


@pytest.mark.asyncio
async def test_get_thumbnail_path_returns_404_style_error_when_source_missing(
    tmp_path: Path,
) -> None:
    missing_source = tmp_path / "source" / "does_not_exist.jpg"
    mongo = _FakeMongoStore({"L21_V001_f10": {"image_path": str(missing_source)}})
    service = _service(mongo, tmp_path)

    with pytest.raises(ThumbnailServiceError) as exc_info:
        await service.get_thumbnail_path("L21_V001_f10")

    assert exc_info.value.error_code == "source_image_missing"


@pytest.mark.asyncio
async def test_get_thumbnail_path_raises_when_keyframe_id_unknown(tmp_path: Path) -> None:
    mongo = _FakeMongoStore({})
    service = _service(mongo, tmp_path)

    with pytest.raises(ThumbnailServiceError) as exc_info:
        await service.get_thumbnail_path("L21_V999_f10")

    assert exc_info.value.error_code == "keyframe_not_found"


@pytest.mark.parametrize(
    "malicious_id",
    ["../../etc/passwd", "..", "a/b", "a\\b", "", "x" * 129],
)
@pytest.mark.asyncio
async def test_get_thumbnail_path_rejects_ids_that_are_not_a_plain_token(
    tmp_path: Path, malicious_id: str
) -> None:
    """A `keyframe_id` is never trusted to build a filesystem path directly
    -- rejecting anything outside `[A-Za-z0-9_-]{1,128}` makes path
    traversal structurally impossible rather than merely checked."""
    mongo = _FakeMongoStore({})
    service = _service(mongo, tmp_path)

    with pytest.raises(ThumbnailServiceError) as exc_info:
        await service.get_thumbnail_path(malicious_id)

    assert exc_info.value.error_code == "invalid_keyframe_id"
    assert mongo.get_keyframes_by_ids_calls == []  # rejected before ever reaching Mongo


@pytest.mark.asyncio
async def test_get_image_path_returns_source_path_unresized(tmp_path: Path) -> None:
    source = tmp_path / "source" / "L21_V001_f10.jpg"
    _make_source_image(source, size=(1920, 1080))
    mongo = _FakeMongoStore({"L21_V001_f10": {"image_path": str(source)}})
    service = _service(mongo, tmp_path)

    result_path = await service.get_image_path("L21_V001_f10")

    assert result_path == source
    with Image.open(result_path) as image:
        assert image.size == (1920, 1080)  # untouched, unlike the thumbnail path


@pytest.mark.asyncio
async def test_get_image_path_returns_404_style_error_when_source_missing(tmp_path: Path) -> None:
    missing_source = tmp_path / "source" / "does_not_exist.jpg"
    mongo = _FakeMongoStore({"L21_V001_f10": {"image_path": str(missing_source)}})
    service = _service(mongo, tmp_path)

    with pytest.raises(ThumbnailServiceError) as exc_info:
        await service.get_image_path("L21_V001_f10")

    assert exc_info.value.error_code == "source_image_missing"


def test_cache_path_for_shards_across_256_subdirectories() -> None:
    path_a = cache_path_for(Path("/cache"), "L21_V001_f10")
    path_b = cache_path_for(Path("/cache"), "L21_V001_f11")

    # Deterministic and namespaced by keyframe_id -- never reused across ids.
    assert path_a == cache_path_for(Path("/cache"), "L21_V001_f10")
    assert path_a != path_b
    assert path_a.parent.parent == Path("/cache")
    assert len(path_a.parent.name) == 2  # 1-byte sha1 shard prefix
