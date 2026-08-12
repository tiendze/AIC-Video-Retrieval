"""RED reproducer for `GET /api/keyframes/{id}/thumbnail` and `/image`
(plan §P1.6 backend gap 1.2) -- thin router: delegate to `ThumbnailService`,
map `ThumbnailServiceError` onto the unified `{detail, error_code,
request_id}` schema (CLAUDE.md §7). A Fake `ThumbnailService` is injected
via `app.dependency_overrides` -- no filesystem/Mongo needed here;
`ThumbnailService`'s own behavior is covered by tests/test_thumbnail_service.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import keyframes as keyframes_api
from app.api.errors import register_error_handling
from app.dependencies import get_thumbnail_service
from app.services.thumbnail_service import ThumbnailServiceError


class _FakeThumbnailService:
    def __init__(
        self,
        thumbnail_path: Path | None = None,
        image_path: Path | None = None,
        error: ThumbnailServiceError | None = None,
    ) -> None:
        self._thumbnail_path = thumbnail_path
        self._image_path = image_path
        self._error = error
        self.cache_max_age_seconds = 3600
        self.thumbnail_calls: list[str] = []
        self.image_calls: list[str] = []

    async def get_thumbnail_path(self, keyframe_id: str) -> Path:
        self.thumbnail_calls.append(keyframe_id)
        if self._error is not None:
            raise self._error
        return self._thumbnail_path

    async def get_image_path(self, keyframe_id: str) -> Path:
        self.image_calls.append(keyframe_id)
        if self._error is not None:
            raise self._error
        return self._image_path


def _client(fake_service: _FakeThumbnailService) -> TestClient:
    app = FastAPI()
    register_error_handling(app)
    app.include_router(keyframes_api.router)
    app.dependency_overrides[get_thumbnail_service] = lambda: fake_service
    return TestClient(app, raise_server_exceptions=False)


def _real_jpeg(tmp_path: Path, name: str) -> Path:
    from PIL import Image

    path = tmp_path / name
    Image.new("RGB", (8, 8), color=(1, 2, 3)).save(path, format="JPEG")
    return path


def test_thumbnail_endpoint_returns_the_image_bytes(tmp_path: Path) -> None:
    thumb_path = _real_jpeg(tmp_path, "thumb.jpg")
    client = _client(_FakeThumbnailService(thumbnail_path=thumb_path))

    response = client.get("/api/keyframes/L21_V001_f10/thumbnail")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.content == thumb_path.read_bytes()


def test_thumbnail_endpoint_sets_long_lived_cache_control(tmp_path: Path) -> None:
    thumb_path = _real_jpeg(tmp_path, "thumb.jpg")
    client = _client(_FakeThumbnailService(thumbnail_path=thumb_path))

    response = client.get("/api/keyframes/L21_V001_f10/thumbnail")

    assert "max-age=3600" in response.headers["cache-control"]


def test_thumbnail_endpoint_maps_service_error_to_404(tmp_path: Path) -> None:
    fake = _FakeThumbnailService(
        error=ThumbnailServiceError("Unknown keyframe_id: 'x'", "keyframe_not_found")
    )
    client = _client(fake)

    response = client.get("/api/keyframes/x/thumbnail")

    assert response.status_code == 404
    body = response.json()
    assert body["error_code"] == "keyframe_not_found"
    assert "request_id" in body


def test_image_endpoint_returns_the_image_bytes(tmp_path: Path) -> None:
    image_path = _real_jpeg(tmp_path, "full.jpg")
    client = _client(_FakeThumbnailService(image_path=image_path))

    response = client.get("/api/keyframes/L21_V001_f10/image")

    assert response.status_code == 200
    assert response.content == image_path.read_bytes()


def test_image_endpoint_maps_service_error_to_404(tmp_path: Path) -> None:
    fake = _FakeThumbnailService(
        error=ThumbnailServiceError("Source image missing", "source_image_missing")
    )
    client = _client(fake)

    response = client.get("/api/keyframes/L21_V001_f10/image")

    assert response.status_code == 404
    assert response.json()["error_code"] == "source_image_missing"


@pytest.mark.parametrize("suffix", ["thumbnail", "image"])
def test_endpoint_calls_service_with_the_path_keyframe_id(tmp_path: Path, suffix: str) -> None:
    image_path = _real_jpeg(tmp_path, "full.jpg")
    fake = _FakeThumbnailService(thumbnail_path=image_path, image_path=image_path)
    client = _client(fake)

    client.get(f"/api/keyframes/L21_V001_f10/{suffix}")

    calls = fake.thumbnail_calls if suffix == "thumbnail" else fake.image_calls
    assert calls == ["L21_V001_f10"]
