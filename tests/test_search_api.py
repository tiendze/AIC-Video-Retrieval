"""RED reproducer for `POST /api/search` (plan §P0.8) -- the router is thin
(validate + delegate, .claude/rules/ecc/python/fastapi.md): request schema
validates `mode`/`top_k`, the route builds a `SearchService` from injected
stores/encoder via `Depends`, and maps `SearchServiceError` onto the unified
`APIError` schema (CLAUDE.md §7).

A Fake `SearchService` is injected via `app.dependency_overrides` -- no
Docker/GPU needed; SearchService's own behavior is already covered by
tests/test_search_service.py.
"""

from __future__ import annotations

import io

import pytest
from fastapi import FastAPI, UploadFile
from fastapi.testclient import TestClient
from PIL import Image
from starlette.datastructures import Headers

from app.api import search as search_api
from app.api.errors import APIError, register_error_handling
from app.dependencies import get_search_service
from app.schemas.request.search import MAX_UPLOAD_SIZE_BYTES
from app.services.search_service import SearchHit, SearchServiceError


def _png_bytes(size: tuple[int, int] = (8, 8), color: tuple[int, int, int] = (10, 20, 30)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color=color).save(buf, format="PNG")
    return buf.getvalue()


def _bmp_bytes(size: tuple[int, int] = (8, 8), color: tuple[int, int, int] = (10, 20, 30)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color=color).save(buf, format="BMP")
    return buf.getvalue()


def _truncated_jpeg_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color=(10, 20, 30)).save(buf, format="JPEG")
    full = buf.getvalue()
    return full[: len(full) // 2]  # header identifies as JPEG; body decode fails with OSError


def _decompression_bomb_png_bytes() -> bytes:
    # Solid color compresses to well under MAX_UPLOAD_SIZE_BYTES while the
    # declared pixel count (196M) exceeds Pillow's default 178956970 limit.
    buf = io.BytesIO()
    Image.new("RGB", (14000, 14000), color=(10, 20, 30)).save(buf, format="PNG")
    return buf.getvalue()


class _FakeSearchService:
    def __init__(self, hits: list[SearchHit] | None = None, error: Exception | None = None) -> None:
        self._hits = hits or []
        self._error = error
        self.search_calls: list[tuple[str, str, int]] = []
        self.search_by_image_calls: list[int] = []

    async def search(self, query: str, mode: str, top_k: int) -> list[SearchHit]:
        self.search_calls.append((query, mode, top_k))
        if self._error is not None:
            raise self._error
        return self._hits

    async def search_by_image(self, image, top_k: int) -> list[SearchHit]:
        self.search_by_image_calls.append(top_k)
        if self._error is not None:
            raise self._error
        return self._hits


def _client(fake_service: _FakeSearchService) -> TestClient:
    app = FastAPI()
    register_error_handling(app)
    search_api.register_upload_guard(app)
    app.include_router(search_api.router)
    app.dependency_overrides[get_search_service] = lambda: fake_service
    return TestClient(app, raise_server_exceptions=False)


_ONE_HIT = [
    SearchHit(
        keyframe_id="L21_V001_f10",
        video_id="L21_V001",
        score=0.9,
        frame_idx=10,
        timestamp=0.4,
        thumbnail_path="t.jpg",
    )
]


def test_search_returns_200_with_mapped_hits():
    service = _FakeSearchService(hits=_ONE_HIT)
    client = _client(service)

    response = client.post("/api/search", json={"query": "thu tuong", "mode": "semantic"})

    assert response.status_code == 200
    assert response.json() == {
        "results": [
            {
                "keyframe_id": "L21_V001_f10",
                "video_id": "L21_V001",
                "score": 0.9,
                "frame_idx": 10,
                "timestamp": 0.4,
                "thumbnail_path": "t.jpg",
                "thumbnail_url": "/api/keyframes/L21_V001_f10/thumbnail",
                "image_url": "/api/keyframes/L21_V001_f10/image",
            }
        ]
    }


def test_search_uses_default_top_k_when_omitted():
    service = _FakeSearchService(hits=[])
    client = _client(service)

    client.post("/api/search", json={"query": "x", "mode": "ocr"})

    assert service.search_calls == [("x", "ocr", 20)]


def test_search_passes_through_an_explicit_top_k():
    service = _FakeSearchService(hits=[])
    client = _client(service)

    client.post("/api/search", json={"query": "x", "mode": "semantic", "top_k": 5})

    assert service.search_calls == [("x", "semantic", 5)]


def test_search_rejects_top_k_above_the_configured_max():
    service = _FakeSearchService(hits=[])
    client = _client(service)

    response = client.post("/api/search", json={"query": "x", "mode": "semantic", "top_k": 101})

    assert response.status_code == 422
    assert not service.search_calls


def test_search_rejects_an_unknown_mode_before_reaching_the_service():
    service = _FakeSearchService(hits=[])
    client = _client(service)

    response = client.post("/api/search", json={"query": "x", "mode": "video"})

    assert response.status_code == 422
    assert not service.search_calls


def test_search_service_error_maps_to_unified_400_error_schema():
    service = _FakeSearchService(error=SearchServiceError("Unknown search mode: 'x'"))
    client = _client(service)

    response = client.post("/api/search", json={"query": "x", "mode": "semantic"})

    assert response.status_code == 400
    body = response.json()
    assert body["error_code"] == "invalid_search_mode"
    assert set(body.keys()) == {"detail", "error_code", "request_id"}


def test_search_by_image_returns_200_with_mapped_hits():
    service = _FakeSearchService(hits=_ONE_HIT)
    client = _client(service)

    response = client.post(
        "/api/search/image", files={"file": ("query.png", _png_bytes(), "image/png")}
    )

    assert response.status_code == 200
    assert response.json() == {
        "results": [
            {
                "keyframe_id": "L21_V001_f10",
                "video_id": "L21_V001",
                "score": 0.9,
                "frame_idx": 10,
                "timestamp": 0.4,
                "thumbnail_path": "t.jpg",
                "thumbnail_url": "/api/keyframes/L21_V001_f10/thumbnail",
                "image_url": "/api/keyframes/L21_V001_f10/image",
            }
        ]
    }
    assert service.search_by_image_calls == [20]  # DEFAULT_TOP_K


def test_search_by_image_passes_through_an_explicit_top_k():
    service = _FakeSearchService(hits=[])
    client = _client(service)

    client.post(
        "/api/search/image",
        files={"file": ("query.png", _png_bytes(), "image/png")},
        data={"top_k": "7"},
    )

    assert service.search_by_image_calls == [7]


def test_search_by_image_rejects_top_k_above_the_configured_max():
    service = _FakeSearchService(hits=[])
    client = _client(service)

    response = client.post(
        "/api/search/image",
        files={"file": ("query.png", _png_bytes(), "image/png")},
        data={"top_k": "101"},
    )

    assert response.status_code == 422
    assert not service.search_by_image_calls


def test_search_by_image_allows_top_k_at_the_configured_max():
    service = _FakeSearchService(hits=[])
    client = _client(service)

    response = client.post(
        "/api/search/image",
        files={"file": ("query.png", _png_bytes(), "image/png")},
        data={"top_k": "100"},
    )

    assert response.status_code == 200
    assert service.search_by_image_calls == [100]


def test_missing_file_field_returns_422():
    service = _FakeSearchService(hits=[])
    client = _client(service)

    response = client.post("/api/search/image")

    assert response.status_code == 422
    assert not service.search_by_image_calls


def test_oversized_upload_is_rejected():
    service = _FakeSearchService(hits=[])
    client = _client(service)

    oversized = b"\x00" * (MAX_UPLOAD_SIZE_BYTES + 1)
    response = client.post(
        "/api/search/image", files={"file": ("query.png", oversized, "image/png")}
    )

    assert response.status_code == 413
    body = response.json()
    assert body["error_code"] == "upload_too_large"
    assert not service.search_by_image_calls


async def test_route_itself_still_rejects_an_oversized_body_without_a_content_length_header():
    # The middleware only catches a truthful Content-Length (module
    # docstring in app/api/search.py); a request that omits/understates it
    # (e.g. chunked transfer) reaches the route, which must still enforce
    # MAX_UPLOAD_SIZE_BYTES itself as the second layer of defense.
    oversized = b"\x00" * (MAX_UPLOAD_SIZE_BYTES + 1)
    upload = UploadFile(
        io.BytesIO(oversized), filename="query.png", headers=Headers({"content-type": "image/png"})
    )
    service = _FakeSearchService(hits=[])

    with pytest.raises(APIError) as exc_info:
        await search_api.search_by_image(file=upload, top_k=20, service=service)

    assert exc_info.value.status_code == 413
    assert exc_info.value.error_code == "upload_too_large"
    assert not service.search_by_image_calls


async def test_content_length_guard_rejects_before_reading_the_body():
    # Direct unit test of the middleware itself (registered via
    # register_upload_guard): a spoofed Content-Length over the configured
    # limit must short-circuit with 413 without ever calling `call_next`,
    # i.e. before Starlette's multipart parser touches the request body.
    from starlette.requests import Request

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/search/image",
        "headers": [(b"content-length", str(MAX_UPLOAD_SIZE_BYTES + 1).encode())],
    }
    request = Request(scope)
    call_next_invoked = False

    async def call_next(_request):
        nonlocal call_next_invoked
        call_next_invoked = True
        raise AssertionError("call_next must not run for an oversized Content-Length")

    response = await search_api._content_length_guard(request, call_next)

    assert response.status_code == 413
    assert call_next_invoked is False


def test_non_image_mime_is_rejected():
    service = _FakeSearchService(hits=[])
    client = _client(service)

    response = client.post(
        "/api/search/image", files={"file": ("query.txt", b"not an image", "text/plain")}
    )

    assert response.status_code == 415
    body = response.json()
    assert body["error_code"] == "unsupported_media_type"
    assert not service.search_by_image_calls


def test_corrupt_image_bytes_are_rejected():
    service = _FakeSearchService(hits=[])
    client = _client(service)

    response = client.post(
        "/api/search/image",
        files={"file": ("query.png", b"not actually a png", "image/png")},
    )

    assert response.status_code == 400
    body = response.json()
    assert body["error_code"] == "invalid_image"
    assert not service.search_by_image_calls


def test_zero_byte_upload_is_rejected():
    service = _FakeSearchService(hits=[])
    client = _client(service)

    response = client.post("/api/search/image", files={"file": ("query.png", b"", "image/png")})

    assert response.status_code == 400
    assert response.json()["error_code"] == "invalid_image"
    assert not service.search_by_image_calls


def test_truncated_image_bytes_are_rejected():
    # Regression: a truncated-but-identifiable file raises plain OSError
    # from Image.load(), not UnidentifiedImageError -- must still map to 400.
    service = _FakeSearchService(hits=[])
    client = _client(service)

    response = client.post(
        "/api/search/image",
        files={"file": ("query.jpg", _truncated_jpeg_bytes(), "image/jpeg")},
    )

    assert response.status_code == 400
    assert response.json()["error_code"] == "invalid_image"
    assert not service.search_by_image_calls


def test_oversized_image_dimensions_are_rejected():
    # Regression: Pillow raises DecompressionBombError (not OSError) for a
    # small file with a huge declared pixel count -- must still map to 400,
    # not the unhandled-exception 500 path.
    service = _FakeSearchService(hits=[])
    client = _client(service)

    response = client.post(
        "/api/search/image",
        files={"file": ("bomb.png", _decompression_bomb_png_bytes(), "image/png")},
    )

    assert response.status_code == 400
    assert response.json()["error_code"] == "invalid_image"
    assert not service.search_by_image_calls


def test_declared_mime_type_mismatching_actual_format_is_rejected():
    # Regression: Content-Type is only a caller-supplied label -- Image.open()
    # detects format from magic bytes. A BMP declared as image/png must be
    # rejected, not silently encoded as whatever format it actually is.
    service = _FakeSearchService(hits=[])
    client = _client(service)

    response = client.post(
        "/api/search/image", files={"file": ("query.png", _bmp_bytes(), "image/png")}
    )

    assert response.status_code == 415
    assert response.json()["error_code"] == "unsupported_media_type"
    assert not service.search_by_image_calls
