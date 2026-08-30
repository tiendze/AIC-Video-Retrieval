from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import view_file as view_file_api
from app.api.errors import register_error_handling
from app.dependencies import get_view_file_service
from app.schemas.response.view_file import ViewFileFrameResolutionResponse


class _FakeViewFileService:
    async def frame_resolution(
        self, video_id: str, frame_idx: int
    ) -> ViewFileFrameResolutionResponse:
        return ViewFileFrameResolutionResponse(
            video_id=video_id,
            frame_idx=frame_idx,
            timestamp_seconds=frame_idx / 25,
            source="local",
            keyframe_id=f"{video_id}_f{frame_idx}",
            thumbnail_url=f"/api/keyframes/{video_id}_f{frame_idx}/thumbnail",
            image_url=f"/api/keyframes/{video_id}_f{frame_idx}/image",
            fps=25,
        )


def _client(fake_service: _FakeViewFileService) -> TestClient:
    app = FastAPI()
    register_error_handling(app)
    app.include_router(view_file_api.router)
    app.dependency_overrides[get_view_file_service] = lambda: fake_service
    return TestClient(app, raise_server_exceptions=False)


def test_frame_resolution_endpoint_returns_exact_frame_state() -> None:
    client = _client(_FakeViewFileService())

    response = client.get("/api/view-file/frame-resolution?video_id=L21_V001&frame_idx=10")

    assert response.status_code == 200
    body = response.json()
    assert body["video_id"] == "L21_V001"
    assert body["frame_idx"] == 10
    assert body["source"] == "local"
    assert body["thumbnail_url"] == "/api/keyframes/L21_V001_f10/thumbnail"
