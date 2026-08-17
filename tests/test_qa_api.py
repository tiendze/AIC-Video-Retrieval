"""RED reproducer for `POST /api/qa` (plan §P3.2, PRD §F7,
architecture.md §7) -- thin router: validate the request, delegate to
`QaService`, shape the response. No business logic here; the grounding gate
and degraded-mode semantics are covered by tests/test_qa_service.py.

The load-bearing case is the plan's own experiment written as an automated
test: **Gemini unreachable => `POST /api/qa` still answers 200 with the
keyframe grid + a readable notice**, never a 5xx. An operator being timed
during the contest must still get their grid (CLAUDE.md §7, US-5).

A Fake `QaService` is injected via `app.dependency_overrides` -- no
network/GPU/Docker.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import qa as qa_api
from app.api.errors import register_error_handling
from app.dependencies import get_qa_service
from app.schemas.request.search import DEFAULT_TOP_K
from app.services.search_service import SearchHit
from app.services.vqa_service import VQA_UNAVAILABLE_NOTICE, QaResult


def _hit(keyframe_id: str) -> SearchHit:
    return SearchHit(
        keyframe_id=keyframe_id,
        video_id="video1",
        score=0.9,
        frame_idx=42,
        timestamp=1.4,
        thumbnail_path=f"/thumb/{keyframe_id}.jpg",
    )


_ANSWERED = QaResult(
    status="answered",
    answer="Uỷ ban nhân dân",
    confidence=0.9,
    evidence_keyframe_ids=("kf1",),
    reasoning="Đọc được trên biển hiệu.",
    keyframes=(_hit("kf1"), _hit("kf2")),
    notice=None,
)

_UNAVAILABLE = QaResult(
    status="unavailable",
    answer=None,
    confidence=None,
    evidence_keyframe_ids=(),
    reasoning=None,
    keyframes=(_hit("kf1"), _hit("kf2")),
    notice=VQA_UNAVAILABLE_NOTICE,
)


class _FakeQaService:
    def __init__(self, result: QaResult) -> None:
        self._result = result
        self.answer_question_calls: list[tuple[str, int]] = []

    async def answer_question(self, question: str, top_k: int) -> QaResult:
        self.answer_question_calls.append((question, top_k))
        return self._result


def _client(fake_service: _FakeQaService) -> TestClient:
    app = FastAPI()
    register_error_handling(app)
    app.include_router(qa_api.router)
    app.dependency_overrides[get_qa_service] = lambda: fake_service
    return TestClient(app, raise_server_exceptions=False)


class TestQaEndpoint:
    def test_returns_the_answer_and_the_grid(self) -> None:
        client = _client(_FakeQaService(_ANSWERED))

        response = client.post("/api/qa", json={"question": "Biển hiệu ghi gì?"})

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "answered"
        assert body["answer"] == "Uỷ ban nhân dân"
        assert body["confidence"] == 0.9
        assert body["evidence_keyframe_ids"] == ["kf1"]
        assert [kf["keyframe_id"] for kf in body["keyframes"]] == ["kf1", "kf2"]
        assert body["notice"] is None

    def test_vqa_failure_still_returns_keyframe_grid(self) -> None:
        """The plan's experiment as an automated test: Gemini down must not
        become a 5xx -- the operator still gets 200 + grid + notice."""
        client = _client(_FakeQaService(_UNAVAILABLE))

        response = client.post("/api/qa", json={"question": "câu hỏi"})

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "unavailable"
        assert body["answer"] is None
        assert body["notice"] == VQA_UNAVAILABLE_NOTICE
        assert [kf["keyframe_id"] for kf in body["keyframes"]] == ["kf1", "kf2"]

    def test_keyframes_carry_thumbnail_and_image_urls(self) -> None:
        """Same contract as /api/search and /api/quest -- the frontend loads
        images through the API, never from a raw filesystem path
        (app/schemas/response/search.py)."""
        client = _client(_FakeQaService(_ANSWERED))

        body = client.post("/api/qa", json={"question": "câu hỏi"}).json()

        first = body["keyframes"][0]
        assert first["thumbnail_url"] == "/api/keyframes/kf1/thumbnail"
        assert first["image_url"] == "/api/keyframes/kf1/image"

    def test_applies_the_default_top_k_when_omitted(self) -> None:
        service = _FakeQaService(_ANSWERED)
        client = _client(service)

        client.post("/api/qa", json={"question": "câu hỏi"})

        assert service.answer_question_calls == [("câu hỏi", DEFAULT_TOP_K)]

    def test_passes_an_explicit_top_k_through(self) -> None:
        service = _FakeQaService(_ANSWERED)
        client = _client(service)

        client.post("/api/qa", json={"question": "câu hỏi", "top_k": 5})

        assert service.answer_question_calls == [("câu hỏi", 5)]

    def test_rejects_an_empty_question(self) -> None:
        client = _client(_FakeQaService(_ANSWERED))

        response = client.post("/api/qa", json={"question": ""})

        assert response.status_code == 422

    def test_rejects_a_top_k_above_the_submission_cap(self) -> None:
        client = _client(_FakeQaService(_ANSWERED))

        response = client.post("/api/qa", json={"question": "câu hỏi", "top_k": 101})

        assert response.status_code == 422

    def test_rejects_a_missing_question_field(self) -> None:
        client = _client(_FakeQaService(_ANSWERED))

        response = client.post("/api/qa", json={"top_k": 5})

        assert response.status_code == 422
