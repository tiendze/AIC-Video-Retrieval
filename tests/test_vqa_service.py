"""RED reproducer for `VqaService`/`GeminiVqaClient` (plan §P3.1 --
`vqa_service.py`, PRD §F7, architecture.md §3.6).

Covers the plan's three named P3.1 tests:
- test_vqa_prompt_contains_only_retrieved_keyframes
- test_vqa_output_matches_json_schema
- test_vqa_can_return_insufficient_evidence

Grounding-gate verification (evidence_keyframe_ids subset-of-retrieved
check) and the Gemini-down -> keyframe-grid fallback are P3.2, not tested
here -- this file only covers the Gemini-calling piece: prompt only carries
retrieved keyframes, structured output matches `VqaAnswer`, and the model is
allowed to abstain. Retry/timeout tests mirror
`tests/test_query_rewrite_service.py` (`GeminiQueryRewriter`/
`QueryRewriteService`) exactly -- same pattern, VQA's client just also takes
images.
"""

from __future__ import annotations

import pytest
from PIL import Image

from app.services.vqa_service import (
    GeminiVqaClient,
    RetrievedKeyframe,
    VqaAnswer,
    VqaService,
    VqaServiceError,
    _build_input_parts,
)


def _keyframe(keyframe_id: str, color: tuple[int, int, int] = (10, 20, 30)) -> RetrievedKeyframe:
    return RetrievedKeyframe(keyframe_id=keyframe_id, image=Image.new("RGB", (8, 8), color=color))


class TestBuildInputParts:
    def test_vqa_prompt_contains_only_retrieved_keyframes(self) -> None:
        retrieved = [_keyframe("kf1"), _keyframe("kf2")]

        parts = _build_input_parts("Biển hiệu ghi gì?", retrieved)

        referenced_ids = [
            part["text"].removeprefix("[keyframe_id=").removesuffix("]")
            for part in parts
            if part["type"] == "text" and part["text"].startswith("[keyframe_id=")
        ]
        assert referenced_ids == ["kf1", "kf2"]

        image_parts = [part for part in parts if part["type"] == "image"]
        assert len(image_parts) == len(retrieved)

    def test_prompt_never_references_a_keyframe_outside_the_retrieved_list(self) -> None:
        retrieved = [_keyframe("kf1")]
        not_retrieved_id = "kf_not_retrieved"

        parts = _build_input_parts("câu hỏi bất kỳ", retrieved)

        assert not any(not_retrieved_id in part["text"] for part in parts if part["type"] == "text")

    def test_question_text_is_included(self) -> None:
        parts = _build_input_parts("Ai đang phát biểu?", [_keyframe("kf1")])

        assert any(
            part["type"] == "text" and "Ai đang phát biểu?" in part["text"] for part in parts
        )

    def test_empty_keyframe_list_still_produces_a_prompt_with_no_images(self) -> None:
        parts = _build_input_parts("câu hỏi", [])

        assert all(part["type"] == "text" for part in parts)
        assert len(parts) == 1


class TestVqaAnswerSchema:
    def test_vqa_output_matches_json_schema(self) -> None:
        schema = VqaAnswer.model_json_schema()

        assert schema["properties"].keys() >= {
            "answer",
            "confidence",
            "evidence_keyframe_ids",
            "reasoning",
        }
        assert set(schema["required"]) == {"answer", "confidence", "reasoning"}

        payload = {
            "answer": "Biển hiệu ghi 'Uỷ ban nhân dân'",
            "confidence": 0.82,
            "evidence_keyframe_ids": ["kf1"],
            "reasoning": "Chữ trên biển hiệu trong kf1 đọc được rõ ràng.",
        }
        parsed = VqaAnswer.model_validate(payload)

        assert parsed.answer == payload["answer"]
        assert parsed.confidence == pytest.approx(0.82)
        assert parsed.evidence_keyframe_ids == ["kf1"]

    def test_confidence_must_be_within_0_and_1(self) -> None:
        with pytest.raises(ValueError):
            VqaAnswer.model_validate(
                {"answer": "x", "confidence": 1.5, "evidence_keyframe_ids": [], "reasoning": "x"}
            )

    def test_vqa_can_return_insufficient_evidence(self) -> None:
        payload = {
            "answer": "Không đủ căn cứ",
            "confidence": 0.1,
            "evidence_keyframe_ids": [],
            "reasoning": "Không khung hình nào cho thấy thông tin liên quan tới câu hỏi.",
        }

        parsed = VqaAnswer.model_validate(payload)

        assert parsed.evidence_keyframe_ids == []
        assert parsed.confidence < 0.5

    def test_evidence_keyframe_ids_defaults_to_empty_when_omitted(self) -> None:
        parsed = VqaAnswer.model_validate(
            {"answer": "Không đủ căn cứ", "confidence": 0.0, "reasoning": "..."}
        )

        assert parsed.evidence_keyframe_ids == []


class _FakeVqaClient:
    def __init__(self, answer: VqaAnswer | None = None, error: Exception | None = None) -> None:
        self._answer = answer
        self._error = error
        self.answer_calls: list[tuple[str, list[RetrievedKeyframe]]] = []

    def answer(self, question: str, keyframes: list[RetrievedKeyframe]) -> VqaAnswer:
        self.answer_calls.append((question, keyframes))
        if self._error is not None:
            raise self._error
        return self._answer


class _SlowFakeVqaClient:
    def __init__(self, sleep_seconds: float) -> None:
        self._sleep_seconds = sleep_seconds

    def answer(self, question: str, keyframes: list[RetrievedKeyframe]) -> VqaAnswer:
        import time

        time.sleep(self._sleep_seconds)
        raise AssertionError("unreachable -- should have timed out first")


_ANSWER = VqaAnswer(
    answer="Uỷ ban nhân dân",
    confidence=0.9,
    evidence_keyframe_ids=["kf1"],
    reasoning="Đọc được trên biển hiệu.",
)


class TestVqaService:
    @pytest.mark.asyncio
    async def test_returns_the_client_answer_on_success(self) -> None:
        client = _FakeVqaClient(answer=_ANSWER)
        service = VqaService(vqa_client=client, timeout_sec=8.0)
        keyframes = [_keyframe("kf1")]

        result = await service.answer("Biển hiệu ghi gì?", keyframes)

        assert result == _ANSWER
        assert client.answer_calls == [("Biển hiệu ghi gì?", keyframes)]

    @pytest.mark.asyncio
    async def test_raises_vqa_service_error_when_gemini_fails_after_exhausted_retries(
        self,
    ) -> None:
        client = _FakeVqaClient(error=RuntimeError("Gemini VQA failed after 3 attempts"))
        service = VqaService(vqa_client=client, timeout_sec=8.0)

        with pytest.raises(VqaServiceError):
            await service.answer("câu hỏi", [_keyframe("kf1")])

    @pytest.mark.asyncio
    async def test_raises_vqa_service_error_when_output_is_malformed(self) -> None:
        client = _FakeVqaClient(error=ValueError("invalid JSON from structured output"))
        service = VqaService(vqa_client=client, timeout_sec=8.0)

        with pytest.raises(VqaServiceError):
            await service.answer("câu hỏi", [_keyframe("kf1")])

    @pytest.mark.asyncio
    async def test_raises_vqa_service_error_when_the_call_exceeds_the_configured_timeout(
        self,
    ) -> None:
        client = _SlowFakeVqaClient(sleep_seconds=0.1)
        service = VqaService(vqa_client=client, timeout_sec=0.01)

        with pytest.raises(VqaServiceError):
            await service.answer("câu hỏi", [_keyframe("kf1")])


class TestGeminiVqaClientRetry:
    def test_retries_on_transient_error_then_succeeds(self, monkeypatch) -> None:
        client = object.__new__(GeminiVqaClient)  # bypass __init__: no real Gemini client
        object.__setattr__(
            client,
            "_config",
            type("_Cfg", (), {"max_retries": 3, "retry_base_delay_seconds": 0.0})(),
        )

        calls = {"count": 0}

        def _flaky_call_once(question: str, keyframes: list[RetrievedKeyframe]) -> VqaAnswer:
            calls["count"] += 1
            if calls["count"] < 3:
                raise ConnectionError("simulated transient network error")
            return _ANSWER

        monkeypatch.setattr(client, "_call_once", _flaky_call_once)

        result = client.answer("câu hỏi", [_keyframe("kf1")])

        assert result == _ANSWER
        assert calls["count"] == 3

    def test_raises_runtime_error_after_exhausting_retries(self, monkeypatch) -> None:
        client = object.__new__(GeminiVqaClient)
        object.__setattr__(
            client,
            "_config",
            type("_Cfg", (), {"max_retries": 2, "retry_base_delay_seconds": 0.0})(),
        )

        def _always_fails(question: str, keyframes: list[RetrievedKeyframe]) -> VqaAnswer:
            raise OSError("simulated permanent network error")

        monkeypatch.setattr(client, "_call_once", _always_fails)

        with pytest.raises(RuntimeError, match="failed after 2 attempts"):
            client.answer("câu hỏi", [_keyframe("kf1")])
