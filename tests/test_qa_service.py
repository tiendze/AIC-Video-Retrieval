"""RED reproducer for the P3.2 grounding gate + `QaService` (plan §P3.2,
PRD §F7, architecture.md §3.6 step "Kiểm tra evidence_ids").

Covers the plan's two named P3.2 tests:
- test_vqa_rejects_evidence_outside_retrieved_set
- test_vqa_failure_still_returns_keyframe_grid

The gate is the anti-fabrication checkpoint between the VLM and the
operator: any `evidence_keyframe_ids` entry that was NOT among the
keyframes actually handed to the model means the model invented a
citation, so the whole answer is untrustworthy and collapses to
"không đủ căn cứ". Without it `Grounded Accuracy` cannot be measured and
`Hallucination Rate` cannot be controlled (plan §P3.2).

`QaService` is the orchestrator: retrieve -> load images -> ask Gemini ->
gate. Its hard guarantee is that the keyframe grid is returned in EVERY
outcome, including when Gemini is unreachable (CLAUDE.md §7: a
network-dependent branch must never kill the main request; US-5 stays
partially usable).

No network/GPU/Docker here -- SearchService, the image loader, and
VqaService are all Fakes injected through the constructor.
"""

from __future__ import annotations

import pytest
from PIL import Image

from app.services.search_service import SearchHit
from app.services.vqa_service import (
    INSUFFICIENT_EVIDENCE_ANSWER,
    VQA_UNAVAILABLE_NOTICE,
    GroundedAnswer,
    KeyframeImageUnavailable,
    QaService,
    ThumbnailImageLoader,
    VqaAnswer,
    VqaServiceError,
    apply_grounding_gate,
)


def _hit(keyframe_id: str, video_id: str = "video1", frame_idx: int = 42) -> SearchHit:
    return SearchHit(
        keyframe_id=keyframe_id,
        video_id=video_id,
        score=0.9,
        frame_idx=frame_idx,
        timestamp=1.4,
        thumbnail_path=f"/thumb/{keyframe_id}.jpg",
    )


def _answer(
    answer: str = "Uỷ ban nhân dân",
    confidence: float = 0.9,
    evidence: list[str] | None = None,
    reasoning: str = "Đọc được trên biển hiệu.",
) -> VqaAnswer:
    return VqaAnswer(
        answer=answer,
        confidence=confidence,
        evidence_keyframe_ids=["kf1"] if evidence is None else evidence,
        reasoning=reasoning,
    )


class TestGroundingGate:
    def test_vqa_rejects_evidence_outside_retrieved_set(self) -> None:
        answer = _answer(evidence=["kf_never_retrieved"])

        result = apply_grounding_gate(answer, {"kf1", "kf2"})

        assert isinstance(result, GroundedAnswer)
        assert result.status == "insufficient_evidence"
        assert result.answer == INSUFFICIENT_EVIDENCE_ANSWER
        assert result.evidence_keyframe_ids == ()

    def test_a_single_fabricated_id_rejects_the_whole_answer(self) -> None:
        """One real citation does not launder a fabricated one -- a model
        that invents any keyframe_id is fabricating, so the entire answer is
        untrustworthy, not just the bad citation."""
        answer = _answer(evidence=["kf1", "kf_never_retrieved"])

        result = apply_grounding_gate(answer, {"kf1", "kf2"})

        assert result.status == "insufficient_evidence"
        assert result.evidence_keyframe_ids == ()

    def test_accepts_an_answer_whose_evidence_is_fully_inside_the_set(self) -> None:
        answer = _answer(evidence=["kf1", "kf2"])

        result = apply_grounding_gate(answer, {"kf1", "kf2", "kf3"})

        assert result.status == "answered"
        assert result.answer == "Uỷ ban nhân dân"
        assert result.evidence_keyframe_ids == ("kf1", "kf2")
        assert result.confidence == pytest.approx(0.9)

    def test_confident_answer_citing_nothing_is_not_grounded(self) -> None:
        """An answer with zero evidence has no frame_id, and the AIC QA
        submission line is `<video_id>, <frame_id>, <answer>` -- it literally
        cannot be submitted, so it can never count as grounded."""
        answer = _answer(answer="Hà Nội", confidence=0.95, evidence=[])

        result = apply_grounding_gate(answer, {"kf1", "kf2"})

        assert result.status == "insufficient_evidence"
        assert result.answer == INSUFFICIENT_EVIDENCE_ANSWER

    def test_model_abstention_passes_through_as_insufficient_evidence(self) -> None:
        answer = _answer(answer=INSUFFICIENT_EVIDENCE_ANSWER, confidence=0.05, evidence=[])

        result = apply_grounding_gate(answer, {"kf1"})

        assert result.status == "insufficient_evidence"

    def test_preserves_confidence_and_reasoning_even_when_rejected(self) -> None:
        """`confidence` is a fact about what the model actually said --
        P3.4's Hallucination Rate is exactly "confident AND ungrounded", so
        the gate must not overwrite it on the way out."""
        answer = _answer(confidence=0.97, evidence=["kf_fake"], reasoning="Vì tôi thấy biển hiệu.")

        result = apply_grounding_gate(answer, {"kf1"})

        assert result.confidence == pytest.approx(0.97)
        assert result.reasoning == "Vì tôi thấy biển hiệu."

    def test_empty_retrieved_set_rejects_every_citation(self) -> None:
        result = apply_grounding_gate(_answer(evidence=["kf1"]), set())

        assert result.status == "insufficient_evidence"

    def test_keyframe_id_matching_is_exact_not_prefix(self) -> None:
        result = apply_grounding_gate(_answer(evidence=["kf1"]), {"kf10", "kf11"})

        assert result.status == "insufficient_evidence"


class _FakeSearchService:
    def __init__(self, hits: list[SearchHit] | None = None, error: Exception | None = None) -> None:
        self._hits = [_hit("kf1"), _hit("kf2")] if hits is None else hits
        self._error = error
        self.search_calls: list[tuple[str, str, int]] = []

    async def search(self, query: str, mode: str, top_k: int) -> list[SearchHit]:
        self.search_calls.append((query, mode, top_k))
        if self._error is not None:
            raise self._error
        return self._hits


class _FakeImageLoader:
    def __init__(self, unavailable_ids: set[str] | None = None) -> None:
        self._unavailable_ids = unavailable_ids or set()
        self.loaded_ids: list[str] = []

    async def load_image(self, keyframe_id: str) -> Image.Image:
        if keyframe_id in self._unavailable_ids:
            raise KeyframeImageUnavailable(f"no image for {keyframe_id!r}")
        self.loaded_ids.append(keyframe_id)
        return Image.new("RGB", (8, 8), color=(10, 20, 30))


class _FakeVqaService:
    def __init__(self, answer: VqaAnswer | None = None, error: Exception | None = None) -> None:
        self._answer = answer if answer is not None else _answer()
        self._error = error
        self.answer_calls: list[tuple[str, list]] = []

    async def answer(self, question: str, keyframes: list) -> VqaAnswer:
        self.answer_calls.append((question, keyframes))
        if self._error is not None:
            raise self._error
        return self._answer


def _service(
    search_service: _FakeSearchService | None = None,
    image_loader: _FakeImageLoader | None = None,
    vqa_service: _FakeVqaService | None = None,
    max_keyframes_to_model: int = 8,
) -> QaService:
    return QaService(
        search_service=search_service or _FakeSearchService(),
        image_loader=image_loader or _FakeImageLoader(),
        vqa_service=vqa_service or _FakeVqaService(),
        max_keyframes_to_model=max_keyframes_to_model,
    )


class TestQaServiceHappyPath:
    @pytest.mark.asyncio
    async def test_returns_answered_with_the_keyframe_grid(self) -> None:
        service = _service()

        result = await service.answer_question("Biển hiệu ghi gì?", top_k=20)

        assert result.status == "answered"
        assert result.answer == "Uỷ ban nhân dân"
        assert result.evidence_keyframe_ids == ("kf1",)
        assert [kf.keyframe_id for kf in result.keyframes] == ["kf1", "kf2"]
        assert result.notice is None

    @pytest.mark.asyncio
    async def test_retrieves_with_semantic_mode_and_the_requested_top_k(self) -> None:
        search_service = _FakeSearchService()
        service = _service(search_service=search_service)

        await service.answer_question("câu hỏi", top_k=5)

        assert search_service.search_calls == [("câu hỏi", "semantic", 5)]

    @pytest.mark.asyncio
    async def test_only_retrieved_keyframes_are_handed_to_the_model(self) -> None:
        vqa_service = _FakeVqaService()
        service = _service(vqa_service=vqa_service)

        await service.answer_question("câu hỏi", top_k=20)

        _question, keyframes_sent = vqa_service.answer_calls[0]
        assert [kf.keyframe_id for kf in keyframes_sent] == ["kf1", "kf2"]


class TestQaServiceGroundingIntegration:
    @pytest.mark.asyncio
    async def test_fabricated_evidence_is_rejected_but_the_grid_survives(self) -> None:
        vqa_service = _FakeVqaService(answer=_answer(evidence=["kf_hallucinated"]))
        service = _service(vqa_service=vqa_service)

        result = await service.answer_question("câu hỏi", top_k=20)

        assert result.status == "insufficient_evidence"
        assert result.evidence_keyframe_ids == ()
        assert [kf.keyframe_id for kf in result.keyframes] == ["kf1", "kf2"]

    @pytest.mark.asyncio
    async def test_gate_uses_the_set_actually_sent_not_the_set_retrieved(self) -> None:
        """A keyframe whose image failed to load was never shown to the
        model, so citing it is fabrication -- even though it IS in the
        retrieved grid."""
        search_service = _FakeSearchService(hits=[_hit("kf1"), _hit("kf_no_image")])
        image_loader = _FakeImageLoader(unavailable_ids={"kf_no_image"})
        vqa_service = _FakeVqaService(answer=_answer(evidence=["kf_no_image"]))
        service = _service(
            search_service=search_service, image_loader=image_loader, vqa_service=vqa_service
        )

        result = await service.answer_question("câu hỏi", top_k=20)

        assert result.status == "insufficient_evidence"
        assert [kf.keyframe_id for kf in result.keyframes] == ["kf1", "kf_no_image"]


class TestQaServiceDegradedMode:
    @pytest.mark.asyncio
    async def test_vqa_failure_still_returns_keyframe_grid(self) -> None:
        vqa_service = _FakeVqaService(error=VqaServiceError("Gemini unreachable"))
        service = _service(vqa_service=vqa_service)

        result = await service.answer_question("câu hỏi", top_k=20)

        assert result.status == "unavailable"
        assert result.answer is None
        assert result.notice == VQA_UNAVAILABLE_NOTICE
        assert [kf.keyframe_id for kf in result.keyframes] == ["kf1", "kf2"]

    @pytest.mark.asyncio
    async def test_vqa_timeout_is_the_same_degraded_path(self) -> None:
        vqa_service = _FakeVqaService(
            error=VqaServiceError("VQA timed out", error_code="vqa_timeout", status_code=504)
        )
        service = _service(vqa_service=vqa_service)

        result = await service.answer_question("câu hỏi", top_k=20)

        assert result.status == "unavailable"
        assert result.keyframes != ()

    @pytest.mark.asyncio
    async def test_a_keyframe_with_no_image_is_skipped_not_fatal(self) -> None:
        search_service = _FakeSearchService(hits=[_hit("kf1"), _hit("kf_no_image")])
        image_loader = _FakeImageLoader(unavailable_ids={"kf_no_image"})
        vqa_service = _FakeVqaService()
        service = _service(
            search_service=search_service, image_loader=image_loader, vqa_service=vqa_service
        )

        result = await service.answer_question("câu hỏi", top_k=20)

        assert result.status == "answered"
        _question, keyframes_sent = vqa_service.answer_calls[0]
        assert [kf.keyframe_id for kf in keyframes_sent] == ["kf1"]

    @pytest.mark.asyncio
    async def test_retrieval_returning_nothing_never_calls_the_model(self) -> None:
        """Asking a VLM a question with zero visual context can only produce
        a background-knowledge answer -- exactly what this phase exists to
        prevent (plan §P3 opening)."""
        vqa_service = _FakeVqaService()
        service = _service(search_service=_FakeSearchService(hits=[]), vqa_service=vqa_service)

        result = await service.answer_question("câu hỏi", top_k=20)

        assert result.status == "insufficient_evidence"
        assert result.keyframes == ()
        assert vqa_service.answer_calls == []

    @pytest.mark.asyncio
    async def test_every_image_failing_never_calls_the_model(self) -> None:
        service = _service(image_loader=_FakeImageLoader(unavailable_ids={"kf1", "kf2"}))

        result = await service.answer_question("câu hỏi", top_k=20)

        assert result.status == "insufficient_evidence"
        assert [kf.keyframe_id for kf in result.keyframes] == ["kf1", "kf2"]

    @pytest.mark.asyncio
    async def test_a_search_failure_is_not_swallowed_into_degraded_mode(self) -> None:
        """Degraded mode covers the VQA branch only. A broken retrieval path
        is a real error the operator must see, not a silent empty grid
        (CLAUDE.md §7: thiếu dữ liệu phải fail rõ ràng)."""
        service = _service(search_service=_FakeSearchService(error=RuntimeError("milvus down")))

        with pytest.raises(RuntimeError, match="milvus down"):
            await service.answer_question("câu hỏi", top_k=20)


class TestQaServiceCostControl:
    @pytest.mark.asyncio
    async def test_sends_at_most_max_keyframes_to_the_model(self) -> None:
        hits = [_hit(f"kf{i}") for i in range(10)]
        vqa_service = _FakeVqaService(answer=_answer(evidence=["kf0"]))
        service = _service(
            search_service=_FakeSearchService(hits=hits),
            vqa_service=vqa_service,
            max_keyframes_to_model=3,
        )

        result = await service.answer_question("câu hỏi", top_k=10)

        _question, keyframes_sent = vqa_service.answer_calls[0]
        assert [kf.keyframe_id for kf in keyframes_sent] == ["kf0", "kf1", "kf2"]
        assert len(result.keyframes) == 10, "the grid still shows every retrieved keyframe"

    @pytest.mark.asyncio
    async def test_images_are_only_loaded_for_keyframes_actually_sent(self) -> None:
        hits = [_hit(f"kf{i}") for i in range(10)]
        image_loader = _FakeImageLoader()
        service = _service(
            search_service=_FakeSearchService(hits=hits),
            image_loader=image_loader,
            max_keyframes_to_model=2,
        )

        await service.answer_question("câu hỏi", top_k=10)

        assert image_loader.loaded_ids == ["kf0", "kf1"]

    @pytest.mark.asyncio
    async def test_citing_a_keyframe_in_the_grid_but_beyond_the_cap_is_rejected(self) -> None:
        hits = [_hit(f"kf{i}") for i in range(5)]
        vqa_service = _FakeVqaService(answer=_answer(evidence=["kf4"]))
        service = _service(
            search_service=_FakeSearchService(hits=hits),
            vqa_service=vqa_service,
            max_keyframes_to_model=2,
        )

        result = await service.answer_question("câu hỏi", top_k=5)

        assert result.status == "insufficient_evidence"


class TestThumbnailImageLoader:
    @pytest.mark.asyncio
    async def test_loads_the_image_the_thumbnail_service_resolves(self, tmp_path) -> None:
        image_path = tmp_path / "kf1.jpg"
        Image.new("RGB", (16, 16), color=(1, 2, 3)).save(image_path, format="JPEG")

        class _FakeThumbnailService:
            async def get_thumbnail_path(self, keyframe_id: str):
                return image_path

        loader = ThumbnailImageLoader(_FakeThumbnailService())

        image = await loader.load_image("kf1")

        assert image.size == (16, 16)

    @pytest.mark.asyncio
    async def test_translates_a_resolver_failure_into_keyframe_image_unavailable(self) -> None:
        class _BrokenThumbnailService:
            async def get_thumbnail_path(self, keyframe_id: str):
                raise OSError("source image missing on disk")

        loader = ThumbnailImageLoader(_BrokenThumbnailService())

        with pytest.raises(KeyframeImageUnavailable):
            await loader.load_image("kf1")

    @pytest.mark.asyncio
    async def test_translates_an_undecodable_file_into_keyframe_image_unavailable(
        self, tmp_path
    ) -> None:
        broken_path = tmp_path / "broken.jpg"
        broken_path.write_bytes(b"this is not a jpeg")

        class _FakeThumbnailService:
            async def get_thumbnail_path(self, keyframe_id: str):
                return broken_path

        loader = ThumbnailImageLoader(_FakeThumbnailService())

        with pytest.raises(KeyframeImageUnavailable):
            await loader.load_image("kf1")
