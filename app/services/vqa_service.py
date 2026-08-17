"""VQA answering (plan §P3.1 -- `vqa_service.py`, PRD §F7, architecture.md
§3.6). Answers a question using ONLY the keyframes SearchService/TrakeService
already retrieved for it -- the prompt never lets Gemini fall back on its own
background knowledge (CLAUDE.md: a VQA system that answers correctly from the
LLM's background knowledge, not from the video, is useless for the contest).

Structured output via `client.interactions.create(..., response_format=...)`
-- google-genai's current Interactions API (references.md §5.1), never the
older `generate_content`. Schema is `VqaAnswer`:
`{answer, confidence, evidence_keyframe_ids[], reasoning}` (PRD §F7). The
prompt explicitly tells the model it may answer "Không đủ căn cứ" with an
empty `evidence_keyframe_ids` -- this is the anti-hallucination mechanism the
plan's `Hallucination Rate` metric measures; nothing in this module
penalizes or filters that answer out.

Retry policy and the lazy `google.genai` import mirror
`pipelines/ocr.py:GeminiOcrClient` and
`app/services/quest_service.py:GeminiQueryRewriter` exactly -- this time both
text AND multiple images go into one `interactions.create` call.

**Four layers, bottom to top** (architecture.md §6 puts "Gemini + kiểm tra
grounding" in this one file; the orchestration sits here too rather than in
a second module, the same way `quest_service.py` holds both of its
independent branches):

1. `GeminiVqaClient` -- one Gemini call, retry + backoff. Sync.
2. `VqaService` -- async boundary + 10s timeout. Raises `VqaServiceError`.
3. `apply_grounding_gate` -- P3.2's anti-fabrication checkpoint. Pure.
4. `QaService` -- `POST /api/qa` end to end, and the layer that guarantees
   the operator always gets a keyframe grid even when Gemini is down.

The split matters: layers 1-2 know nothing about grounding, and layer 3 is
a pure function over data, so the rule "evidence must be inside the set the
model was shown" is testable without any of the I/O above it.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import time
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from PIL import Image
from pydantic import BaseModel, Field

from app.services.search_service import SearchHit
from app.services.thumbnail_service import ThumbnailServiceError

logger = logging.getLogger(__name__)

# The exact string the model is told to answer with when the keyframes do
# not support a confident answer, AND the string the gate substitutes when
# it rejects an answer. One constant, because drift between the prompt's
# wording and the gate's wording would make abstentions unrecognisable to
# P3.4's Abstention Rate without any test failing.
INSUFFICIENT_EVIDENCE_ANSWER = "Không đủ căn cứ"

VQA_UNAVAILABLE_NOTICE = "VQA tạm không khả dụng — chỉ hiển thị lưới keyframe truy hồi được."

_VQA_PROMPT_TEMPLATE = (
    "Bạn được cung cấp {count} khung hình (keyframe) trích từ video, là TOÀN "
    "BỘ ngữ cảnh thị giác hệ thống đã truy hồi được cho câu hỏi dưới đây. CHỈ "
    "được trả lời dựa trên những gì nhìn thấy được TRONG các khung hình này "
    "-- tuyệt đối không dùng kiến thức nền bên ngoài, kể cả khi bạn biết câu "
    "trả lời từ nguồn khác. Nếu các khung hình không đủ căn cứ để trả lời "
    'chắc chắn, hãy trả answer="{insufficient}" với confidence thấp và '
    "evidence_keyframe_ids rỗng thay vì đoán.\n\n"
    "Mỗi khung hình được gắn nhãn keyframe_id ngay trước ảnh của nó -- khi "
    "trích dẫn bằng chứng trong evidence_keyframe_ids, chỉ được dùng đúng "
    "các keyframe_id này.\n\n"
    "Câu hỏi: {question}"
)


class VqaAnswer(BaseModel):
    """PRD §F7's exact output schema. `evidence_keyframe_ids` defaults to an
    empty list -- required for the insufficient-evidence path, not just
    optional sugar."""

    answer: str = Field(
        description=(
            "Câu trả lời bằng tiếng Việt hoặc tiếng Anh, dựa hoàn toàn trên "
            'các khung hình đã cấp. "Không đủ căn cứ" nếu không thể trả lời '
            "chắc chắn từ các khung hình này."
        )
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Độ tự tin vào câu trả lời, 0.0-1.0.")
    evidence_keyframe_ids: list[str] = Field(
        default_factory=list,
        description=(
            "keyframe_id của các khung hình làm căn cứ cho câu trả lời -- "
            'rỗng nếu answer là "Không đủ căn cứ".'
        ),
    )
    reasoning: str = Field(
        description="Giải thích ngắn gọn vì sao các khung hình trên dẫn tới câu trả lời."
    )


@dataclass(frozen=True)
class RetrievedKeyframe:
    """One keyframe SearchService already retrieved, with its image loaded.
    Resolving keyframe_id -> file path stays in ThumbnailService/MongoStore;
    `ThumbnailImageLoader` below is the only place the two meet."""

    keyframe_id: str
    image: Image.Image


class VqaServiceError(Exception):
    """Raised when Gemini fails after exhausting retries, times out, or
    returns output that doesn't match `VqaAnswer`'s schema -- an expected,
    caller-facing failure (mirrors QuestServiceError/SearchServiceError),
    not a bare exception a caller has to guess about."""

    def __init__(
        self, message: str, error_code: str = "vqa_unavailable", status_code: int = 502
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.status_code = status_code


def _image_to_jpeg_b64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _build_input_parts(question: str, keyframes: list[RetrievedKeyframe]) -> list[dict[str, Any]]:
    """Pure, side-effect-free -- `keyframes` (exactly the list the caller
    retrieved) is the only source of images this function can reach for, so
    the resulting prompt structurally cannot reference a keyframe the caller
    didn't pass in."""
    parts: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": _VQA_PROMPT_TEMPLATE.format(
                count=len(keyframes),
                question=question,
                insufficient=INSUFFICIENT_EVIDENCE_ANSWER,
            ),
        }
    ]
    for keyframe in keyframes:
        parts.append({"type": "text", "text": f"[keyframe_id={keyframe.keyframe_id}]"})
        parts.append(
            {
                "type": "image",
                "data": _image_to_jpeg_b64(keyframe.image),
                "mime_type": "image/jpeg",
            }
        )
    return parts


@dataclass(frozen=True)
class VqaConfig:
    """Mirrors `configs/qa/vqa.yaml` -- see that file for the rationale
    behind each default."""

    gemini_model: str = "gemini-3.6-flash"
    max_retries: int = 3
    retry_base_delay_seconds: float = 1.0
    timeout_seconds: float = 10.0
    max_keyframes_to_model: int = 8


_DEFAULT_VQA_CONFIG = VqaConfig()


class VqaClient(Protocol):
    """Firewall around the Gemini SDK -- swappable for a fake in tests."""

    def answer(self, question: str, keyframes: list[RetrievedKeyframe]) -> VqaAnswer: ...


class GeminiVqaClient:
    """Wraps `client.interactions.create(...)` (references.md §5.1). Retry
    policy mirrors `GeminiOcrClient`/`GeminiQueryRewriter` exactly: retry on
    built-in network exceptions, exponential backoff, raise `RuntimeError`
    after `max_retries` attempts -- this SDK's own retryable-error hierarchy
    isn't verified against a live key yet, only the happy path is documented
    in references.md."""

    def __init__(self, api_key: str, config: VqaConfig = _DEFAULT_VQA_CONFIG) -> None:
        from google import genai  # lazy: only needed when this class is used

        self._client = genai.Client(api_key=api_key)
        self._config = config

    def answer(self, question: str, keyframes: list[RetrievedKeyframe]) -> VqaAnswer:
        last_exc: Exception | None = None
        for attempt in range(self._config.max_retries):
            try:
                return self._call_once(question, keyframes)
            except (TimeoutError, ConnectionError, OSError) as exc:
                last_exc = exc
                time.sleep(self._config.retry_base_delay_seconds * (2**attempt))
        raise RuntimeError(
            f"Gemini VQA failed after {self._config.max_retries} attempts"
        ) from last_exc

    def _call_once(self, question: str, keyframes: list[RetrievedKeyframe]) -> VqaAnswer:
        interaction = self._client.interactions.create(
            model=self._config.gemini_model,
            input=_build_input_parts(question, keyframes),
            response_format={
                "type": "text",
                "mime_type": "application/json",
                "schema": VqaAnswer.model_json_schema(),
            },
        )
        return VqaAnswer.model_validate_json(interaction.output_text)


class VqaService:
    """Async boundary in front of `VqaClient` -- the sync SDK call runs off
    the event loop via `asyncio.to_thread`, wrapped in `asyncio.wait_for` for
    the 10s timeout (.claude/rules/ecc/python/fastapi.md). Raises
    `VqaServiceError` on any failure; deciding what degraded response to
    show (keyframe grid + notice) is P3.2's grounding-gate layer, not this
    service."""

    def __init__(
        self, vqa_client: VqaClient, timeout_sec: float = _DEFAULT_VQA_CONFIG.timeout_seconds
    ) -> None:
        self._vqa_client = vqa_client
        self._timeout_sec = timeout_sec

    async def answer(self, question: str, keyframes: list[RetrievedKeyframe]) -> VqaAnswer:
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(self._vqa_client.answer, question, keyframes),
                timeout=self._timeout_sec,
            )
        except asyncio.TimeoutError as exc:
            logger.warning("vqa: Gemini call exceeded %.1fs timeout", self._timeout_sec)
            raise VqaServiceError(
                "VQA timed out", error_code="vqa_timeout", status_code=504
            ) from exc
        except (RuntimeError, ValueError) as exc:
            logger.warning("vqa: Gemini call failed: %r", exc)
            raise VqaServiceError(
                f"VQA failed: {exc}", error_code="vqa_unavailable", status_code=502
            ) from exc


# --------------------------------------------------------------------------
# P3.2 -- grounding gate (see module docstring)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GroundedAnswer:
    """The gate's verdict. `confidence` and `reasoning` are passed through
    UNCHANGED even on rejection: they are facts about what the model actually
    said, and P3.4's `Hallucination Rate` is defined as exactly "confident
    AND ungrounded" -- overwriting them here would erase the very signal that
    metric measures."""

    status: Literal["answered", "insufficient_evidence"]
    answer: str
    confidence: float
    evidence_keyframe_ids: tuple[str, ...]
    reasoning: str


def apply_grounding_gate(answer: VqaAnswer, keyframe_ids_shown: Collection[str]) -> GroundedAnswer:
    """The anti-fabrication checkpoint (plan §P3.2, architecture.md §3.6).

    `keyframe_ids_shown` must be the set of keyframes ACTUALLY handed to the
    model, not the set retrieved -- a keyframe that was retrieved but whose
    image failed to load was never visible to the model, so citing it is
    fabrication just the same.

    An answer survives only if it cites at least one keyframe AND every
    citation is inside that set:

    - **Any** fabricated id rejects the **whole** answer. A model that
      invents one citation is fabricating; the remaining citations earn no
      benefit of the doubt.
    - **Zero** citations is also rejected, including for a confident answer.
      The AIC QA submission line is `<video_id>, <frame_id>, <answer>`
      (references.md §1.2) -- an answer with no evidence keyframe has no
      `frame_id` and literally cannot be submitted, so it can never count
      toward `Grounded Accuracy`.

    No `confidence` threshold is applied here on purpose: thresholding is
    P3.4's job (`configs/`-driven, per the plan), and a gate that silently
    dropped low-confidence-but-correctly-grounded answers would corrupt
    `Abstention Rate`.
    """
    allowed = set(keyframe_ids_shown)
    cited = tuple(answer.evidence_keyframe_ids)
    fabricated = [keyframe_id for keyframe_id in cited if keyframe_id not in allowed]

    if fabricated:
        logger.warning(
            "grounding gate: rejecting answer -- cited keyframe_id(s) %s were never "
            "shown to the model (shown: %d keyframe(s))",
            fabricated,
            len(allowed),
        )
    if not cited or fabricated:
        return GroundedAnswer(
            status="insufficient_evidence",
            answer=INSUFFICIENT_EVIDENCE_ANSWER,
            confidence=answer.confidence,
            evidence_keyframe_ids=(),
            reasoning=answer.reasoning,
        )
    return GroundedAnswer(
        status="answered",
        answer=answer.answer,
        confidence=answer.confidence,
        evidence_keyframe_ids=cited,
        reasoning=answer.reasoning,
    )


# --------------------------------------------------------------------------
# P3.2 -- keyframe images
# --------------------------------------------------------------------------


class KeyframeImageUnavailable(Exception):
    """One keyframe's image could not be loaded. Deliberately its own type so
    `QaService` can skip exactly this case (log + drop that keyframe) without
    catching anything broader (CLAUDE.md §7)."""


class KeyframeImageLoader(Protocol):
    async def load_image(self, keyframe_id: str) -> Image.Image: ...


class _ThumbnailPathResolver(Protocol):
    async def get_thumbnail_path(self, keyframe_id: str) -> Path: ...


def _open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


class ThumbnailImageLoader:
    """Adapts `ThumbnailService` to `KeyframeImageLoader`.

    Reading the THUMBNAIL rather than the full-res source is a cost decision,
    not an accident: the live corpus is 221,903 full-res JPEGs (~40 GB,
    `thumbnail_service.py`), and VQA sends several images per single
    question. `configs/search/search.yaml`'s `thumbnail_max_px` already
    bounds them, so the resize that `pipelines/ocr.py` does inline
    (`max_ocr_side`) is here obtained for free from a cache the search grid
    is warming anyway.

    Translating every failure into `KeyframeImageUnavailable` is the point of
    this class -- `QaService` must not have to know ThumbnailService's
    exception vocabulary.
    """

    def __init__(self, thumbnail_service: _ThumbnailPathResolver) -> None:
        self._thumbnail_service = thumbnail_service

    async def load_image(self, keyframe_id: str) -> Image.Image:
        try:
            path = await self._thumbnail_service.get_thumbnail_path(keyframe_id)
            return await asyncio.to_thread(_open_rgb, path)
        except (ThumbnailServiceError, OSError, Image.DecompressionBombError) as exc:
            raise KeyframeImageUnavailable(
                f"No usable image for keyframe_id={keyframe_id!r}: {exc}"
            ) from exc


# --------------------------------------------------------------------------
# P3.2 -- QaService (POST /api/qa orchestration)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class QaResult:
    """`keyframes` is the retrieved grid and is populated in EVERY status,
    including `unavailable` -- that is this type's whole reason for existing
    (plan §P3.2: a Gemini outage must not kill the main request; US-5 stays
    partially usable). `answer`/`confidence`/`reasoning` are None only when
    no model verdict exists at all."""

    status: Literal["answered", "insufficient_evidence", "unavailable"]
    answer: str | None
    confidence: float | None
    evidence_keyframe_ids: tuple[str, ...]
    reasoning: str | None
    keyframes: tuple[SearchHit, ...]
    notice: str | None


class _SearchServiceLike(Protocol):
    async def search(self, query: str, mode: str, top_k: int) -> list[SearchHit]: ...


class _VqaServiceLike(Protocol):
    async def answer(self, question: str, keyframes: list[RetrievedKeyframe]) -> VqaAnswer: ...


class QaService:
    """`POST /api/qa` end to end (architecture.md §3.6): retrieve keyframes
    for the question, load their images, ask the VLM, then run the grounding
    gate over what comes back.

    Two failure modes are treated very differently on purpose:

    - **VQA branch fails** (Gemini down/timeout/malformed) -> degraded
      result: the keyframe grid plus a readable notice, never an exception.
    - **Retrieval fails** -> propagates. A broken Milvus is not a partial
      answer; returning an empty grid there would look identical to "nothing
      matched" and quietly mislead an operator being timed (CLAUDE.md §7:
      thiếu dữ liệu phải fail rõ ràng, không trả rỗng).

    The model is never asked a question with zero visual context -- with no
    images it could only answer from background knowledge, which is exactly
    what this phase exists to prevent.
    """

    def __init__(
        self,
        search_service: _SearchServiceLike,
        image_loader: KeyframeImageLoader,
        vqa_service: _VqaServiceLike,
        max_keyframes_to_model: int = _DEFAULT_VQA_CONFIG.max_keyframes_to_model,
    ) -> None:
        self._search_service = search_service
        self._image_loader = image_loader
        self._vqa_service = vqa_service
        self._max_keyframes_to_model = max_keyframes_to_model

    async def answer_question(self, question: str, top_k: int) -> QaResult:
        hits = await self._search_service.search(question, mode="semantic", top_k=top_k)
        grid = tuple(hits)
        if not hits:
            logger.info(
                "qa: no keyframe retrieved for question=%r -- not calling the model", question
            )
            return self._no_context(grid)

        keyframes = await self._load_keyframes(hits[: self._max_keyframes_to_model])
        if not keyframes:
            logger.warning(
                "qa: every candidate keyframe image failed to load -- not calling the model"
            )
            return self._no_context(grid)

        try:
            answer = await self._vqa_service.answer(question, keyframes)
        except VqaServiceError as exc:
            logger.warning("qa: degraded -- returning keyframe grid only: %r", exc)
            return QaResult(
                status="unavailable",
                answer=None,
                confidence=None,
                evidence_keyframe_ids=(),
                reasoning=None,
                keyframes=grid,
                notice=VQA_UNAVAILABLE_NOTICE,
            )

        grounded = apply_grounding_gate(answer, {kf.keyframe_id for kf in keyframes})
        return QaResult(
            status=grounded.status,
            answer=grounded.answer,
            confidence=grounded.confidence,
            evidence_keyframe_ids=grounded.evidence_keyframe_ids,
            reasoning=grounded.reasoning,
            keyframes=grid,
            notice=None,
        )

    async def _load_keyframes(self, hits: Sequence[SearchHit]) -> list[RetrievedKeyframe]:
        keyframes: list[RetrievedKeyframe] = []
        for hit in hits:
            try:
                image = await self._image_loader.load_image(hit.keyframe_id)
            except KeyframeImageUnavailable as exc:
                # One missing image must not fail the question -- mirrors
                # SearchService._search_ocr dropping a frame_idx-less hit.
                logger.warning("qa: skipping keyframe_id=%s -- %s", hit.keyframe_id, exc)
                continue
            keyframes.append(RetrievedKeyframe(keyframe_id=hit.keyframe_id, image=image))
        return keyframes

    @staticmethod
    def _no_context(grid: tuple[SearchHit, ...]) -> QaResult:
        return QaResult(
            status="insufficient_evidence",
            answer=INSUFFICIENT_EVIDENCE_ANSWER,
            confidence=None,
            evidence_keyframe_ids=(),
            reasoning=None,
            keyframes=grid,
            notice=None,
        )
