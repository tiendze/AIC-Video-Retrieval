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

Grounding-gate verification (checking every `evidence_keyframe_ids` entry is
actually inside the retrieved set) and the Gemini-down -> keyframe-grid
fallback are P3.2, not this module's job yet -- `VqaService.answer()` raises
`VqaServiceError` on any Gemini failure so a later caller decides what
degraded response to show (mirrors `QuestServiceError`/`SearchServiceError`:
an expected, caller-facing failure with its own error_code, not a bare
exception the caller has to guess about).

Retry policy and the lazy `google.genai` import mirror
`pipelines/ocr.py:GeminiOcrClient` and
`app/services/quest_service.py:GeminiQueryRewriter` exactly -- this time both
text AND multiple images go into one `interactions.create` call.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import time
from dataclasses import dataclass
from typing import Any, Protocol

from PIL import Image
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_VQA_PROMPT_TEMPLATE = (
    "Bạn được cung cấp {count} khung hình (keyframe) trích từ video, là TOÀN "
    "BỘ ngữ cảnh thị giác hệ thống đã truy hồi được cho câu hỏi dưới đây. CHỈ "
    "được trả lời dựa trên những gì nhìn thấy được TRONG các khung hình này "
    "-- tuyệt đối không dùng kiến thức nền bên ngoài, kể cả khi bạn biết câu "
    "trả lời từ nguồn khác. Nếu các khung hình không đủ căn cứ để trả lời "
    'chắc chắn, hãy trả answer="Không đủ căn cứ" với confidence thấp và '
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
    """One keyframe SearchService/TrakeService already retrieved, with its
    image already loaded -- this module never resolves keyframe_id -> file
    path itself (that stays in ThumbnailService/MongoStore; wiring the two
    together is P3.2's job)."""

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
            "text": _VQA_PROMPT_TEMPLATE.format(count=len(keyframes), question=question),
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
