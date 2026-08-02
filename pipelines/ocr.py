"""OCR branch (P0.7 TEXT pipeline) — architecture.md §3.2, references.md §5.1.

    keyframe -> [1] filter frames WITH text -> [2] perceptual-hash dedupe
             -> [3] Gemini OCR (structured output) -> [4] Elasticsearch bulk index

Order is the cost control (PRD risk R2): [1] and [2] must run before any
Gemini call. For this news-broadcast dataset, [1] is a cheap *approximate*
heuristic (Canny edge density) rather than a real text detector — captions
cover most frames anyway, so [2] dedupe (graphics/lower-thirds persisting
across many seconds) is this dataset's actual cost saver, not [1]
(references.md §5.1). An OCR failure for one frame is logged and skipped,
never raised — it must not abort the VISION branch (CLAUDE.md §4).

Every tunable threshold lives in `OcrConfig`, not as module-level magic
numbers (CLAUDE.md §6) — the field defaults here match `configs/ingest/ocr.yaml`,
which is the source of truth a caller should load and pass in.
"""

from __future__ import annotations

import base64
import io
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import cv2
import imagehash
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

from app.stores.es_client import EsStore, OcrDocument

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OcrConfig:
    """Mirrors `configs/ingest/ocr.yaml` — see that file for the rationale
    behind each default."""

    edge_density_threshold: float = 0.02
    hamming_dedupe_threshold: int = 4
    max_ocr_side: int = 384
    tokens_per_image_at_max_side: int = 258
    max_retries: int = 3
    retry_base_delay_seconds: float = 1.0
    gemini_model: str = "gemini-3.6-flash"


_DEFAULT_OCR_CONFIG = OcrConfig()


class OcrExtraction(BaseModel):
    text: str = Field(
        description=(
            "All text visible in the image, transcribed exactly as shown on "
            "screen. Empty string if no text is visible."
        )
    )


class OcrClient(Protocol):
    """Firewall around the Gemini SDK — swappable for a fake in tests, or a
    different OCR backend (PaddleOCR/EasyOCR, references.md §7) later."""

    def extract_text(self, image: Image.Image) -> str: ...


@dataclass(frozen=True)
class OcrCandidate:
    keyframe_id: str
    video_id: str
    timestamp: float
    image: Image.Image


@dataclass(frozen=True)
class OcrRunSummary:
    total_candidates: int
    after_text_filter: int
    after_dedupe: int
    gemini_calls: int
    tokens_used: int
    stopped_on_budget: bool
    failed_keyframe_ids: list[str]


def has_likely_text(image: Image.Image, config: OcrConfig = _DEFAULT_OCR_CONFIG) -> bool:
    """Cheap approximate pre-filter (R2) — Canny edge density above
    `config.edge_density_threshold`. Not a real text detector: false
    positives (busy footage) and false negatives (faint captions) are
    expected and accepted, since [2] dedupe is this dataset's real cost
    saver, not this filter (references.md §5.1)."""
    gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.count_nonzero(edges) / edges.size
    return edge_density >= config.edge_density_threshold


def deduplicate_by_phash(
    candidates: Sequence[OcrCandidate], config: OcrConfig = _DEFAULT_OCR_CONFIG
) -> list[OcrCandidate]:
    """Keeps a candidate only if its perceptual hash differs from every
    already-kept hash by more than `config.hamming_dedupe_threshold` bits."""
    kept: list[OcrCandidate] = []
    kept_hashes: list[imagehash.ImageHash] = []
    for candidate in candidates:
        candidate_hash = imagehash.phash(candidate.image)
        if any(candidate_hash - h <= config.hamming_dedupe_threshold for h in kept_hashes):
            continue
        kept.append(candidate)
        kept_hashes.append(candidate_hash)
    return kept


def _resize_for_ocr(image: Image.Image, config: OcrConfig) -> Image.Image:
    resized = image.copy()
    resized.thumbnail((config.max_ocr_side, config.max_ocr_side))
    return resized


class GeminiOcrClient:
    """Wraps `client.interactions.create(...)` — the current Gemini
    Interactions API (references.md §5.1), never the older `generate_content`.

    google-genai's exact retryable-error exception hierarchy is not verified
    against references.md (only the happy path is documented there) — retries
    on built-in network exceptions for now; narrow this once real API error
    responses have been observed against a live key.
    """

    def __init__(self, api_key: str, config: OcrConfig = _DEFAULT_OCR_CONFIG) -> None:
        from google import genai  # lazy: only needed when this class is used

        self._client = genai.Client(api_key=api_key)
        self._config = config

    def extract_text(self, image: Image.Image) -> str:
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG")
        image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        last_exc: Exception | None = None
        for attempt in range(self._config.max_retries):
            try:
                return self._call_once(image_b64)
            except (TimeoutError, ConnectionError, OSError) as exc:
                last_exc = exc
                time.sleep(self._config.retry_base_delay_seconds * (2**attempt))
        raise RuntimeError(
            f"Gemini OCR failed after {self._config.max_retries} attempts"
        ) from last_exc

    def _call_once(self, image_b64: str) -> str:
        interaction = self._client.interactions.create(
            model=self._config.gemini_model,
            input=[
                {"type": "text", "text": "Extract all visible text from this image, verbatim."},
                {"type": "image", "data": image_b64, "mime_type": "image/jpeg"},
            ],
            response_format={
                "type": "text",
                "mime_type": "application/json",
                "schema": OcrExtraction.model_json_schema(),
            },
        )
        return OcrExtraction.model_validate_json(interaction.output_text).text


async def run_ocr(
    candidates: Sequence[OcrCandidate],
    *,
    ocr_client: OcrClient,
    es: EsStore,
    token_budget: int,
    config: OcrConfig = _DEFAULT_OCR_CONFIG,
) -> OcrRunSummary:
    """Orchestrates the full TEXT branch for one video's keyframes."""
    text_filtered = [c for c in candidates if has_likely_text(c.image, config)]
    deduped = deduplicate_by_phash(text_filtered, config)

    documents: list[OcrDocument] = []
    failed: list[str] = []
    tokens_used = 0
    stopped_on_budget = False
    gemini_calls = 0

    for candidate in deduped:
        if tokens_used + config.tokens_per_image_at_max_side > token_budget:
            stopped_on_budget = True
            logger.warning(
                "OCR token budget (%d) reached after %d call(s) — stopping, not skipping silently.",
                token_budget,
                gemini_calls,
            )
            break
        try:
            text = ocr_client.extract_text(_resize_for_ocr(candidate.image, config))
        except (RuntimeError, ValueError) as exc:
            logger.error("OCR failed for keyframe_id=%s: %s", candidate.keyframe_id, exc)
            failed.append(candidate.keyframe_id)
            continue
        gemini_calls += 1
        tokens_used += config.tokens_per_image_at_max_side
        documents.append(
            OcrDocument(
                keyframe_id=candidate.keyframe_id,
                ocr_text=text,
                video_id=candidate.video_id,
                timestamp=candidate.timestamp,
            )
        )

    await es.bulk_index(documents)

    return OcrRunSummary(
        total_candidates=len(candidates),
        after_text_filter=len(text_filtered),
        after_dedupe=len(deduped),
        gemini_calls=gemini_calls,
        tokens_used=tokens_used,
        stopped_on_budget=stopped_on_budget,
        failed_keyframe_ids=failed,
    )
