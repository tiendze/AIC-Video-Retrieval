"""Request shape for `POST /api/trake` (candidate-chaining pipeline).

Each event row carries its own Text / OCR / Title / ASR boxes, mirroring the
KIS Stage 1 form (`app/schemas/request/search.py: KisCandidateRequest`) -- TRAKE
runs one `retrieve_kis_candidates()` call per event, so every knob KIS exposes
is meaningful per event here too.

Bounds come from configs/search/trake.yaml and configs/search/search.yaml,
never hardcoded (CLAUDE.md §6).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from app.schemas.request.search import (
    MAX_ASR_QUERY_LENGTH,
    MAX_CANDIDATE_K,
    _normalize_local_phrase_group,
)

_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent / "configs" / "search" / "trake.yaml"
)


def _load_config() -> dict:
    with _CONFIG_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


_config = _load_config()
DEFAULT_TOP_K: int = _config["default_top_k"]
MAX_TOP_K: int = _config["max_top_k"]
DEFAULT_SCORE_THRESHOLD: float = _config["default_score_threshold"]
DEFAULT_EVENT_TEXT_TOP_K: int = _config["event_text_top_k"]
DEFAULT_EVENT_OCR_TOP_K: int = _config["event_ocr_top_k"]
DEFAULT_EVENT_TITLE_TOP_K: int = _config["event_title_top_k"]
DEFAULT_EVENT_ASR_VIDEO_TOP_K: int = _config["event_asr_video_top_k"]


def _blank_to_none(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


class TrakeEventRequest(BaseModel):
    """One event of the ordered sequence.

    `text_query` and `text_queries` coexist for the same reason
    `KisCandidateRequest` keeps both: a single box is the common case, several
    boxes are prompt-ensembling (encoded and averaged, see
    `TrakeService._encode_events`). Same for `asr_query` / `asr_queries`.

    `local_phrases` is ONE list per event, not per text query: a phrase
    describes the visual region of the moment, not a wording of it, so
    `TrakeService._retrieve_one_pool` fans it out to every text query of the
    event when it calls KIS (`SearchService.retrieve_kis_candidates`).
    """

    text_query: str | None = None
    text_queries: list[str] = Field(default_factory=list)
    text_top_k: int = Field(default=DEFAULT_EVENT_TEXT_TOP_K, ge=1, le=MAX_CANDIDATE_K)
    local_phrases: list[str] = Field(default_factory=list)
    ocr_query: str | None = None
    ocr_top_k: int = Field(default=DEFAULT_EVENT_OCR_TOP_K, ge=1, le=MAX_CANDIDATE_K)
    title_query: str | None = None
    title_top_k: int = Field(default=DEFAULT_EVENT_TITLE_TOP_K, ge=1, le=MAX_CANDIDATE_K)
    asr_query: str | None = Field(default=None, max_length=MAX_ASR_QUERY_LENGTH)
    asr_queries: list[str] = Field(default_factory=list)
    asr_match_mode: Literal["any", "all"] = "any"
    asr_video_top_k: int = Field(default=DEFAULT_EVENT_ASR_VIDEO_TOP_K, ge=1)

    @field_validator("text_query", "ocr_query", "title_query", "asr_query", mode="before")
    @classmethod
    def empty_strings_are_absent(cls, value: str | None) -> str | None:
        return _blank_to_none(value)

    @field_validator("text_queries", "asr_queries", "local_phrases", mode="before")
    @classmethod
    def accepts_single_string(cls, value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return value  # type: ignore[return-value]

    @model_validator(mode="after")
    def at_least_one_channel(self) -> TrakeEventRequest:
        self.text_queries = _merge_queries(self.text_query, self.text_queries)
        self.text_query = self.text_queries[0] if self.text_queries else None
        # A phrase pins a visual region of the event's moment; without a text
        # query there is no KIS Stage 1 text call for it to attach to, so an
        # operator typing into a disabled-looking box does not silently do
        # nothing on the server too -- it is dropped here, deterministically.
        self.local_phrases = (
            _normalize_local_phrase_group(self.local_phrases) if self.text_queries else []
        )
        self.asr_queries = _merge_queries(self.asr_query, self.asr_queries)
        for query in self.asr_queries:
            if len(query) > MAX_ASR_QUERY_LENGTH:
                raise ValueError("ASR query must be at most max_asr_query_length")
        self.asr_query = self.asr_queries[0] if self.asr_queries else None
        if (
            not self.text_queries
            and self.ocr_query is None
            and self.title_query is None
            and not self.asr_queries
        ):
            raise ValueError("Each event requires a Text, OCR, Title or ASR query")
        return self


def _merge_queries(single: str | None, many: list[str]) -> list[str]:
    merged: list[str] = []
    if single is not None:
        merged.append(single)
    for query in many:
        stripped = query.strip()
        if stripped:
            merged.append(stripped)
    return list(dict.fromkeys(merged))


class TrakeRequest(BaseModel):
    """N event rows, in time order. The order IS the array index -- the
    response's keyframes come back in the same order and are never re-sorted
    (CLAUDE.md §4)."""

    events: list[TrakeEventRequest] = Field(min_length=1)
    threshold: float = DEFAULT_SCORE_THRESHOLD
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=MAX_TOP_K)
