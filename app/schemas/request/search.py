"""Request shape for `POST /api/search` (plan §P0.8). `top_k` bounds come
from configs/search/search.yaml, never hardcoded (CLAUDE.md §6)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent / "configs" / "search" / "search.yaml"
)


def _load_config() -> dict:
    with _CONFIG_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


_config = _load_config()
DEFAULT_TOP_K: int = _config["default_top_k"]
MAX_TOP_K: int = _config["max_top_k"]
DEFAULT_TEXT_CANDIDATE_K: int = _config["default_text_candidate_k"]
DEFAULT_OCR_CANDIDATE_K: int = _config["default_ocr_candidate_k"]
MAX_CANDIDATE_K: int = _config["max_candidate_k"]
MAX_UPLOAD_SIZE_BYTES: int = _config["max_upload_size_bytes"]
ALLOWED_UPLOAD_MIME_TYPES: frozenset[str] = frozenset(_config["allowed_upload_mime_types"])


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    mode: Literal["semantic", "ocr"]
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=MAX_TOP_K)


class KisQueryRequest(BaseModel):
    type: Literal["text", "ocr"]
    text: str = Field(min_length=1)

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("query text must not be blank")
        return stripped


class KisSearchRequest(BaseModel):
    queries: list[KisQueryRequest] = Field(min_length=1)
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=MAX_TOP_K)


def _blank_to_none(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


class KisCandidateRequest(BaseModel):
    text_query: str | None = None
    text_top_k: int = Field(default=DEFAULT_TEXT_CANDIDATE_K, ge=1, le=MAX_CANDIDATE_K)
    ocr_query: str | None = None
    ocr_top_k: int = Field(default=DEFAULT_OCR_CANDIDATE_K, ge=1, le=MAX_CANDIDATE_K)

    @field_validator("text_query", "ocr_query", mode="before")
    @classmethod
    def empty_strings_are_absent(cls, value: str | None) -> str | None:
        return _blank_to_none(value)

    @model_validator(mode="after")
    def at_least_one_stage1_query(self) -> KisCandidateRequest:
        if self.text_query is None and self.ocr_query is None:
            raise ValueError("Stage 1 requires Text Q0 or OCR query")
        return self


class KisRerankRequest(BaseModel):
    candidate_ids: list[str] = Field(min_length=1)
    subqueries: list[KisQueryRequest] = Field(min_length=1)
    final_top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=MAX_TOP_K)

    @field_validator("candidate_ids")
    @classmethod
    def candidate_ids_must_be_nonblank_and_unique(cls, value: list[str]) -> list[str]:
        unique_ids: list[str] = []
        seen: set[str] = set()
        for keyframe_id in value:
            stripped = keyframe_id.strip()
            if not stripped:
                raise ValueError("candidate_ids cannot contain blank ids")
            if stripped not in seen:
                seen.add(stripped)
                unique_ids.append(stripped)
        return unique_ids

    @model_validator(mode="after")
    def final_top_k_must_fit_candidate_pool(self) -> KisRerankRequest:
        if self.final_top_k > len(self.candidate_ids):
            raise ValueError("final_top_k must be <= candidate_pool_size")
        return self
