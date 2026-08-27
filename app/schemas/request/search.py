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
DEFAULT_TITLE_CANDIDATE_K: int = _config.get("default_title_candidate_k", DEFAULT_OCR_CANDIDATE_K)
DEFAULT_ASR_VIDEO_CANDIDATE_K: int = _config["default_asr_video_candidate_k"]
MAX_ASR_VIDEO_CANDIDATE_K: int = _config["max_asr_video_candidate_k"]
MAX_ASR_QUERY_LENGTH: int = _config["max_asr_query_length"]
MAX_UPLOAD_SIZE_BYTES: int = _config["max_upload_size_bytes"]
ALLOWED_UPLOAD_MIME_TYPES: frozenset[str] = frozenset(_config["allowed_upload_mime_types"])


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    mode: Literal["semantic", "ocr"]
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=MAX_TOP_K)


class AbePairRequest(BaseModel):
    attribute: str = Field(min_length=1)
    object: str = Field(min_length=1)

    @field_validator("attribute", "object")
    @classmethod
    def pair_fields_must_not_be_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("ABE pair fields must not be blank")
        return stripped


class KisQueryRequest(BaseModel):
    type: Literal["text", "ocr", "abe"]
    text: str | None = None
    pairs: list[AbePairRequest] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_by_query_type(self) -> KisQueryRequest:
        if self.type in {"text", "ocr"}:
            stripped = (self.text or "").strip()
            if not stripped:
                raise ValueError("query text must not be blank")
            self.text = stripped
            self.pairs = []
            return self

        valid_pairs = [
            pair for pair in self.pairs if pair.attribute.strip() and pair.object.strip()
        ]
        if not valid_pairs:
            raise ValueError("ABE query requires at least one valid attribute-object pair")
        self.text = ""
        self.pairs = valid_pairs
        return self


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
    text_queries: list[str] = Field(default_factory=list)
    local_phrases_by_query: list[list[str]] = Field(default_factory=list)
    text_top_k: int = Field(default=DEFAULT_TEXT_CANDIDATE_K, ge=1)
    ocr_query: str | None = None
    ocr_top_k: int = Field(default=DEFAULT_OCR_CANDIDATE_K, ge=1, le=MAX_CANDIDATE_K)
    title_query: str | None = None
    title_top_k: int = Field(default=DEFAULT_TITLE_CANDIDATE_K, ge=1, le=MAX_CANDIDATE_K)
    asr_query: str | None = Field(default=None, max_length=MAX_ASR_QUERY_LENGTH)
    asr_queries: list[str] = Field(default_factory=list)
    asr_match_mode: Literal["any", "all"] = "any"
    asr_video_top_k: int = Field(
        default=DEFAULT_ASR_VIDEO_CANDIDATE_K,
        ge=1,
        le=MAX_ASR_VIDEO_CANDIDATE_K,
    )

    @field_validator("text_query", "ocr_query", "title_query", "asr_query", mode="before")
    @classmethod
    def empty_strings_are_absent(cls, value: str | None) -> str | None:
        return _blank_to_none(value)

    @field_validator("text_queries", mode="before")
    @classmethod
    def text_queries_accepts_single_string(cls, value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return value  # type: ignore[return-value]

    @field_validator("local_phrases_by_query", mode="before")
    @classmethod
    def local_phrases_accepts_absent_or_single_group(cls, value: object) -> list[list[str]]:
        if value is None:
            return []
        if isinstance(value, str):
            return [[value]]
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            return [value]
        return value  # type: ignore[return-value]

    @field_validator("asr_queries", mode="before")
    @classmethod
    def asr_queries_accepts_single_string(cls, value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return value  # type: ignore[return-value]

    @model_validator(mode="after")
    def at_least_one_stage1_query(self) -> KisCandidateRequest:
        text_entries: list[tuple[str, list[str]]] = []
        local_groups = [
            _normalize_local_phrase_group(group) for group in self.local_phrases_by_query
        ]
        if self.text_queries:
            if self.text_query is not None and (
                not self.text_queries or self.text_query.strip() != self.text_queries[0].strip()
            ):
                text_entries.append((self.text_query, []))
            for index, query in enumerate(self.text_queries):
                phrases = local_groups[index] if index < len(local_groups) else []
                text_entries.append((query, phrases))
        elif self.text_query is not None:
            text_entries.append((self.text_query, local_groups[0] if local_groups else []))

        normalized_text_queries: list[str] = []
        normalized_local_phrases: list[list[str]] = []
        query_index_by_text: dict[str, int] = {}
        for query, phrases in text_entries:
            stripped = query.strip()
            if not stripped:
                continue
            existing_index = query_index_by_text.get(stripped)
            if existing_index is None:
                query_index_by_text[stripped] = len(normalized_text_queries)
                normalized_text_queries.append(stripped)
                normalized_local_phrases.append(list(phrases))
            else:
                normalized_local_phrases[existing_index] = _merge_local_phrases(
                    normalized_local_phrases[existing_index], phrases
                )
        self.text_queries = normalized_text_queries
        self.local_phrases_by_query = normalized_local_phrases
        self.text_query = self.text_queries[0] if self.text_queries else None
        normalized_asr_queries: list[str] = []
        if self.asr_query is not None:
            normalized_asr_queries.append(self.asr_query)
        for query in self.asr_queries:
            stripped = query.strip()
            if stripped:
                if len(stripped) > MAX_ASR_QUERY_LENGTH:
                    raise ValueError("ASR query must be at most max_asr_query_length")
                normalized_asr_queries.append(stripped)
        self.asr_queries = list(dict.fromkeys(normalized_asr_queries))
        self.asr_query = self.asr_queries[0] if self.asr_queries else None
        if (
            self.text_query is None
            and self.ocr_query is None
            and self.title_query is None
            and not self.asr_queries
        ):
            raise ValueError("Stage 1 requires Text, OCR, Title, or ASR query")
        return self


def _normalize_local_phrase_group(phrases: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for phrase in phrases:
        stripped = phrase.strip()
        if not stripped:
            continue
        key = stripped.casefold()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(stripped)
    return normalized


def _merge_local_phrases(left: list[str], right: list[str]) -> list[str]:
    merged = list(left)
    seen = {phrase.casefold() for phrase in merged}
    for phrase in right:
        key = phrase.casefold()
        if key not in seen:
            seen.add(key)
            merged.append(phrase)
    return merged


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
