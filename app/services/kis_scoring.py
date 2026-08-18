"""Pure scoring utilities for Textual KIS V1.

The online retrieval service owns candidate generation and store access; this
module owns the deterministic math that must stay easy to test and inspect.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from enum import Enum

import numpy as np

try:  # RapidFuzz is the intended lightweight scorer; tests can run without it.
    from rapidfuzz import fuzz as _rapidfuzz_fuzz
except ImportError:  # pragma: no cover - exercised only in dependency-light envs.
    _rapidfuzz_fuzz = None


class QueryType(str, Enum):
    TEXT = "text"
    OCR = "ocr"


@dataclass(frozen=True)
class KisQuery:
    query_index: int
    type: QueryType
    text: str


@dataclass(frozen=True)
class KisConfig:
    candidate_multiplier: int = 5
    candidate_min: int = 100
    geomean_epsilon: float = 1e-8
    bm25_percentile: float = 95.0
    bm25_percentile_min_samples: int = 20
    ocr_short_token_max_length: int = 3
    ocr_short_token_threshold: float = 0.95
    ocr_long_token_threshold: float = 0.8
    ocr_ngram_max_tokens: int = 2

    def __post_init__(self) -> None:
        if self.candidate_multiplier < 2:
            raise ValueError("kis_candidate_multiplier must be at least 2")
        if self.candidate_min < 1:
            raise ValueError("kis_candidate_min must be positive")
        if not 0 < self.geomean_epsilon < 1:
            raise ValueError("kis_geomean_epsilon must be in (0, 1)")
        if not 0 < self.bm25_percentile <= 100:
            raise ValueError("kis_bm25_percentile must be in (0, 100]")
        if self.bm25_percentile_min_samples < 1:
            raise ValueError("kis_bm25_percentile_min_samples must be positive")
        for name, threshold in (
            ("kis_ocr_short_token_threshold", self.ocr_short_token_threshold),
            ("kis_ocr_long_token_threshold", self.ocr_long_token_threshold),
        ):
            if not 0 <= threshold <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.ocr_short_token_max_length < 1:
            raise ValueError("kis_ocr_short_token_max_length must be positive")
        if self.ocr_ngram_max_tokens < 1:
            raise ValueError("kis_ocr_ngram_max_tokens must be positive")

    @classmethod
    def from_mapping(cls, config: Mapping[str, object]) -> KisConfig:
        return cls(
            candidate_multiplier=int(config.get("kis_candidate_multiplier", 5)),
            candidate_min=int(config.get("kis_candidate_min", 100)),
            geomean_epsilon=float(config.get("kis_geomean_epsilon", 1e-8)),
            bm25_percentile=float(config.get("kis_bm25_percentile", 95.0)),
            bm25_percentile_min_samples=int(config.get("kis_bm25_percentile_min_samples", 20)),
            ocr_short_token_max_length=int(config.get("kis_ocr_short_token_max_length", 3)),
            ocr_short_token_threshold=float(config.get("kis_ocr_short_token_threshold", 0.95)),
            ocr_long_token_threshold=float(config.get("kis_ocr_long_token_threshold", 0.8)),
            ocr_ngram_max_tokens=int(config.get("kis_ocr_ngram_max_tokens", 2)),
        )


@dataclass(frozen=True)
class TextQueryScore:
    query_index: int
    query: str
    raw_cosine: float
    rank: int
    rank_percentile: float


@dataclass(frozen=True)
class OcrQueryScore:
    query_index: int
    query: str
    raw_bm25: float
    bm25_reference: float
    bm25_normalized: float
    coverage: float
    query_score: float
    rank: int
    rank_percentile: float


@dataclass(frozen=True)
class TextModalityScore:
    geometric_mean: float
    score: float
    queries: list[TextQueryScore]


@dataclass(frozen=True)
class OcrModalityScore:
    geometric_mean: float
    score: float
    queries: list[OcrQueryScore]


@dataclass(frozen=True)
class Stage1TextScore:
    query: str
    raw_cosine: float
    rank: int


@dataclass(frozen=True)
class Stage1OcrScore:
    query: str
    raw_bm25: float
    bm25_reference: float
    bm25_normalized: float
    coverage: float
    query_score: float
    rank: int


@dataclass(frozen=True)
class Stage1RetrievalScore:
    source_modalities: list[str]
    text: Stage1TextScore | None = None
    ocr: Stage1OcrScore | None = None


@dataclass(frozen=True)
class RankedScore:
    raw_score: float
    rank: int
    rank_percentile: float


def candidate_depth(top_k: int, config: KisConfig) -> int:
    return max(top_k * config.candidate_multiplier, config.candidate_min)


def clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def clamp01(value: float) -> float:
    return clamp(value, 0.0, 1.0)


def geometric_mean_score(scores: Sequence[float], config: KisConfig) -> float:
    if not scores:
        raise ValueError("cannot fuse an empty score list")
    arr = np.asarray(scores, dtype=np.float64)
    if np.any((arr < 0) | (arr > 1)):
        raise ValueError("geometric-mean inputs must be in [0, 1]")
    if arr.size == 1:
        return float(arr[0])
    safe = np.maximum(arr, config.geomean_epsilon)
    return clamp01(float(np.exp(np.log(safe).mean())))


def rank_percentiles(
    scores_by_id: Mapping[str, float], candidate_ids: Sequence[str]
) -> dict[str, RankedScore]:
    """Rank one query over a shared candidate pool and map ranks to [0, 1].

    Equal raw scores receive the same equivalent rank. The top tied group
    maps to 1.0, the worst tied group maps to 0.0, and middle tied groups use
    the last 1-based rank occupied by the group. That keeps non-matching OCR
    candidates from looking strong just because they appear early in the
    candidate union.
    """
    if not candidate_ids:
        raise ValueError("cannot rank an empty candidate pool")

    positions = {keyframe_id: index for index, keyframe_id in enumerate(candidate_ids)}
    missing_ids = [keyframe_id for keyframe_id in candidate_ids if keyframe_id not in scores_by_id]
    if missing_ids:
        raise ValueError(f"missing query scores for candidate ids: {missing_ids[:5]}")

    def sort_key(keyframe_id: str) -> tuple[float, int]:
        score = float(scores_by_id[keyframe_id])
        sortable_score = score if math.isfinite(score) else -math.inf
        return (-sortable_score, positions[keyframe_id])

    ranked_ids = sorted(candidate_ids, key=sort_key)
    pool_size = len(ranked_ids)
    result: dict[str, RankedScore] = {}
    group_start = 0
    while group_start < pool_size:
        group_score = sort_key(ranked_ids[group_start])[0]
        group_end = group_start + 1
        while group_end < pool_size and sort_key(ranked_ids[group_end])[0] == group_score:
            group_end += 1

        if pool_size == 1:
            rank = 1
            percentile = 1.0
        elif group_start == 0 and group_end == pool_size:
            raw_score = float(scores_by_id[ranked_ids[group_start]])
            if math.isfinite(raw_score) and raw_score > 0:
                rank = 1
                percentile = 1.0
            else:
                rank = pool_size
                percentile = 0.0
        elif group_start == 0:
            rank = 1
            percentile = 1.0
        elif group_end == pool_size:
            rank = pool_size
            percentile = 0.0
        else:
            rank = group_end
            percentile = 1.0 - (rank - 1) / (pool_size - 1)
        for keyframe_id in ranked_ids[group_start:group_end]:
            result[keyframe_id] = RankedScore(
                raw_score=float(scores_by_id[keyframe_id]),
                rank=rank,
                rank_percentile=clamp01(percentile),
            )
        group_start = group_end
    return result


def bm25_reference(raw_scores: Sequence[float], config: KisConfig) -> float:
    positive = [float(score) for score in raw_scores if score > 0]
    if not positive:
        return 0.0
    if len(positive) < config.bm25_percentile_min_samples:
        return max(positive)
    return float(np.percentile(np.asarray(positive, dtype=np.float64), config.bm25_percentile))


def normalize_bm25(raw_score: float, reference: float) -> float:
    if reference <= 0 or raw_score <= 0:
        return 0.0
    return clamp01(raw_score / reference)


_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class NormalizedOcrText:
    text: str
    stripped_text: str
    tokens: list[str]
    stripped_tokens: list[str]


@dataclass(frozen=True)
class _CandidateUnit:
    start: int
    end: int
    forms: tuple[str, ...]
    stripped_forms: tuple[str, ...]


def normalize_ocr_text(text: str) -> NormalizedOcrText:
    normalized = _normalize_ocr_form(text, strip_accents=False)
    stripped = _normalize_ocr_form(text, strip_accents=True)
    return NormalizedOcrText(
        text=normalized,
        stripped_text=stripped,
        tokens=normalized.split() if normalized else [],
        stripped_tokens=stripped.split() if stripped else [],
    )


def ocr_coverage(query: str, candidate_ocr_text: str, config: KisConfig) -> float:
    query_norm = normalize_ocr_text(query)
    candidate_norm = normalize_ocr_text(candidate_ocr_text)
    if not query_norm.tokens or not candidate_norm.tokens:
        return 0.0

    query_tokens = list(zip(query_norm.tokens, query_norm.stripped_tokens, strict=False))
    candidate_units = _candidate_units(candidate_norm, config)
    pairs: list[tuple[float, int, int, int]] = []

    for query_pos, (query_token, query_stripped) in enumerate(query_tokens):
        for unit in candidate_units:
            similarity = _best_token_similarity(query_token, query_stripped, unit, config)
            if similarity > 0:
                pairs.append((similarity, query_pos, unit.start, unit.end))

    pairs.sort(key=lambda item: (-item[0], item[1], item[2], item[3]))
    matched_query_positions: set[int] = set()
    used_candidate_positions: set[int] = set()
    scores = [0.0] * len(query_tokens)

    for similarity, query_pos, start, end in pairs:
        if query_pos in matched_query_positions:
            continue
        span_positions = set(range(start, end))
        if span_positions & used_candidate_positions:
            continue
        matched_query_positions.add(query_pos)
        used_candidate_positions.update(span_positions)
        scores[query_pos] = similarity

    return clamp01(sum(scores) / len(scores))


def _normalize_ocr_form(text: str, strip_accents: bool) -> str:
    normalized = unicodedata.normalize("NFKC", text).lower()
    if strip_accents:
        normalized = _strip_vietnamese_accents(normalized)
    chars = []
    for char in normalized:
        category = unicodedata.category(char)
        if category.startswith("P") or category.startswith("S"):
            chars.append(" ")
        else:
            chars.append(char)
    return _WHITESPACE_RE.sub(" ", "".join(chars)).strip()


def _strip_vietnamese_accents(text: str) -> str:
    text = text.replace("đ", "d").replace("Đ", "D")
    decomposed = unicodedata.normalize("NFD", text)
    without_marks = "".join(char for char in decomposed if unicodedata.category(char) != "Mn")
    return unicodedata.normalize("NFC", without_marks)


def _candidate_units(candidate: NormalizedOcrText, config: KisConfig) -> list[_CandidateUnit]:
    units: list[_CandidateUnit] = []
    max_ngram = min(config.ocr_ngram_max_tokens, len(candidate.tokens))
    for start in range(len(candidate.tokens)):
        for width in range(1, max_ngram + 1):
            end = start + width
            if end > len(candidate.tokens):
                continue
            forms = _unit_forms(candidate.tokens[start:end])
            stripped_forms = _unit_forms(candidate.stripped_tokens[start:end])
            units.append(
                _CandidateUnit(start=start, end=end, forms=forms, stripped_forms=stripped_forms)
            )
    return units


def _unit_forms(tokens: Sequence[str]) -> tuple[str, ...]:
    spaced = " ".join(tokens)
    joined = "".join(tokens)
    forms = {spaced, joined}
    return tuple(form for form in forms if form)


def _best_token_similarity(
    query_token: str,
    query_stripped: str,
    unit: _CandidateUnit,
    config: KisConfig,
) -> float:
    if query_stripped.isdigit():
        return 1.0 if query_stripped in unit.stripped_forms else 0.0

    threshold = (
        config.ocr_short_token_threshold
        if len(query_stripped) <= config.ocr_short_token_max_length
        else config.ocr_long_token_threshold
    )
    similarity = max(
        [_ratio(query_token, form) for form in unit.forms]
        + [_ratio(query_stripped, form) for form in unit.stripped_forms],
        default=0.0,
    )
    return similarity if similarity >= threshold else 0.0


def _ratio(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    if _rapidfuzz_fuzz is not None:
        return float(_rapidfuzz_fuzz.ratio(left, right)) / 100.0
    return SequenceMatcher(a=left, b=right).ratio()
