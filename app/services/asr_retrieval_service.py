"""Deterministic lexical ASR transcript detection.

ASR is a boolean detector/filter in KIS Stage 1. It deliberately does not
use Elasticsearch BM25 scores, semantic similarity, or any numeric ASR
ranking signal.
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from elasticsearch import ApiError
from elasticsearch import ConnectionError as EsConnectionError
from elasticsearch import ConnectionTimeout as EsConnectionTimeout

from app.stores.es_client import AsrEsStore

AsrMatchMode = Literal["any", "all"]

_WHITESPACE_RE = re.compile(r"\s+")


class AsrRetrievalUnavailable(Exception):
    """Raised when ASR transcript detection cannot reach Elasticsearch."""


@dataclass(frozen=True)
class AsrRetrievalConfig:
    evidence_per_video: int = 3
    token_coverage_threshold: float = 0.8
    default_match_mode: AsrMatchMode = "any"

    def __post_init__(self) -> None:
        if self.evidence_per_video < 1:
            raise ValueError("asr_evidence_per_video must be positive")
        if not 0.0 < self.token_coverage_threshold <= 1.0:
            raise ValueError("asr_token_coverage_threshold must be in (0, 1]")
        if self.default_match_mode not in {"any", "all"}:
            raise ValueError("asr_match_mode must be 'any' or 'all'")

    @classmethod
    def from_mapping(cls, config: Mapping[str, object]) -> AsrRetrievalConfig:
        return cls(
            evidence_per_video=int(config.get("asr_evidence_per_video", 3)),
            token_coverage_threshold=float(config.get("asr_token_coverage_threshold", 0.8)),
            default_match_mode=_asr_match_mode(config.get("asr_match_mode", "any")),
        )


@dataclass(frozen=True)
class AsrEvidenceSegment:
    segment_id: str
    text: str
    start_time: float
    end_time: float


@dataclass(frozen=True)
class AsrQueryMatch:
    query: str
    detected: bool
    matches: list[AsrEvidenceSegment]


@dataclass(frozen=True)
class AsrVideoEvidence:
    video_id: str
    detected: bool
    asr_matches: list[AsrQueryMatch]
    segments: list[AsrEvidenceSegment]


@dataclass(frozen=True)
class AsrGateResult:
    queries: list[str]
    match_mode: AsrMatchMode
    scanned_segment_count: int
    videos: list[AsrVideoEvidence]
    asr_matches: list[AsrQueryMatch]

    @property
    def query(self) -> str:
        return self.queries[0] if len(self.queries) == 1 else " | ".join(self.queries)

    @property
    def detected(self) -> bool:
        if not self.queries:
            return False
        detected_by_query = {match.query: match.detected for match in self.asr_matches}
        if self.match_mode == "all":
            return all(detected_by_query.get(query, False) for query in self.queries)
        return any(detected_by_query.get(query, False) for query in self.queries)

    @property
    def asr_detected_any(self) -> bool:
        return any(match.detected for match in self.asr_matches)

    @property
    def matched_segment_count(self) -> int:
        seen: set[str] = set()
        for video in self.videos:
            for segment in video.segments:
                seen.add(segment.segment_id)
        return len(seen)

    @property
    def candidate_video_count(self) -> int:
        return len(self.videos)

    @property
    def video_ids(self) -> list[str]:
        return [video.video_id for video in self.videos]


@dataclass
class _VideoMatches:
    by_query: dict[str, list[AsrEvidenceSegment]]


class AsrRetrievalService:
    """Turns timestamped ASR segments into deterministic boolean detections."""

    def __init__(self, store: AsrEsStore, config: AsrRetrievalConfig) -> None:
        self._store = store
        self._config = config

    async def retrieve(
        self,
        query: str | Sequence[str],
        video_top_k: int,
        match_mode: AsrMatchMode | None = None,
    ) -> AsrGateResult:
        queries = _normalize_asr_queries(query)
        if not queries:
            raise ValueError("ASR query must not be blank")
        active_match_mode = match_mode or self._config.default_match_mode
        if active_match_mode not in {"any", "all"}:
            raise ValueError("ASR match mode must be 'any' or 'all'")

        matches_by_video: dict[str, _VideoMatches] = {}
        scanned_segment_count = 0
        try:
            async for row in self._store.scroll_all():
                scanned_segment_count += 1
                text = str(row.get("text", ""))
                matched_queries = _detect_queries(
                    queries, text, self._config.token_coverage_threshold
                )
                if not matched_queries:
                    continue
                video_id = str(row["video_id"])
                video_matches = matches_by_video.setdefault(
                    video_id, _VideoMatches(by_query=defaultdict(list))
                )
                segment = AsrEvidenceSegment(
                    segment_id=str(row["segment_id"]),
                    text=text,
                    start_time=float(row["start_time"]),
                    end_time=float(row["end_time"]),
                )
                for matched_query in matched_queries:
                    video_matches.by_query[matched_query].append(segment)
        except (EsConnectionError, EsConnectionTimeout, ApiError) as exc:
            raise AsrRetrievalUnavailable("ASR transcript detection is unavailable") from exc

        videos = _build_video_evidence(
            matches_by_video=matches_by_video,
            queries=queries,
            match_mode=active_match_mode,
            evidence_per_video=self._config.evidence_per_video,
            video_top_k=video_top_k,
        )
        aggregate_matches = _aggregate_query_matches(queries, videos)
        return AsrGateResult(
            queries=queries,
            match_mode=active_match_mode,
            scanned_segment_count=scanned_segment_count,
            videos=videos,
            asr_matches=aggregate_matches,
        )


def normalize_asr_text(text: str) -> str:
    """Normalizes ASR text while preserving Vietnamese diacritics."""
    chars: list[str] = []
    for char in text.strip().lower():
        category = unicodedata.category(char)
        if category[0] in {"P", "S"}:
            chars.append(" ")
        else:
            chars.append(char)
    return _WHITESPACE_RE.sub(" ", "".join(chars)).strip()


def asr_query_detected(
    query: str, transcript: str, token_coverage_threshold: float = 0.8
) -> bool:
    clean_query = normalize_asr_text(query)
    clean_transcript = normalize_asr_text(transcript)
    if not clean_query or not clean_transcript:
        return False

    query_tokens = clean_query.split()
    transcript_tokens = clean_transcript.split()
    if len(query_tokens) == 1:
        return query_tokens[0] in set(transcript_tokens)
    if _contains_contiguous_phrase(query_tokens, transcript_tokens):
        return True
    matched_token_count = sum(1 for token in query_tokens if token in set(transcript_tokens))
    if len(query_tokens) <= 3:
        return matched_token_count == len(query_tokens)
    return matched_token_count / len(query_tokens) >= token_coverage_threshold


def _asr_match_mode(value: object) -> AsrMatchMode:
    if value == "any":
        return "any"
    if value == "all":
        return "all"
    raise ValueError("asr_match_mode must be 'any' or 'all'")


def _normalize_asr_queries(query: str | Sequence[str]) -> list[str]:
    raw_queries = [query] if isinstance(query, str) else list(query)
    normalized: list[str] = []
    for raw_query in raw_queries:
        stripped = str(raw_query).strip()
        if stripped:
            normalized.append(stripped)
    return list(dict.fromkeys(normalized))


def _detect_queries(
    queries: Sequence[str], transcript: str, token_coverage_threshold: float
) -> tuple[str, ...]:
    return tuple(
        query
        for query in queries
        if asr_query_detected(query, transcript, token_coverage_threshold)
    )


def _contains_contiguous_phrase(
    query_tokens: Sequence[str], transcript_tokens: Sequence[str]
) -> bool:
    phrase_length = len(query_tokens)
    if phrase_length > len(transcript_tokens):
        return False
    return any(
        list(transcript_tokens[index : index + phrase_length]) == list(query_tokens)
        for index in range(len(transcript_tokens) - phrase_length + 1)
    )


def _build_video_evidence(
    *,
    matches_by_video: dict[str, _VideoMatches],
    queries: Sequence[str],
    match_mode: AsrMatchMode,
    evidence_per_video: int,
    video_top_k: int,
) -> list[AsrVideoEvidence]:
    videos: list[AsrVideoEvidence] = []
    for video_id in sorted(matches_by_video):
        video_matches = matches_by_video[video_id]
        detected_by_query = {
            query: bool(video_matches.by_query.get(query))
            for query in queries
        }
        detected = (
            all(detected_by_query.values())
            if match_mode == "all"
            else any(detected_by_query.values())
        )
        if not detected:
            continue
        all_segments = _unique_segments(
            segment
            for query in queries
            for segment in video_matches.by_query.get(query, [])
        )
        query_matches = [
            AsrQueryMatch(
                query=query,
                detected=detected_by_query[query],
                matches=video_matches.by_query.get(query, [])[:evidence_per_video],
            )
            for query in queries
        ]
        videos.append(
            AsrVideoEvidence(
                video_id=video_id,
                detected=True,
                asr_matches=query_matches,
                segments=all_segments[:evidence_per_video],
            )
        )
        if len(videos) >= video_top_k:
            break
    return videos


def _aggregate_query_matches(
    queries: Sequence[str], videos: Sequence[AsrVideoEvidence]
) -> list[AsrQueryMatch]:
    matches_by_query: dict[str, list[AsrEvidenceSegment]] = {query: [] for query in queries}
    for video in videos:
        for match in video.asr_matches:
            matches_by_query[match.query].extend(match.matches)
    return [
        AsrQueryMatch(
            query=query,
            detected=bool(matches_by_query[query]),
            matches=_unique_segments(matches_by_query[query]),
        )
        for query in queries
    ]


def _unique_segments(segments: Sequence[AsrEvidenceSegment]) -> list[AsrEvidenceSegment]:
    unique: list[AsrEvidenceSegment] = []
    seen_ids: set[str] = set()
    for segment in sorted(segments, key=lambda item: (item.start_time, item.segment_id)):
        if segment.segment_id in seen_ids:
            continue
        seen_ids.add(segment.segment_id)
        unique.append(segment)
    return unique
