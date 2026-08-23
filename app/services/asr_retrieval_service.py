"""Deterministic BM25 transcript aggregation for the optional ASR video gate."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from elasticsearch import ApiError
from elasticsearch import ConnectionError as EsConnectionError

from app.stores.es_client import AsrEsStore, AsrSegmentSearchResult


class AsrRetrievalUnavailable(Exception):
    """Raised when the active ASR transcript search cannot reach Elasticsearch."""


@dataclass(frozen=True)
class AsrRetrievalConfig:
    segment_retrieval_k: int = 200
    evidence_per_video: int = 3

    def __post_init__(self) -> None:
        if self.segment_retrieval_k < 1:
            raise ValueError("asr_segment_retrieval_k must be positive")
        if self.evidence_per_video < 1:
            raise ValueError("asr_evidence_per_video must be positive")

    @classmethod
    def from_mapping(cls, config: Mapping[str, object]) -> AsrRetrievalConfig:
        return cls(
            segment_retrieval_k=int(config.get("asr_segment_retrieval_k", 200)),
            evidence_per_video=int(config.get("asr_evidence_per_video", 3)),
        )


@dataclass(frozen=True)
class AsrEvidenceSegment:
    segment_id: str
    text: str
    start_time: float
    end_time: float
    raw_bm25: float


@dataclass(frozen=True)
class AsrVideoEvidence:
    video_id: str
    video_rank: int
    best_bm25_score: float
    segments: list[AsrEvidenceSegment]


@dataclass(frozen=True)
class AsrGateResult:
    query: str
    segment_hit_count: int
    videos: list[AsrVideoEvidence]

    @property
    def candidate_video_count(self) -> int:
        return len(self.videos)

    @property
    def video_ids(self) -> list[str]:
        return [video.video_id for video in self.videos]


@dataclass
class _VideoGroup:
    best_bm25_score: float
    first_segment_rank: int
    segments: list[tuple[AsrEvidenceSegment, int]]


class AsrRetrievalService:
    """Turns BM25-ranked ASR segments into a deterministic video-only gate."""

    def __init__(self, store: AsrEsStore, config: AsrRetrievalConfig) -> None:
        self._store = store
        self._config = config

    async def retrieve(self, query: str, video_top_k: int) -> AsrGateResult:
        clean_query = query.strip()
        if not clean_query:
            raise ValueError("ASR query must not be blank")
        try:
            response = await self._store.search(clean_query, self._config.segment_retrieval_k)
        except (EsConnectionError, ApiError) as exc:
            raise AsrRetrievalUnavailable("ASR transcript search is unavailable") from exc
        return _aggregate_gate(clean_query, response, video_top_k, self._config.evidence_per_video)


def _aggregate_gate(
    query: str,
    response: AsrSegmentSearchResult,
    video_top_k: int,
    evidence_per_video: int,
) -> AsrGateResult:
    groups: dict[str, _VideoGroup] = {}
    seen_segment_ids: set[str] = set()

    for original_rank, hit in enumerate(response.hits, start=1):
        if hit.segment_id in seen_segment_ids:
            continue
        seen_segment_ids.add(hit.segment_id)
        evidence = AsrEvidenceSegment(
            segment_id=hit.segment_id,
            text=hit.text,
            start_time=hit.start_time,
            end_time=hit.end_time,
            raw_bm25=hit.score,
        )
        group = groups.get(hit.video_id)
        if group is None:
            groups[hit.video_id] = _VideoGroup(
                best_bm25_score=hit.score,
                first_segment_rank=original_rank,
                segments=[(evidence, original_rank)],
            )
            continue
        group.best_bm25_score = max(group.best_bm25_score, hit.score)
        group.segments.append((evidence, original_rank))

    ranked_videos = sorted(
        groups.items(),
        key=lambda item: (-item[1].best_bm25_score, item[1].first_segment_rank, item[0]),
    )[:video_top_k]
    videos = [
        AsrVideoEvidence(
            video_id=video_id,
            video_rank=video_rank,
            best_bm25_score=group.best_bm25_score,
            segments=[
                evidence
                for evidence, _ in sorted(
                    group.segments,
                    key=lambda item: (-item[0].raw_bm25, item[1]),
                )[:evidence_per_video]
            ],
        )
        for video_rank, (video_id, group) in enumerate(ranked_videos, start=1)
    ]
    return AsrGateResult(query=query, segment_hit_count=response.total_hits, videos=videos)
