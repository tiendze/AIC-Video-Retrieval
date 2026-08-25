"""TrakeService -- `POST /api/trake`, candidate-set chaining pipeline.

Replaces the corpus-wide DANTE dynamic program. The stages, and what each one
exists to prevent:

B1  Per-event KIS retrieval. Every event runs its own
    `SearchService.retrieve_kis_candidates()` with its own Text / OCR / Title /
    ASR boxes -- the same Stage 1 the KIS tab uses, so TRAKE inherits whatever
    makes KIS good rather than reimplementing it.

    Candidate videos are the UNION over events, not the intersection. Requiring
    every event to place a frame in the global top-K multiplies the per-event
    recalls together: five events at 0.85 leave 0.44, and one vaguely worded
    event throws away a video whose other four events matched perfectly.

B2  Local backfill. KIS decides WHICH videos are candidates; it does not decide
    which frames answer each event. Inside a candidate video, every event is
    rescored against that video's whole keyframe slice from the in-RAM cache, so
    an event too weak to clear the global top-K still has candidates in the
    right video. Buckets are therefore never empty for a candidate video.

B3  Per-event z-normalization. Cosines from two different event texts are not
    comparable (`trake_chain.zscore_rows`). KIS evidence enters as a separate
    rank-based bonus rather than by mixing BM25/title/fused scores into the same
    number as the cosine.

B4  Temporal NMS per bucket, B5 chaining on the layered DAG, B7 sequence NMS +
    weak-link rerank, B8 video rerank, B9 slot allocation -- all in
    `app/services/trake_chain.py`.

Each event is translated VI->EN before encoding when a `QueryTranslator` is
injected -- same reasoning as `SearchService._search_semantic`: the production
text tower is English-caption oriented. Confirmed 2026-08-14 by a real
`eval/run.py` run: `video_level_recall_at_k` was 0.0000 across every TRAKE query
before translation was wired in.

An event may carry no text at all (an OCR-only or ASR-only row). That event
contributes no semantic axis, so its bucket comes purely from the KIS hits for
that video; if KIS returned none there, the video cannot be chained and is
dropped with a log line rather than silently.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from app.encoders.base import MultimodalEncoder
from app.services.query_translator import QueryTranslator
from app.services.search_service import KisCandidatePool, SearchService, SearchServiceError
from app.services.trake_chain import (
    CandidateFrame,
    ChainCandidate,
    TrakeChainError,
    VideoScore,
    allocate_submission_slots,
    chain_sequences,
    kis_evidence_score,
    score_videos,
    sequence_nms,
    temporal_nms,
    zscore_rows,
)
from app.services.trake_embedding_cache import TrakeEmbeddingCache, TrakeVideoSlice

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_CONFIG_PATH = _REPO_ROOT / "configs" / "search" / "trake.yaml"


def load_trake_config(path: Path = _CONFIG_PATH) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


_config = load_trake_config()
DEFAULT_SCORE_THRESHOLD: float = _config["default_score_threshold"]


class TrakeServiceError(Exception):
    """Raised for a caller-fixable TRAKE request problem. Carries
    `error_code`/`status_code` so the router maps each cause to its own HTTP
    response without parsing message text (CLAUDE.md §7)."""

    def __init__(self, message: str, error_code: str, status_code: int) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.status_code = status_code


@dataclass(frozen=True)
class TrakeEventQuery:
    """One event row of the operator's form.

    All four channels are per-event by design: OCR pins a moment (a scoreboard
    that only appears at the final event), and ASR/Title, although they gate at
    the video level, are allowed to differ per event so a commentary line tied
    to one specific moment can be used without constraining the others.
    """

    text_queries: tuple[str, ...] = ()
    text_top_k: int | None = None
    ocr_query: str | None = None
    ocr_top_k: int | None = None
    title_query: str | None = None
    title_top_k: int | None = None
    asr_queries: tuple[str, ...] = ()
    asr_match_mode: str = "any"
    asr_video_top_k: int | None = None

    @property
    def has_text(self) -> bool:
        return bool(self.text_queries)

    @property
    def is_active(self) -> bool:
        return bool(self.text_queries or self.ocr_query or self.title_query or self.asr_queries)


@dataclass(frozen=True)
class TrakeEventDiagnostics:
    event_index: int
    kis_candidate_count: int
    kis_video_count: int
    has_text: bool


@dataclass(frozen=True)
class TrakeSearchResult:
    """What the router renders. `candidates` is already in submission order --
    round-robin across videos, not a flat sort by sequence score.

    `sequences_by_video` is a SEPARATE view: for every video that produced at
    least one chain, its own top sequences (before the round-robin submission
    cap). The UI's per-video "Details" button expands into this so an operator
    can inspect a video's other sequences without them consuming submission
    slots."""

    candidates: tuple[ChainCandidate, ...]
    video_scores: dict[str, VideoScore]
    events: tuple[TrakeEventDiagnostics, ...] = ()
    candidate_video_count: int = 0
    chained_video_count: int = 0
    dropped_video_count: int = 0
    sequences_by_video: dict[str, tuple[ChainCandidate, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class _CandidateScores:
    """Z-scored semantic similarity over ONLY the candidate frames.

    `matrix` is `(N events, M candidate frames)`; `col_of` maps a corpus row
    index (position in the TRAKE cache matrix) to its column in `matrix`. Only
    frames belonging to a candidate video have a column -- a lookup for any
    other corpus index is a bug, not a miss."""

    matrix: np.ndarray
    col_of: dict[int, int]


@dataclass
class _VideoEvidence:
    """Per-candidate-video bookkeeping from stage B1."""

    slice: TrakeVideoSlice
    hits_by_event: dict[int, list[tuple[int, int]]] = field(default_factory=dict)
    best_rank_by_event: dict[int, int] = field(default_factory=dict)

    @property
    def coverage(self) -> int:
        return len(self.hits_by_event)


class TrakeService:
    def __init__(
        self,
        encoder: MultimodalEncoder,
        trake_cache: TrakeEmbeddingCache,
        search_service: SearchService,
        query_translator: QueryTranslator | None = None,
        config: dict | None = None,
    ) -> None:
        self._encoder = encoder
        self._trake_cache = trake_cache
        self._search_service = search_service
        self._query_translator = query_translator
        self._config = config or load_trake_config()
        self._corpus_index_by_keyframe: dict[str, int] | None = None
        self._corpus_index_load_count = -1

    async def search(
        self,
        events: Sequence[TrakeEventQuery],
        top_k: int,
        threshold: float = DEFAULT_SCORE_THRESHOLD,
    ) -> TrakeSearchResult:
        total_start = time.perf_counter()
        self._validate_events(events)
        self._trake_cache.ensure_ready()

        # B0 -- encode events (the semantic axis). Z-scoring is deferred until
        # the candidate videos are known, so the normalization baseline is the
        # candidate pool, never the whole corpus.
        encode_start = time.perf_counter()
        event_vectors = await self._encode_events(events)
        encode_ms = (time.perf_counter() - encode_start) * 1000

        # B1 -- per-event KIS candidates. Every event's own pool, run
        # concurrently; each already has its own ASR/Title video gate applied.
        kis_start = time.perf_counter()
        pools = await self._retrieve_event_pools(events)
        kis_ms = (time.perf_counter() - kis_start) * 1000

        evidence = self._collect_video_evidence(pools)
        candidates_by_video = self._select_candidate_videos(evidence)

        # B3 -- semantic scores z-scored over ONLY the candidate frames. KIS has
        # already narrowed the corpus to `candidates_by_video`, so scoring the
        # full ~221K-frame matrix here would be wasted work.
        score_start = time.perf_counter()
        candidate_scores = await asyncio.to_thread(
            self._normalized_scores_for_candidates, event_vectors, candidates_by_video
        )
        score_ms = (time.perf_counter() - score_start) * 1000

        # B2/B4/B5/B7 -- buckets, chaining, sequence NMS.
        chain_start = time.perf_counter()
        chains, dropped = await asyncio.to_thread(
            self._chain_candidate_videos, candidates_by_video, candidate_scores, events
        )
        chain_ms = (time.perf_counter() - chain_start) * 1000

        chains = [chain for chain in chains if chain.score >= threshold]
        chains = sequence_nms(chains, frame_tolerance=int(self._config["sequence_nms_frames"]))
        sequences_by_video = self._top_sequences_by_video(chains)

        # B8/B9 -- video rerank, then round-robin slot allocation.
        video_scores = score_videos(
            chains,
            keyframe_counts={
                video_id: item.slice.length for video_id, item in candidates_by_video.items()
            },
            rank_evidence=self._rank_evidence(candidates_by_video),
            top_chains=int(self._config["video_top_chains"]),
            logsumexp_weight=float(self._config["video_logsumexp_weight"]),
            coverage_weight=float(self._config["video_coverage_weight"]),
            rank_evidence_weight=float(self._config["video_rank_evidence_weight"]),
            length_penalty=float(self._config["video_length_penalty"]),
        )
        allocated = allocate_submission_slots(
            chains,
            video_scores,
            max_sequences_per_video=int(self._config["max_sequences_per_video"]),
            limit=top_k,
        )

        total_ms = (time.perf_counter() - total_start) * 1000
        logger.info(
            "[TRAKE TIMING] encode_ms=%.1f kis_ms=%.1f score_ms=%.1f chain_ms=%.1f "
            "total_ms=%.1f events=%d candidate_videos=%d chained_videos=%d dropped_videos=%d "
            "sequences=%d returned=%d",
            encode_ms,
            kis_ms,
            score_ms,
            chain_ms,
            total_ms,
            len(events),
            len(candidates_by_video),
            len({chain.video_id for chain in chains}),
            dropped,
            len(chains),
            len(allocated),
        )
        return TrakeSearchResult(
            candidates=tuple(allocated),
            video_scores=video_scores,
            events=tuple(
                TrakeEventDiagnostics(
                    event_index=index,
                    kis_candidate_count=len(pool.results),
                    kis_video_count=len({hit.video_id for hit in pool.results}),
                    has_text=event.has_text,
                )
                for index, (event, pool) in enumerate(zip(events, pools))
            ),
            candidate_video_count=len(candidates_by_video),
            chained_video_count=len({chain.video_id for chain in chains}),
            dropped_video_count=dropped,
            sequences_by_video=sequences_by_video,
        )

    def _top_sequences_by_video(
        self, chains: Sequence[ChainCandidate]
    ) -> dict[str, tuple[ChainCandidate, ...]]:
        """B9-adjacent: every video's own top sequences, before the round-robin
        submission cap. Feeds the per-video "Details" expander in the UI so an
        operator can see a video's other sequences without them taking a
        submission slot."""
        details_top_k = int(self._config["details_top_k"])
        grouped: dict[str, list[ChainCandidate]] = {}
        for chain in chains:
            grouped.setdefault(chain.video_id, []).append(chain)
        return {
            video_id: tuple(
                sorted(sequences, key=lambda item: (-item.score, item.frame_indices))[:details_top_k]
            )
            for video_id, sequences in grouped.items()
        }

    # ---------------------------------------------------------------- B0/B3

    def _validate_events(self, events: Sequence[TrakeEventQuery]) -> None:
        if not events:
            raise TrakeServiceError(
                "TRAKE requires at least one event",
                error_code="no_events",
                status_code=400,
            )
        for index, event in enumerate(events):
            if not event.is_active:
                raise TrakeServiceError(
                    f"Event {index + 1} has no Text, OCR, Title or ASR query",
                    error_code="empty_event",
                    status_code=400,
                )

    async def _encode_events(self, events: Sequence[TrakeEventQuery]) -> np.ndarray:
        """One L2-normalized vector per event.

        Multiple text boxes on one event are prompt-ensembled: encode each,
        average the already-normalized vectors, renormalize. Averaging
        normalized vectors is the standard CLIP-family ensemble and is cheap
        relative to any other accuracy lever here.

        An event with no text gets a zero row. Callers must treat a zero row as
        "no semantic axis for this event" rather than as a real query vector --
        `_build_buckets` does, falling back to that event's KIS hits.
        """
        dim = self._encoder.embedding_dim
        vectors = np.zeros((len(events), dim), dtype=np.float32)
        for index, event in enumerate(events):
            if not event.has_text:
                continue
            texts = list(event.text_queries)
            if self._query_translator is not None:
                texts = await asyncio.to_thread(self._translate, texts)
            encoded = await asyncio.to_thread(self._encoder.encode_text, texts)
            pooled = np.asarray(encoded, dtype=np.float32).mean(axis=0)
            norm = float(np.linalg.norm(pooled))
            if norm > 0:
                pooled = pooled / norm
            vectors[index] = pooled
        return vectors

    def _translate(self, texts: Sequence[str]) -> list[str]:
        assert self._query_translator is not None
        return [self._query_translator.translate(text) for text in texts]

    def _normalized_scores_for_candidates(
        self, event_vectors: np.ndarray, videos: dict[str, _VideoEvidence]
    ) -> _CandidateScores:
        """`(N, M)` z-scored similarity over ONLY the candidate frames.

        Both sides are already L2-normalized by the encoder (CLAUDE.md §3), so
        the dot product IS the cosine -- renormalizing here would paper over an
        encoder that broke that invariant instead of surfacing it.

        Restricting the matmul (and the z-score baseline) to the candidate
        videos is the point: KIS already decided which videos matter, so the
        full-corpus score matrix that the DANTE-era path computed on every
        request is pure waste here. The trade-off is that `coverage_threshold`
        (configs/search/trake.yaml) now measures a frame against the candidate
        pool's mean, not the whole corpus's.
        """
        corpus_indices = sorted(
            {
                index
                for item in videos.values()
                for index in range(item.slice.start, item.slice.stop)
            }
        )
        if not corpus_indices:
            return _CandidateScores(
                matrix=np.zeros((event_vectors.shape[0], 0), dtype=np.float64), col_of={}
            )
        sub = self._trake_cache.embeddings[corpus_indices]
        raw = event_vectors @ sub.T
        if self._config.get("score_normalization", "zscore") != "zscore":
            matrix = np.asarray(raw, dtype=np.float64)
        else:
            matrix = zscore_rows(raw)
        col_of = {corpus_index: column for column, corpus_index in enumerate(corpus_indices)}
        return _CandidateScores(matrix=matrix, col_of=col_of)

    # ------------------------------------------------------------------- B1

    async def _retrieve_event_pools(
        self, events: Sequence[TrakeEventQuery]
    ) -> list[KisCandidatePool]:
        pools = await asyncio.gather(
            *[self._retrieve_one_pool(index, event) for index, event in enumerate(events)]
        )
        return list(pools)

    async def _retrieve_one_pool(self, index: int, event: TrakeEventQuery) -> KisCandidatePool:
        try:
            return await self._search_service.retrieve_kis_candidates(
                text_query=None,
                text_queries=list(event.text_queries),
                text_top_k=int(event.text_top_k or self._config["event_text_top_k"]),
                ocr_query=event.ocr_query,
                ocr_top_k=int(event.ocr_top_k or self._config["event_ocr_top_k"]),
                title_query=event.title_query,
                title_top_k=int(event.title_top_k or self._config["event_title_top_k"]),
                asr_query=None,
                asr_queries=list(event.asr_queries),
                asr_match_mode=event.asr_match_mode,  # type: ignore[arg-type]
                asr_video_top_k=int(event.asr_video_top_k or self._config["event_asr_video_top_k"]),
            )
        except SearchServiceError as exc:
            # Deliberately fatal, not degraded. The operator asked for this
            # event's OCR/ASR filter; quietly running without it would change
            # the result set while looking like it worked (CLAUDE.md §7:
            # missing data must fail loudly, never return an empty-looking
            # success).
            raise TrakeServiceError(
                f"Event {index + 1}: {exc}",
                error_code=exc.error_code,
                status_code=exc.status_code,
            ) from exc

    def _collect_video_evidence(
        self, pools: Sequence[KisCandidatePool]
    ) -> dict[str, _VideoEvidence]:
        slices_by_video = {
            video_slice.video_id: video_slice for video_slice in self._trake_cache.video_slices
        }
        index_by_keyframe = self._keyframe_index()
        evidence: dict[str, _VideoEvidence] = {}
        for event_index, pool in enumerate(pools):
            for rank, hit in enumerate(pool.results):
                video_slice = slices_by_video.get(hit.video_id)
                corpus_index = index_by_keyframe.get(hit.keyframe_id)
                if video_slice is None or corpus_index is None:
                    # A KIS hit outside the TRAKE cache: the video was not
                    # `ingest_status == complete`, or its global_index range is
                    # not contiguous (trake_embedding_cache skips those). Not
                    # chainable -- log at debug volume, it is expected.
                    logger.debug(
                        "trake: dropping KIS hit keyframe_id=%s (video_id=%s) -- "
                        "not in the TRAKE embedding cache",
                        hit.keyframe_id,
                        hit.video_id,
                    )
                    continue
                item = evidence.setdefault(hit.video_id, _VideoEvidence(slice=video_slice))
                item.hits_by_event.setdefault(event_index, []).append((corpus_index, rank))
                previous = item.best_rank_by_event.get(event_index)
                if previous is None or rank < previous:
                    item.best_rank_by_event[event_index] = rank
        return evidence

    def _select_candidate_videos(
        self, evidence: dict[str, _VideoEvidence]
    ) -> dict[str, _VideoEvidence]:
        min_coverage = int(self._config["min_event_coverage"])
        max_videos = int(self._config["max_candidate_videos"])
        eligible = [
            (video_id, item) for video_id, item in evidence.items() if item.coverage >= min_coverage
        ]
        # Coverage first, then how highly KIS ranked this video's best frames.
        # Both are recall-oriented: this stage only decides who gets chained.
        eligible.sort(
            key=lambda pair: (
                -pair[1].coverage,
                -self._video_rank_evidence(pair[1]),
                pair[0],
            )
        )
        return dict(eligible[:max_videos])

    def _video_rank_evidence(self, item: _VideoEvidence) -> float:
        rrf_k = int(self._config["kis_evidence_rrf_k"])
        return sum(kis_evidence_score(rank, rrf_k) for rank in item.best_rank_by_event.values())

    def _rank_evidence(self, videos: dict[str, _VideoEvidence]) -> dict[str, float]:
        return {video_id: self._video_rank_evidence(item) for video_id, item in videos.items()}

    def _keyframe_index(self) -> dict[str, int]:
        """`keyframe_id -> row in the cache matrix`, built once per cache load.

        Keyed on `load_count` so a reloaded cache invalidates it instead of
        silently mapping ids onto the previous matrix's rows.
        """
        if (
            self._corpus_index_by_keyframe is None
            or self._corpus_index_load_count != self._trake_cache.load_count
        ):
            self._corpus_index_by_keyframe = {
                keyframe.keyframe_id: index
                for index, keyframe in enumerate(self._trake_cache.metadata)
            }
            self._corpus_index_load_count = self._trake_cache.load_count
        return self._corpus_index_by_keyframe

    # ---------------------------------------------------------------- B2/B5

    def _chain_candidate_videos(
        self,
        videos: dict[str, _VideoEvidence],
        candidate_scores: _CandidateScores,
        events: Sequence[TrakeEventQuery],
    ) -> tuple[list[ChainCandidate], int]:
        chains: list[ChainCandidate] = []
        dropped = 0
        for video_id, item in videos.items():
            buckets = self._build_buckets(item, candidate_scores, events)
            if buckets is None:
                dropped += 1
                continue
            video_chains = self._chain_with_relax(video_id, buckets)
            if not video_chains:
                dropped += 1
                continue
            chains.extend(video_chains)
        return chains, dropped

    def _build_buckets(
        self,
        item: _VideoEvidence,
        candidate_scores: _CandidateScores,
        events: Sequence[TrakeEventQuery],
    ) -> list[list[CandidateFrame]] | None:
        """B2 + B4: candidates for each event inside one video.

        Union of (a) the frames KIS returned for this event in this video and
        (b) a local top-k backfill against this video's whole slice. (b) is
        what stops a weak event from emptying the bucket; (a) is what carries
        the OCR / Title / ASR evidence in.
        """
        bucket_top_k = int(self._config["bucket_top_k"])
        nms_seconds = float(self._config["bucket_nms_seconds"])
        rrf_k = int(self._config["kis_evidence_rrf_k"])
        evidence_weight = float(self._config["kis_evidence_weight"])
        video_slice = item.slice

        buckets: list[list[CandidateFrame]] = []
        for event_index, event in enumerate(events):
            ranks: dict[int, int] = {}
            for corpus_index, rank in item.hits_by_event.get(event_index, []):
                previous = ranks.get(corpus_index)
                if previous is None or rank < previous:
                    ranks[corpus_index] = rank

            corpus_indices: set[int] = set(ranks)
            if event.has_text:
                corpus_indices |= self._local_top_indices(
                    candidate_scores, event_index, video_slice, bucket_top_k
                )
            if not corpus_indices:
                logger.info(
                    "trake: dropping video_id=%s -- event %d has no candidate frame "
                    "(no KIS hit here and no text query to backfill with)",
                    video_slice.video_id,
                    event_index + 1,
                )
                return None

            frames = [
                self._to_candidate_frame(
                    corpus_index,
                    semantic_z=(
                        float(candidate_scores.matrix[event_index, candidate_scores.col_of[corpus_index]])
                        if event.has_text
                        else 0.0
                    ),
                    rank=ranks.get(corpus_index),
                    rrf_k=rrf_k,
                    evidence_weight=evidence_weight,
                )
                for corpus_index in corpus_indices
            ]
            kept = temporal_nms(frames, min_seconds=nms_seconds, keep=bucket_top_k)
            # The chain DP walks each bucket in time order.
            kept.sort(key=lambda frame: frame.timestamp)
            buckets.append(kept)
        return buckets

    def _local_top_indices(
        self,
        scores: _CandidateScores,
        event_index: int,
        video_slice: TrakeVideoSlice,
        top_k: int,
    ) -> set[int]:
        # Every corpus row of a candidate video has a column in `scores` (the
        # matrix was built from exactly these slices), so `col_of` never misses.
        columns = [scores.col_of[index] for index in range(video_slice.start, video_slice.stop)]
        if not columns:
            return set()
        window = scores.matrix[event_index, columns]
        take = min(top_k, window.size)
        local = np.argpartition(window, -take)[-take:]
        # `columns` is in ascending corpus-index order, so local column i maps
        # back to corpus index `video_slice.start + i`.
        return {video_slice.start + int(index) for index in local}

    def _to_candidate_frame(
        self,
        corpus_index: int,
        *,
        semantic_z: float,
        rank: int | None,
        rrf_k: int,
        evidence_weight: float,
    ) -> CandidateFrame:
        cached = self._trake_cache.metadata[corpus_index]
        evidence = kis_evidence_score(rank, rrf_k)
        return CandidateFrame(
            corpus_index=corpus_index,
            keyframe_id=cached.keyframe_id,
            video_id=cached.video_id,
            frame_idx=cached.frame_idx,
            timestamp=cached.timestamp,
            score=semantic_z + evidence_weight * evidence,
            semantic_z=semantic_z,
            kis_evidence=evidence,
            kis_rank=rank,
        )

    def _chain_with_relax(
        self, video_id: str, buckets: list[list[CandidateFrame]]
    ) -> list[ChainCandidate]:
        min_gap = float(self._config["min_gap_seconds"])
        max_gap = float(self._config["max_gap_seconds"])
        relax = float(self._config["max_gap_relax_factor"])
        kwargs = dict(
            min_gap_seconds=min_gap,
            beam_width=int(self._config["beam_width"]),
            weak_link_beta=float(self._config["weak_link_beta"]),
            coverage_threshold=float(self._config["coverage_threshold"]),
            coverage_weight=float(self._config["coverage_weight"]),
        )
        try:
            chains = chain_sequences(buckets, max_gap_seconds=max_gap, **kwargs)
            if chains:
                return chains
            # One relax step. A process filmed with a long cutaway in the middle
            # is a real case; a video with no ordered path even then is dropped
            # with this log line, never silently.
            chains = chain_sequences(buckets, max_gap_seconds=max_gap * relax, **kwargs)
            if not chains:
                logger.info(
                    "trake: dropping video_id=%s -- no ordered sequence within "
                    "min_gap=%.2fs / max_gap=%.2fs (relaxed to %.2fs)",
                    video_id,
                    min_gap,
                    max_gap,
                    max_gap * relax,
                )
            return chains
        except TrakeChainError as exc:
            logger.info("trake: skipping video_id=%s: %s", video_id, exc)
            return []
