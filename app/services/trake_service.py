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
from app.services.kis_local_retrieval import (
    KISLocalEmbeddingStore,
    VideoLocalFiles,
    maxsim_query_scores,
)
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
    local_phrases: tuple[str, ...] = ()
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
    # True when this event's (translated) text exceeded the encoder's
    # max_text_tokens cap and was silently truncated -- see
    # TrakeService._encode_events. A warning, not a fix (plan §T3).
    text_truncated: bool = False


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
    # B2b diagnostics (plan §T3): how many candidate videos actually got the
    # phrase-to-patch rerank, and how many of their frames had no dense
    # embedding to score against (missing data or dense/keyframe mismatch).
    local_rerank_video_count: int = 0
    local_missing_frame_count: int = 0


@dataclass(frozen=True)
class _CandidateScores:
    """Z-scored semantic similarity over ONLY the candidate frames.

    `matrix` is `(N events, M candidate frames)`; `col_of` maps a corpus row
    index (position in the TRAKE cache matrix) to its column in `matrix`. Only
    frames belonging to a candidate video have a column -- a lookup for any
    other corpus index is a bug, not a miss."""

    matrix: np.ndarray
    col_of: dict[int, int]


@dataclass(frozen=True)
class _LocalFrameScore:
    """One (event, corpus_index) MaxSim result from B2b. `raw` is the mean
    MaxSim value (unbounded roughly [-1, 1], same definition as KIS's
    `LocalCandidateScore.local_score`); `z` is `raw` z-scored across the
    video's own frames for this event (mean 0 per video -- see
    TrakeService._score_local_rerank)."""

    raw: float
    z: float


@dataclass
class _LocalRerankStats:
    """Accumulated across every candidate video's B2b pass, for the
    `[TRAKE TIMING]` log line and `TrakeSearchResult`'s diagnostics fields."""

    videos_reranked: int = 0
    missing_frame_count: int = 0
    io_ms: float = 0.0
    scoring_ms: float = 0.0


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
        local_embedding_store: KISLocalEmbeddingStore | None = None,
    ) -> None:
        self._encoder = encoder
        self._trake_cache = trake_cache
        self._search_service = search_service
        self._query_translator = query_translator
        self._config = config or load_trake_config()
        self._corpus_index_by_keyframe: dict[str, int] | None = None
        self._corpus_index_load_count = -1
        # Shared with SearchService's KIS Stage 1 phrase rerank (app/main.py
        # lifespan wiring) so both read the same LRU-mmap cache instead of
        # each parsing manifest.csv and opening their own memory maps.
        self._local_embedding_store = local_embedding_store or search_service.local_embedding_store
        self._local_score_batch_size = search_service.kis_config.local_score_batch_size

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
        event_vectors, truncated_events = await self._encode_events(events)
        phrase_vectors_by_event = await self._encode_local_phrases(events)
        encode_ms = (time.perf_counter() - encode_start) * 1000

        # B1 -- per-event KIS candidates. Every event's own pool, run
        # concurrently; each already has its own ASR/Title video gate applied.
        kis_start = time.perf_counter()
        pools = await self._retrieve_event_pools(events)
        kis_ms = (time.perf_counter() - kis_start) * 1000

        evidence = self._collect_video_evidence(pools)
        candidates_by_video = self._select_candidate_videos(evidence)

        # B2b -- which of those candidate videos actually get the phrase-to-
        # patch rerank. Capped independently of max_candidate_videos: reading
        # a whole video's dense array is far more expensive than a KIS hit,
        # so this is a SUBSET, ranked by the same coverage/evidence order
        # (configs/search/trake.yaml: local_rerank.max_videos).
        local_rerank_cfg = self._config.get("local_rerank", {})
        local_rerank_video_ids: frozenset[str] = frozenset()
        if phrase_vectors_by_event and bool(local_rerank_cfg.get("enabled", True)):
            local_max_videos = int(local_rerank_cfg.get("max_videos", 50))
            local_rerank_video_ids = frozenset(list(candidates_by_video.keys())[:local_max_videos])

        # B3 -- semantic scores z-scored over ONLY the candidate frames. KIS has
        # already narrowed the corpus to `candidates_by_video`, so scoring the
        # full ~221K-frame matrix here would be wasted work.
        score_start = time.perf_counter()
        candidate_scores = await asyncio.to_thread(
            self._normalized_scores_for_candidates, event_vectors, candidates_by_video
        )
        score_ms = (time.perf_counter() - score_start) * 1000

        # B2/B2b/B4/B5/B7 -- buckets (with local rerank folded in), chaining,
        # sequence NMS.
        chain_start = time.perf_counter()
        chains, dropped, local_stats = await asyncio.to_thread(
            self._chain_candidate_videos,
            candidates_by_video,
            candidate_scores,
            events,
            phrase_vectors_by_event,
            local_rerank_video_ids,
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
            rank_evidence=self._rank_evidence(candidates_by_video, len(events)),
            top_chains=int(self._config["video_top_chains"]),
            logsumexp_weight=float(self._config["video_logsumexp_weight"]),
            coverage_weight=float(self._config["video_coverage_weight"]),
            rank_evidence_weight=float(self._config["video_rank_evidence_weight"]),
            length_penalty=float(self._config["video_length_penalty"]),
            smooth_max_bonus_cap=float(self._config["video_smooth_max_bonus_cap"]),
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
            "sequences=%d returned=%d local_videos=%d local_missing=%d local_io_ms=%.1f "
            "local_scoring_ms=%.1f",
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
            local_stats.videos_reranked,
            local_stats.missing_frame_count,
            local_stats.io_ms,
            local_stats.scoring_ms,
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
                    text_truncated=index in truncated_events,
                )
                for index, (event, pool) in enumerate(zip(events, pools))
            ),
            candidate_video_count=len(candidates_by_video),
            chained_video_count=len({chain.video_id for chain in chains}),
            dropped_video_count=dropped,
            sequences_by_video=sequences_by_video,
            local_rerank_video_count=local_stats.videos_reranked,
            local_missing_frame_count=local_stats.missing_frame_count,
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

    async def _encode_events(
        self, events: Sequence[TrakeEventQuery]
    ) -> tuple[np.ndarray, set[int]]:
        """One L2-normalized vector per event, plus the set of event indices
        whose (translated) text exceeded the encoder's token cap.

        Multiple text boxes on one event are prompt-ensembled: encode each,
        average the already-normalized vectors, renormalize. Averaging
        normalized vectors is the standard CLIP-family ensemble and is cheap
        relative to any other accuracy lever here.

        An event with no text gets a zero row. Callers must treat a zero row as
        "no semantic axis for this event" rather than as a real query vector --
        `_build_buckets` does, falling back to that event's KIS hits.

        Truncation detection (plan §T3): `encode_text` always pads/truncates
        silently (app/encoders/fg_clip.py: `_tokenize`), so it can never tell
        the caller a text was cut. `count_text_tokens`/`max_text_tokens` are
        optional (duck-typed, same `hasattr` pattern as `encode_local_phrases`
        below) -- an encoder without them just never flags truncation.
        """
        dim = self._encoder.embedding_dim
        vectors = np.zeros((len(events), dim), dtype=np.float32)
        truncated: set[int] = set()
        count_tokens = getattr(self._encoder, "count_text_tokens", None)
        max_tokens = getattr(self._encoder, "max_text_tokens", None)
        for index, event in enumerate(events):
            if not event.has_text:
                continue
            texts = list(event.text_queries)
            if self._query_translator is not None:
                texts = await asyncio.to_thread(self._translate, texts)
            if count_tokens is not None and max_tokens is not None:
                token_counts = await asyncio.to_thread(count_tokens, texts)
                if any(count > max_tokens for count in token_counts):
                    truncated.add(index)
                    logger.warning(
                        "trake: event %d text exceeds the %d-token encoder cap and "
                        "will be silently truncated (token counts=%s)",
                        index + 1,
                        max_tokens,
                        token_counts,
                    )
            encoded = await asyncio.to_thread(self._encoder.encode_text, texts)
            pooled = np.asarray(encoded, dtype=np.float32).mean(axis=0)
            norm = float(np.linalg.norm(pooled))
            if norm > 0:
                pooled = pooled / norm
            vectors[index] = pooled
        return vectors, truncated

    async def _encode_local_phrases(
        self, events: Sequence[TrakeEventQuery]
    ) -> dict[int, np.ndarray]:
        """One `[num_phrases, dim]` L2-normalized matrix per event that
        carries local phrases (plan §T2 step 3). Unlike KIS's Stage 1 phrase
        rerank, phrases here go through the SAME `QueryTranslator` as event
        text: the TRAKE operator UI is Vietnamese and event text is already
        translated before encoding, so leaving a Vietnamese phrase untranslated
        would score it against an English-trained text tower in silence.
        """
        unique_phrases: dict[str, None] = {}
        for event in events:
            for phrase in event.local_phrases:
                unique_phrases.setdefault(phrase, None)
        if not unique_phrases:
            return {}

        phrases = list(unique_phrases)
        translated = phrases
        if self._query_translator is not None:
            translated = await asyncio.to_thread(self._translate, phrases)
            for original, result in zip(phrases, translated):
                if result != original:
                    logger.info("trake: local phrase translated %r -> %r", original, result)

        encode = getattr(self._encoder, "encode_local_phrases", None) or self._encoder.encode_text
        encoded = await asyncio.to_thread(encode, translated)
        vectors = np.asarray(encoded, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
        vectors = vectors / np.maximum(norms, 1e-12)
        vector_by_phrase = dict(zip(phrases, vectors))

        return {
            index: np.stack([vector_by_phrase[phrase] for phrase in event.local_phrases])
            for index, event in enumerate(events)
            if event.local_phrases
        }

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
        # A local phrase pins a visual region of the event's moment, not a
        # wording of it -- KIS aligns phrase groups by text-query index, so
        # the same phrase group is fanned out to every text query of this
        # event (plan §T1) rather than sent once.
        local_phrases_by_query = (
            [list(event.local_phrases)] * len(event.text_queries) if event.local_phrases else []
        )
        try:
            return await self._search_service.retrieve_kis_candidates(
                text_query=None,
                text_queries=list(event.text_queries),
                local_phrases_by_query=local_phrases_by_query,
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

    def _rank_evidence(
        self, videos: dict[str, _VideoEvidence], num_events: int
    ) -> dict[str, float]:
        """B8 input, normalized to [0, 1] like `coverage` -- `_video_rank_evidence`
        itself is a SUM over covered events, so its ceiling is `num_events`, not
        1. Left un-normalized, `score_videos`'s `rank_evidence_weight` would mean
        a different thing for a 2-event query than a 7-event one: at N=7 the raw
        sum can reach 3.5x `coverage`'s ceiling, so video ranking degenerates
        into "how many events did KIS surface this video for" and stops caring
        how good the chained sequence actually is (measured 2026-08-28; see
        configs/search/trake.yaml: video_rank_evidence_weight).

        Dividing by `num_events` (not by how many of them this video covers)
        is deliberate: dividing by covered-count would erase the breadth signal
        entirely -- a video hit in 1 of 5 events at rank 0 would tie a video hit
        in 5 of 5 events at rank 0. Dividing by `num_events` keeps that
        distinction (0.2 vs 1.0) while landing on the same scale as `coverage`.
        """
        return {
            video_id: self._video_rank_evidence(item) / max(num_events, 1)
            for video_id, item in videos.items()
        }

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
        phrase_vectors_by_event: dict[int, np.ndarray],
        local_rerank_video_ids: frozenset[str],
    ) -> tuple[list[ChainCandidate], int, _LocalRerankStats]:
        chains: list[ChainCandidate] = []
        dropped = 0
        stats = _LocalRerankStats()
        for video_id, item in videos.items():
            local_by_event: dict[int, dict[int, _LocalFrameScore]] = {}
            if phrase_vectors_by_event and video_id in local_rerank_video_ids:
                io_start = time.perf_counter()
                files = self._local_embedding_store.load_video(video_id)
                stats.io_ms += (time.perf_counter() - io_start) * 1000
                if files is None:
                    # Missing dense features (plan §"Trạng thái dữ liệu": 20%
                    # of the corpus today). Already logged once by the store
                    # itself; count every frame this video would have offered
                    # so the diagnostics badge reflects it.
                    stats.missing_frame_count += item.slice.length
                else:
                    scoring_start = time.perf_counter()
                    local_by_event, missing = self._score_local_rerank(
                        files, item.slice, phrase_vectors_by_event
                    )
                    stats.scoring_ms += (time.perf_counter() - scoring_start) * 1000
                    stats.missing_frame_count += missing
                    stats.videos_reranked += 1

            buckets = self._build_buckets(item, candidate_scores, events, local_by_event)
            if buckets is None:
                dropped += 1
                continue
            video_chains = self._chain_with_relax(video_id, buckets)
            if not video_chains:
                dropped += 1
                continue
            chains.extend(video_chains)
        return chains, dropped, stats

    def _score_local_rerank(
        self,
        files: VideoLocalFiles,
        video_slice: TrakeVideoSlice,
        phrase_vectors_by_event: dict[int, np.ndarray],
    ) -> tuple[dict[int, dict[int, _LocalFrameScore]], int]:
        """B2b: MaxSim-score one candidate video's FULL dense array against
        every event that carries a local phrase, then z-score each event's
        row across the video's own frames (mean 0 per video -- the invariant
        that lets `local_rerank.max_videos` cap this step without corrupting
        the B8 cross-video comparison; see configs/search/trake.yaml).

        Reads the whole video, not a pre-filtered pool: measured 2026-08-27
        on data/fgclip_dense_8x8, the frame MaxSim ends up preferring can
        land as deep as rank 273-308 in the video's own global-cosine order,
        so a small pre-filtered pool would routinely miss it -- and one
        video's per-event buckets overlap enough that a full read costs
        about the same I/O as a filtered one anyway.
        """
        row_by_corpus: dict[int, int] = {}
        missing = 0
        for corpus_index in range(video_slice.start, video_slice.stop):
            frame_idx = self._trake_cache.metadata[corpus_index].frame_idx
            row = files.row_by_frame_idx.get(frame_idx)
            if row is None:
                missing += 1
                continue
            row_by_corpus[corpus_index] = row
        if not row_by_corpus:
            return {}, missing

        features = np.asarray(files.embeddings, dtype=np.float32)
        local_by_event: dict[int, dict[int, _LocalFrameScore]] = {}
        for event_index, phrase_vectors in phrase_vectors_by_event.items():
            raw_scores = np.empty(features.shape[0], dtype=np.float32)
            for start in range(0, features.shape[0], self._local_score_batch_size):
                chunk = features[start : start + self._local_score_batch_size]
                raw_scores[start : start + self._local_score_batch_size] = maxsim_query_scores(
                    chunk, phrase_vectors
                )
            z_row = zscore_rows(raw_scores.reshape(1, -1))[0]
            local_by_event[event_index] = {
                corpus_index: _LocalFrameScore(raw=float(raw_scores[row]), z=float(z_row[row]))
                for corpus_index, row in row_by_corpus.items()
            }
        return local_by_event, missing

    def _build_buckets(
        self,
        item: _VideoEvidence,
        candidate_scores: _CandidateScores,
        events: Sequence[TrakeEventQuery],
        local_by_event: dict[int, dict[int, _LocalFrameScore]],
    ) -> list[list[CandidateFrame]] | None:
        """B2 + B4: candidates for each event inside one video.

        Union of (a) the frames KIS returned for this event in this video and
        (b) a local top-k backfill against this video's whole slice. (b) is
        what stops a weak event from emptying the bucket; (a) is what carries
        the OCR / Title / ASR evidence in. When B2b ran for this video,
        `local_by_event[event_index]`'s z-scores also steer which frames (b)
        picks -- otherwise a frame MaxSim prefers but global cosine ranks low
        (measured: as low as rank 273-308) never reaches the bucket at all.
        """
        bucket_top_k = int(self._config["bucket_top_k"])
        nms_seconds = float(self._config["bucket_nms_seconds"])
        rrf_k = int(self._config["kis_evidence_rrf_k"])
        evidence_weight = float(self._config["kis_evidence_weight"])
        local_weight = float(self._config.get("local_rerank", {}).get("weight", 0.0))
        video_slice = item.slice

        buckets: list[list[CandidateFrame]] = []
        for event_index, event in enumerate(events):
            ranks: dict[int, int] = {}
            for corpus_index, rank in item.hits_by_event.get(event_index, []):
                previous = ranks.get(corpus_index)
                if previous is None or rank < previous:
                    ranks[corpus_index] = rank

            local_for_event = local_by_event.get(event_index, {})

            corpus_indices: set[int] = set(ranks)
            if event.has_text:
                corpus_indices |= self._local_top_indices(
                    candidate_scores,
                    event_index,
                    video_slice,
                    bucket_top_k,
                    local_for_event,
                    local_weight,
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
                    local=local_for_event.get(corpus_index),
                    rank=ranks.get(corpus_index),
                    rrf_k=rrf_k,
                    evidence_weight=evidence_weight,
                    local_weight=local_weight,
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
        local_for_event: dict[int, _LocalFrameScore],
        local_weight: float,
    ) -> set[int]:
        # Every corpus row of a candidate video has a column in `scores` (the
        # matrix was built from exactly these slices), so `col_of` never misses.
        columns = [scores.col_of[index] for index in range(video_slice.start, video_slice.stop)]
        if not columns:
            return set()
        # Fancy indexing on both axes already returns a copy, so this window
        # is safe to mutate in place without touching `scores.matrix`.
        window = scores.matrix[event_index, columns]
        if local_for_event:
            for offset, corpus_index in enumerate(range(video_slice.start, video_slice.stop)):
                local_score = local_for_event.get(corpus_index)
                if local_score is not None:
                    window[offset] += local_weight * local_score.z
        take = min(top_k, window.size)
        top_positions = np.argpartition(window, -take)[-take:]
        # `columns` is in ascending corpus-index order, so local column i maps
        # back to corpus index `video_slice.start + i`.
        return {video_slice.start + int(index) for index in top_positions}

    def _to_candidate_frame(
        self,
        corpus_index: int,
        *,
        semantic_z: float,
        local: _LocalFrameScore | None,
        rank: int | None,
        rrf_k: int,
        evidence_weight: float,
        local_weight: float,
    ) -> CandidateFrame:
        cached = self._trake_cache.metadata[corpus_index]
        evidence = kis_evidence_score(rank, rrf_k)
        local_z = local.z if local is not None else 0.0
        return CandidateFrame(
            corpus_index=corpus_index,
            keyframe_id=cached.keyframe_id,
            video_id=cached.video_id,
            frame_idx=cached.frame_idx,
            timestamp=cached.timestamp,
            score=semantic_z + local_weight * local_z + evidence_weight * evidence,
            semantic_z=semantic_z,
            kis_evidence=evidence,
            kis_rank=rank,
            local_score=local.raw if local is not None else None,
            local_z=local_z,
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
