"""SearchService — the first end-to-end query bridge (plan §P0.8).

`mode` is a parameter of one function, not copy-pasted code paths (CLAUDE.md
§4): `semantic` embeds the query with the injected encoder and searches
Milvus; `ocr` full-text-searches Elasticsearch and must join to MongoDB
afterward because ES OCR hits carry no `frame_idx` at all (CLAUDE.md §3 only
requires `frame_idx` in Milvus/MongoDB).

`semantic` translates the query VI->EN before encoding when a
`QueryTranslator` is injected (plan §P0.9): the production vision-language
text tower is English-caption oriented, while the OCR index stores Vietnamese
text. `ocr` never translates -- an English OCR query would measure translation
loss, not the retrieval channel.

`search_by_image` (plan §P1.1) is image-to-image search: it shares the same
Milvus-vector-to-SearchHit mapping as `semantic` (`_search_milvus_vector`),
differing only in which encoder branch produces the query vector.

`hybrid` mode (RRF-fused `semantic` + `ocr`, plan §P1.3) was REMOVED
2026-08-12: measured (tests/integration/test_hybrid_fusion.py, now deleted)
as worse than the best single channel in both directions, and the eval
label set structurally cannot measure the ocr channel fairly -- the caption
prompt in eval/build_eval_set.py explicitly forbids transcribing on-screen
text, so `ocr`'s 0.0254 reflects incidental word overlap, not retrieval
quality (CLAUDE.md §8: "BM25 cao bất thường => nghi bộ nhãn rò rỉ" cuts both
ways -- an artificially LOW score from a label set that structurally can't
credit the channel is just as invalid to act on). RRF fusing by RANK, not
score magnitude, made this worse: mixing a much weaker channel into a much
stronger one drags the strong one down regardless of true channel quality.
OCR is being repositioned as its own first-class TEXT/BM25 mode instead
(docs/research-encoder-vietnamese.md §8, chủ dự án 2026-08-12) -- not fused
into ranking, matching how the strongest AIC teams (Vortex, MERVIN) use OCR
as an operator-driven filter/mode rather than an automatic-ranking channel.

`_reciprocal_rank_fusion` below is KEPT (not deleted) for a future genuine
use: fusing two semantic encoders of comparable quality once a second
encoder exists -- unlike semantic-vs-ocr, two same-kind channels are the
case RRF-by-rank is designed for. It is currently unused/unwired.

Milvus search, encoder inference, and translation are synchronous, CPU/GPU-
bound calls; they run off the event loop via `asyncio.to_thread` so this
async method never blocks other requests (.claude/rules/ecc/python/fastapi.md).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import yaml
from elasticsearch import ConnectionError as EsConnectionError
from PIL import Image

from app.encoders.base import MultimodalEncoder
from app.services.abe import AbeConceptBank, AbeError, AbePatchCache, AbeRankedQuery, AbeScorer
from app.services.asr_retrieval_service import (
    AsrGateResult,
    AsrRetrievalService,
    AsrRetrievalUnavailable,
)
from app.services.kis_scoring import (
    AbeModalityScore,
    AbeQueryScore,
    KisConfig,
    KisQuery,
    OcrModalityScore,
    OcrQueryScore,
    QueryType,
    RankedScore,
    Stage1OcrScore,
    Stage1RetrievalScore,
    Stage1TextScore,
    TextModalityScore,
    TextQueryScore,
    bm25_reference,
    candidate_depth,
    clamp01,
    geometric_mean_score,
    normalize_bm25,
    ocr_coverage,
    rank_percentiles,
)
from app.services.query_translator import QueryTranslator
from app.stores.es_client import EsStore
from app.stores.milvus_client import MilvusStore
from app.stores.mongo_client import MongoStore

logger = logging.getLogger(__name__)

SearchMode = Literal["semantic", "ocr"]
_SEARCH_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent.parent / "configs" / "search" / "search.yaml"
)
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_RRF_K = 60  # plan §P1.3: RRF's own well-known literature default,
# kept only as _reciprocal_rank_fusion's own default -- no caller wires it
# today (module docstring: hybrid mode removed 2026-08-12).
_MISSING_TEXT_RAW_SCORE = -1.0
_EXPECTED_KIS_ENCODER_KIND = "fg_clip"


@dataclass(frozen=True)
class _OcrQueryComponents:
    raw_bm25: float
    bm25_reference: float
    bm25_normalized: float
    coverage: float
    query_score: float


@dataclass(frozen=True)
class _Stage1OcrCandidate:
    hit: dict
    score: Stage1OcrScore


class SearchServiceError(Exception):
    """Raised for an expected failure the caller must react to. Two distinct
    causes share this type: a caller mistake (unknown mode -- the original,
    default meaning) and a degraded dependency (Elasticsearch down for
    `ocr`, plan §P1.8). `error_code`/`status_code` let
    `app/api/search.py` map each cause to the right HTTP response without
    parsing the message text (CLAUDE.md §7: every error needs a
    machine-readable code, not just a human-readable one)."""

    def __init__(
        self,
        message: str,
        error_code: str = "invalid_search_mode",
        status_code: int = 400,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.status_code = status_code


@dataclass(frozen=True)
class SearchHit:
    keyframe_id: str
    video_id: str
    score: float
    frame_idx: int
    timestamp: float
    thumbnail_path: str
    final_score: float | None = None
    active_modalities: int | None = None
    modalities: dict[str, TextModalityScore | OcrModalityScore | AbeModalityScore] | None = None
    stage1: Stage1RetrievalScore | None = None


@dataclass(frozen=True)
class KisCandidatePool:
    candidate_ids: list[str]
    text_candidate_count: int
    ocr_candidate_count: int
    results: list[SearchHit]
    asr_gate: AsrGateResult | None = None

    @property
    def candidate_pool_size(self) -> int:
        return len(self.candidate_ids)


class SearchService:
    def __init__(
        self,
        encoder: MultimodalEncoder,
        milvus_store: MilvusStore,
        es_store: EsStore,
        mongo_store: MongoStore,
        query_translator: QueryTranslator | None = None,
        kis_config: KisConfig | None = None,
        abe_concept_bank: AbeConceptBank | None = None,
        abe_patch_cache: AbePatchCache | None = None,
        asr_retrieval_service: AsrRetrievalService | None = None,
    ) -> None:
        self._encoder = encoder
        self._milvus_store = milvus_store
        self._es_store = es_store
        self._mongo_store = mongo_store
        self._query_translator = query_translator
        self._kis_config = kis_config or _load_kis_config()
        self._abe_concept_bank = abe_concept_bank
        self._abe_patch_cache = abe_patch_cache or AbePatchCache(
            dtype=self._kis_config.abe_patch_cache_dtype
        )
        self._abe_scorer: AbeScorer | None = None
        self._asr_retrieval_service = asr_retrieval_service

    async def search(self, query: str, mode: SearchMode, top_k: int) -> list[SearchHit]:
        if mode == "semantic":
            return await self._search_semantic(query, top_k)
        if mode == "ocr":
            return await self._search_ocr(query, top_k)
        raise SearchServiceError(f"Unknown search mode: {mode!r}")

    async def search_by_image(self, image: Image.Image, top_k: int) -> list[SearchHit]:
        """Image-to-image search (plan §P1.1, US-3): encodes `image` with the
        vision branch of the SAME encoder/preprocessing used at ingest time
        (CLAUDE.md §3 -- vectors from two different pipelines is a silent
        failure, not a crash) and searches the same Milvus collection text
        queries do, so both modes share one vector space."""
        vector = await asyncio.to_thread(self._encoder.encode_image, [image])
        return await self._search_milvus_vector(vector[0], top_k)

    async def search_kis(self, queries: Sequence[KisQuery], top_k: int) -> list[SearchHit]:
        raise SearchServiceError(
            "KIS search now requires Stage 1 candidate retrieval followed by Stage 2 reranking",
            error_code="invalid_kis_request",
            status_code=400,
        )

    async def retrieve_kis_candidates(
        self,
        text_query: str | None,
        text_top_k: int,
        ocr_query: str | None,
        ocr_top_k: int,
        asr_query: str | None = None,
        asr_video_top_k: int = 20,
    ) -> KisCandidatePool:
        text_query = text_query.strip() if text_query is not None else None
        ocr_query = ocr_query.strip() if ocr_query is not None else None
        asr_query = asr_query.strip() if asr_query is not None else None
        if not text_query and not ocr_query:
            raise SearchServiceError(
                "Stage 1 requires Text Q0 or OCR query",
                error_code="invalid_kis_request",
                status_code=400,
            )

        asr_gate: AsrGateResult | None = None
        video_ids: list[str] | None = None
        if asr_query:
            if self._asr_retrieval_service is None:
                raise self._asr_unavailable_error()
            try:
                asr_gate = await self._asr_retrieval_service.retrieve(asr_query, asr_video_top_k)
            except AsrRetrievalUnavailable as exc:
                raise self._asr_unavailable_error() from exc
            video_ids = asr_gate.video_ids
            if not video_ids:
                return KisCandidatePool([], 0, 0, [], asr_gate=asr_gate)

        text_stage1_by_id: dict[str, Stage1TextScore] = {}
        text_metadata_by_id: dict[str, dict] = {}
        text_candidate_ids: list[str] = []
        text_query_vector: np.ndarray | None = None
        if text_query:
            _assert_kis_fg_clip_encoder(self._encoder)
            text_inputs = await self._prepare_text_inputs([text_query])
            text_vectors = await asyncio.to_thread(self._encoder.encode_text, text_inputs)
            text_query_vector = np.asarray(text_vectors[0], dtype=np.float32)
            text_hits = await asyncio.to_thread(
                self._milvus_store.search, text_vectors[0], text_top_k, "", video_ids
            )
            for rank, hit in enumerate(text_hits, start=1):
                keyframe_id = hit["keyframe_id"]
                if keyframe_id not in text_stage1_by_id:
                    text_candidate_ids.append(keyframe_id)
                    text_stage1_by_id[keyframe_id] = Stage1TextScore(
                        query=text_query,
                        raw_cosine=float(hit["distance"]),
                        rank=rank,
                    )
                    text_metadata_by_id[keyframe_id] = _metadata_from_milvus_hit(hit)

        ocr_stage1_by_id: dict[str, Stage1OcrScore] = {}
        ocr_candidate_ids: list[str] = []
        if ocr_query:
            ocr_candidates = await self._retrieve_stage1_ocr_candidates(
                ocr_query, ocr_top_k, video_ids=video_ids
            )
            for candidate in ocr_candidates:
                keyframe_id = candidate.hit["keyframe_id"]
                if keyframe_id not in ocr_stage1_by_id:
                    ocr_candidate_ids.append(keyframe_id)
                    ocr_stage1_by_id[keyframe_id] = candidate.score

        candidate_ids = _union_stage1_candidate_ids(text_candidate_ids, ocr_candidate_ids)
        if not candidate_ids:
            if asr_gate is not None:
                logger.warning(
                    "retrieve_kis_candidates: ASR gate returned %d video(s) but no valid "
                    "keyframe hits",
                    asr_gate.candidate_video_count,
                )
                for video_id in asr_gate.video_ids:
                    logger.warning(
                        "retrieve_kis_candidates: ASR-gated video_id=%s has no valid keyframe hit",
                        video_id,
                    )
            return KisCandidatePool(
                candidate_ids=[],
                text_candidate_count=0,
                ocr_candidate_count=0,
                results=[],
                asr_gate=asr_gate,
            )

        text_source_ids = set(text_candidate_ids)
        ocr_source_ids = set(ocr_candidate_ids)

        vector_rows_by_id: dict[str, dict] = {}
        if text_query and text_query_vector is not None:
            vector_rows_by_id = await asyncio.to_thread(
                self._milvus_store.get_vectors_by_ids, candidate_ids
            )
            text_scores_by_id: dict[str, float] = {}
            for keyframe_id in candidate_ids:
                vector_row = vector_rows_by_id.get(keyframe_id)
                if vector_row is not None:
                    text_scores_by_id[keyframe_id] = _cosine_similarity(
                        text_query_vector,
                        np.asarray(vector_row["embedding"], dtype=np.float32),
                    )
                else:
                    text_scores_by_id[keyframe_id] = text_stage1_by_id.get(
                        keyframe_id,
                        Stage1TextScore(text_query, _MISSING_TEXT_RAW_SCORE, len(candidate_ids)),
                    ).raw_cosine
            text_rankings = rank_percentiles(text_scores_by_id, candidate_ids)
            text_stage1_by_id = {
                keyframe_id: Stage1TextScore(
                    query=text_query,
                    raw_cosine=text_rankings[keyframe_id].raw_score,
                    rank=text_rankings[keyframe_id].rank,
                )
                for keyframe_id in candidate_ids
            }

        if ocr_query:
            ocr_reference = _stage1_ocr_reference(ocr_stage1_by_id)
            try:
                ocr_texts_by_id = await self._es_store.get_by_ids(candidate_ids)
                bm25_by_id = await self._es_store.score_by_ids(ocr_query, candidate_ids)
            except EsConnectionError as exc:
                raise self._ocr_unavailable_error() from exc
            ocr_components_by_id: dict[str, _OcrQueryComponents] = {}
            ocr_scores_by_id: dict[str, float] = {}
            for keyframe_id in candidate_ids:
                raw_bm25 = bm25_by_id.get(keyframe_id, 0.0)
                bm25_normalized = normalize_bm25(raw_bm25, ocr_reference)
                coverage = ocr_coverage(
                    ocr_query,
                    ocr_texts_by_id.get(keyframe_id, ""),
                    self._kis_config,
                )
                query_score = clamp01(bm25_normalized * coverage)
                ocr_components_by_id[keyframe_id] = _OcrQueryComponents(
                    raw_bm25=float(raw_bm25),
                    bm25_reference=float(ocr_reference),
                    bm25_normalized=float(bm25_normalized),
                    coverage=float(coverage),
                    query_score=float(query_score),
                )
                ocr_scores_by_id[keyframe_id] = float(query_score)
            ocr_rankings = rank_percentiles(ocr_scores_by_id, candidate_ids)
            ocr_stage1_by_id = {
                keyframe_id: Stage1OcrScore(
                    query=ocr_query,
                    raw_bm25=ocr_components_by_id[keyframe_id].raw_bm25,
                    bm25_reference=ocr_components_by_id[keyframe_id].bm25_reference,
                    bm25_normalized=ocr_components_by_id[keyframe_id].bm25_normalized,
                    coverage=ocr_components_by_id[keyframe_id].coverage,
                    query_score=ocr_components_by_id[keyframe_id].query_score,
                    rank=ocr_rankings[keyframe_id].rank,
                )
                for keyframe_id in candidate_ids
            }

        stage1_fused_scores = _stage1_fused_scores(
            candidate_ids=candidate_ids,
            text_scores_by_id=(
                {
                    keyframe_id: text_stage1_by_id[keyframe_id].raw_cosine
                    for keyframe_id in candidate_ids
                }
                if text_query
                else None
            ),
            ocr_scores_by_id=(
                {
                    keyframe_id: ocr_stage1_by_id[keyframe_id].query_score
                    for keyframe_id in candidate_ids
                }
                if ocr_query
                else None
            ),
            config=self._kis_config,
        )

        mongo_docs = await self._mongo_store.get_keyframes_by_ids(candidate_ids)
        valid_candidate_ids: list[str] = []
        results: list[SearchHit] = []
        valid_text_ids: set[str] = set()
        valid_ocr_ids: set[str] = set()
        for keyframe_id in candidate_ids:
            metadata = _stage1_candidate_metadata(
                keyframe_id=keyframe_id,
                text_metadata_by_id=text_metadata_by_id,
                mongo_docs=mongo_docs,
            )
            if metadata is None:
                logger.warning(
                    "retrieve_kis_candidates: dropping keyframe_id=%s -- no Stage 1 metadata",
                    keyframe_id,
                )
                continue

            text_score = text_stage1_by_id.get(keyframe_id)
            ocr_score = ocr_stage1_by_id.get(keyframe_id)
            source_modalities: list[str] = []
            if keyframe_id in text_source_ids:
                source_modalities.append("text")
                valid_text_ids.add(keyframe_id)
            if keyframe_id in ocr_source_ids:
                source_modalities.append("ocr")
                valid_ocr_ids.add(keyframe_id)
            if not source_modalities:
                continue

            valid_candidate_ids.append(keyframe_id)
            stage1 = Stage1RetrievalScore(
                source_modalities=source_modalities,
                text=text_score,
                ocr=ocr_score,
            )
            results.append(
                SearchHit(
                    keyframe_id=keyframe_id,
                    video_id=metadata["video_id"],
                    score=stage1_fused_scores.get(
                        keyframe_id,
                        _stage1_display_score(text_score, ocr_score),
                    ),
                    frame_idx=metadata["frame_idx"],
                    timestamp=metadata["timestamp"],
                    thumbnail_path=metadata["thumbnail_path"],
                    stage1=stage1,
                )
            )

        results.sort(key=lambda hit: hit.score, reverse=True)
        if asr_gate is not None:
            returned_video_ids = {hit.video_id for hit in results}
            for video_id in asr_gate.video_ids:
                if video_id not in returned_video_ids:
                    logger.warning(
                        "retrieve_kis_candidates: ASR-gated video_id=%s has no valid keyframe hit",
                        video_id,
                    )
        return KisCandidatePool(
            candidate_ids=valid_candidate_ids,
            text_candidate_count=len(valid_text_ids),
            ocr_candidate_count=len(valid_ocr_ids),
            results=results,
            asr_gate=asr_gate,
        )

    async def _retrieve_stage1_ocr_candidates(
        self,
        query_text: str,
        top_k: int,
        video_ids: Sequence[str] | None = None,
    ) -> list[_Stage1OcrCandidate]:
        retrieval_k = candidate_depth(top_k, self._kis_config)
        try:
            hits = await self._es_store.search(query_text, retrieval_k, video_ids=video_ids)
        except EsConnectionError as exc:
            raise self._ocr_unavailable_error() from exc

        reference = bm25_reference([float(hit.get("score", 0.0)) for hit in hits], self._kis_config)
        candidates: list[tuple[dict, _OcrQueryComponents, int]] = []
        for initial_rank, hit in enumerate(hits, start=1):
            raw_bm25 = float(hit.get("score", 0.0))
            bm25_normalized = normalize_bm25(raw_bm25, reference)
            coverage = ocr_coverage(query_text, hit.get("ocr_text", ""), self._kis_config)
            query_score = clamp01(bm25_normalized * coverage)
            candidates.append(
                (
                    hit,
                    _OcrQueryComponents(
                        raw_bm25=raw_bm25,
                        bm25_reference=float(reference),
                        bm25_normalized=float(bm25_normalized),
                        coverage=float(coverage),
                        query_score=float(query_score),
                    ),
                    initial_rank,
                )
            )

        candidates.sort(key=lambda item: (-item[1].query_score, -item[1].raw_bm25, item[2]))
        selected: list[_Stage1OcrCandidate] = []
        for rank, (hit, components, _) in enumerate(candidates[:top_k], start=1):
            selected.append(
                _Stage1OcrCandidate(
                    hit=hit,
                    score=Stage1OcrScore(
                        query=query_text,
                        raw_bm25=components.raw_bm25,
                        bm25_reference=components.bm25_reference,
                        bm25_normalized=components.bm25_normalized,
                        coverage=components.coverage,
                        query_score=components.query_score,
                        rank=rank,
                    ),
                )
            )
        return selected

    async def rerank_kis_candidates(
        self,
        candidate_ids: Sequence[str],
        queries: Sequence[KisQuery],
        final_top_k: int,
    ) -> list[SearchHit]:
        candidate_ids = list(
            dict.fromkeys(
                keyframe_id.strip() for keyframe_id in candidate_ids if keyframe_id.strip()
            )
        )
        clean_queries = [q for q in queries if _kis_query_is_active(q)]
        if not candidate_ids:
            raise SearchServiceError(
                "Stage 2 requires a non-empty Stage 1 candidate pool",
                error_code="invalid_kis_request",
                status_code=400,
            )
        if not clean_queries:
            raise SearchServiceError(
                "Stage 2 requires at least one subquery",
                error_code="invalid_kis_request",
                status_code=400,
            )
        if final_top_k > len(candidate_ids):
            raise SearchServiceError(
                "final_top_k must be <= candidate_pool_size",
                error_code="invalid_kis_request",
                status_code=400,
            )

        text_queries = [q for q in clean_queries if q.type == QueryType.TEXT]
        ocr_queries = [q for q in clean_queries if q.type == QueryType.OCR]
        abe_queries = [q for q in clean_queries if q.type == QueryType.ABE]

        text_vectors = np.empty((0, self._encoder.embedding_dim), dtype=np.float32)
        vector_rows_by_id: dict[str, dict] = {}
        if text_queries:
            _assert_kis_fg_clip_encoder(self._encoder)
            text_inputs = await self._prepare_text_inputs([q.text for q in text_queries])
            text_vectors = await asyncio.to_thread(self._encoder.encode_text, text_inputs)
        if abe_queries:
            _assert_kis_fg_clip_encoder(self._encoder)
        if text_queries or abe_queries:
            vector_rows_by_id = await asyncio.to_thread(
                self._milvus_store.get_vectors_by_ids, candidate_ids
            )

        mongo_docs = await self._mongo_store.get_keyframes_by_ids(candidate_ids)

        ocr_texts_by_id: dict[str, str] = {}
        bm25_by_query: dict[int, dict[str, float]] = {}
        bm25_references: dict[int, float] = {}
        if ocr_queries:
            try:
                ocr_texts_by_id = await self._es_store.get_by_ids(candidate_ids)
                for query in ocr_queries:
                    bm25_scores = await self._es_store.score_by_ids(query.text, candidate_ids)
                    bm25_by_query[query.query_index] = bm25_scores
                    bm25_references[query.query_index] = bm25_reference(
                        list(bm25_scores.values()), self._kis_config
                    )
            except EsConnectionError as exc:
                raise self._ocr_unavailable_error() from exc

        metadata_by_id = _candidate_metadata_by_id(candidate_ids, mongo_docs, vector_rows_by_id)
        candidate_pool_ids = [
            keyframe_id for keyframe_id in candidate_ids if keyframe_id in metadata_by_id
        ]
        if not candidate_pool_ids:
            return []
        if final_top_k > len(candidate_pool_ids):
            raise SearchServiceError(
                "final_top_k must be <= candidate_pool_size",
                error_code="invalid_kis_request",
                status_code=400,
            )

        text_vector_index = {query.query_index: i for i, query in enumerate(text_queries)}
        text_rankings: dict[int, dict[str, RankedScore]] = {}
        if text_queries:
            text_rankings = _rank_text_queries(
                candidate_pool_ids=candidate_pool_ids,
                text_queries=text_queries,
                text_vectors=text_vectors,
                text_vector_index=text_vector_index,
                vector_rows_by_id=vector_rows_by_id,
            )

        ocr_components_by_query: dict[int, dict[str, _OcrQueryComponents]] = {}
        ocr_rankings: dict[int, dict[str, RankedScore]] = {}
        if ocr_queries:
            ocr_components_by_query, ocr_rankings = _rank_ocr_queries(
                candidate_pool_ids=candidate_pool_ids,
                ocr_queries=ocr_queries,
                ocr_texts_by_id=ocr_texts_by_id,
                bm25_by_query=bm25_by_query,
                bm25_references=bm25_references,
                config=self._kis_config,
            )

        abe_ranked_by_query: dict[int, AbeRankedQuery] = {}
        if abe_queries:
            try:
                abe_ranked_by_query = await asyncio.to_thread(
                    self._rank_abe_queries,
                    candidate_pool_ids,
                    abe_queries,
                    mongo_docs,
                    vector_rows_by_id,
                )
            except AbeError as exc:
                raise SearchServiceError(
                    str(exc), error_code="abe_unavailable", status_code=503
                ) from exc

        scored_hits: list[SearchHit] = []
        for keyframe_id in candidate_pool_ids:
            metadata = metadata_by_id[keyframe_id]
            modalities: dict[str, TextModalityScore | OcrModalityScore | AbeModalityScore] = {}
            if text_queries:
                modalities["text"] = _build_text_modality_score(
                    keyframe_id=keyframe_id,
                    text_queries=text_queries,
                    text_rankings=text_rankings,
                    config=self._kis_config,
                )
            if abe_queries:
                modalities["abe"] = _build_abe_modality_score(
                    keyframe_id=keyframe_id,
                    abe_queries=abe_queries,
                    abe_ranked_by_query=abe_ranked_by_query,
                    config=self._kis_config,
                )
            if ocr_queries:
                modalities["ocr"] = _build_ocr_modality_score(
                    keyframe_id=keyframe_id,
                    ocr_queries=ocr_queries,
                    ocr_components_by_query=ocr_components_by_query,
                    ocr_rankings=ocr_rankings,
                    config=self._kis_config,
                )

            active_modalities = len(modalities)
            final_score = geometric_mean_score(
                [modality.score for modality in modalities.values()], self._kis_config
            )
            scored_hits.append(
                SearchHit(
                    keyframe_id=keyframe_id,
                    video_id=metadata["video_id"],
                    score=final_score,
                    frame_idx=metadata["frame_idx"],
                    timestamp=metadata["timestamp"],
                    thumbnail_path=metadata["thumbnail_path"],
                    final_score=final_score,
                    active_modalities=active_modalities,
                    modalities=modalities,
                )
            )

        scored_hits.sort(
            key=lambda hit: hit.final_score if hit.final_score is not None else hit.score,
            reverse=True,
        )
        return scored_hits[:final_top_k]

    def _rank_abe_queries(
        self,
        candidate_pool_ids: Sequence[str],
        abe_queries: Sequence[KisQuery],
        mongo_docs: dict[str, dict],
        vector_rows_by_id: dict[str, dict],
    ) -> dict[int, AbeRankedQuery]:
        missing_image_ids = [
            keyframe_id
            for keyframe_id in candidate_pool_ids
            if not (mongo_docs.get(keyframe_id) or {}).get("image_path")
        ]
        if missing_image_ids:
            raise AbeError(
                "ABE requires Mongo image_path for every candidate; "
                f"missing ids include {missing_image_ids[:5]}"
            )

        image_paths_by_id = {
            keyframe_id: Path(mongo_docs[keyframe_id]["image_path"])
            for keyframe_id in candidate_pool_ids
        }
        global_vectors_by_id = {
            keyframe_id: np.asarray(vector_rows_by_id[keyframe_id]["embedding"], dtype=np.float32)
            for keyframe_id in candidate_pool_ids
            if keyframe_id in vector_rows_by_id
        }
        return self._get_abe_scorer().score_queries(
            candidate_ids=candidate_pool_ids,
            queries=abe_queries,
            image_paths_by_id=image_paths_by_id,
            global_image_vectors_by_id=global_vectors_by_id,
            patch_cache=self._abe_patch_cache,
        )

    def _get_abe_scorer(self) -> AbeScorer:
        if self._abe_scorer is None:
            self._abe_scorer = AbeScorer(
                encoder=self._encoder,
                concept_bank=self._get_abe_concept_bank(),
                config=self._kis_config,
            )
        return self._abe_scorer

    def _get_abe_concept_bank(self) -> AbeConceptBank:
        expected_checkpoint = getattr(self._encoder, "checkpoint_id", None)
        if self._abe_concept_bank is not None:
            self._abe_concept_bank.validate_compatibility(
                expected_model_id=self._encoder.model_id,
                expected_embedding_dim=self._encoder.embedding_dim,
                expected_encoder_kind=_EXPECTED_KIS_ENCODER_KIND,
                expected_checkpoint=expected_checkpoint,
            )
            return self._abe_concept_bank

        path = Path(self._kis_config.abe_concept_bank_path)
        if not path.is_absolute():
            path = _REPO_ROOT / path
        try:
            self._abe_concept_bank = AbeConceptBank.load(
                path,
                expected_model_id=self._encoder.model_id,
                expected_embedding_dim=self._encoder.embedding_dim,
                expected_encoder_kind=_EXPECTED_KIS_ENCODER_KIND,
                expected_checkpoint=expected_checkpoint,
            )
        except FileNotFoundError as exc:
            raise AbeError(f"ABE concept bank not found: {path}") from exc
        except ValueError as exc:
            raise AbeError(str(exc)) from exc
        return self._abe_concept_bank

    async def _search_semantic(self, query: str, top_k: int) -> list[SearchHit]:
        if self._query_translator is not None:
            query = await asyncio.to_thread(self._query_translator.translate, query)
        vector = await asyncio.to_thread(self._encoder.encode_text, [query])
        return await self._search_milvus_vector(vector[0], top_k)

    async def _prepare_text_inputs(self, queries: Sequence[str]) -> list[str]:
        if self._query_translator is None:
            return list(queries)

        def translate_all() -> list[str]:
            return [self._query_translator.translate(query) for query in queries]

        return await asyncio.to_thread(translate_all)

    async def _search_milvus_vector(self, vector: np.ndarray, top_k: int) -> list[SearchHit]:
        hits = await asyncio.to_thread(self._milvus_store.search, vector, top_k)
        mongo_docs = await self._mongo_store.get_keyframes_by_ids(
            [hit["keyframe_id"] for hit in hits]
        )

        results = []
        for hit in hits:
            keyframe_id = hit["keyframe_id"]
            entity = hit["entity"]
            mongo_doc = mongo_docs.get(keyframe_id)
            if mongo_doc is None:
                # frame_idx/timestamp already came from Milvus, so the hit
                # is still submittable -- only thumbnail_path is missing.
                logger.warning(
                    "search_milvus_vector: keyframe_id=%s has no MongoDB record, "
                    "thumbnail_path unavailable",
                    keyframe_id,
                )
            results.append(
                SearchHit(
                    keyframe_id=keyframe_id,
                    video_id=entity["video_id"],
                    score=hit["distance"],
                    frame_idx=entity["frame_idx"],
                    timestamp=entity["timestamp"],
                    thumbnail_path=mongo_doc["thumbnail_path"] if mongo_doc else "",
                )
            )
        return results

    async def _search_ocr(self, query: str, top_k: int) -> list[SearchHit]:
        try:
            hits = await self._es_store.search(query, top_k)
        except EsConnectionError as exc:
            # plan §P1.8, PRD T5: degraded mode -- Elasticsearch down must
            # surface as a readable "OCR unavailable" error, not an
            # unhandled exception that falls through to a generic 500.
            # Semantic search is a separate code path (_search_semantic)
            # and is unaffected.
            raise SearchServiceError(
                "OCR search is unavailable: Elasticsearch is unreachable "
                "(semantic search is unaffected)",
                error_code="ocr_unavailable",
                status_code=503,
            ) from exc
        mongo_docs = await self._mongo_store.get_keyframes_by_ids(
            [hit["keyframe_id"] for hit in hits]
        )

        results = []
        for hit in hits:
            keyframe_id = hit["keyframe_id"]
            mongo_doc = mongo_docs.get(keyframe_id)
            if mongo_doc is None:
                # ES OCR hits never carry frame_idx (CLAUDE.md §3); with no
                # Mongo record to supply it, the hit has no submittable
                # frame_idx and must be dropped, not silently guessed.
                logger.warning(
                    "search_ocr: dropping keyframe_id=%s -- no MongoDB record, "
                    "frame_idx unavailable",
                    keyframe_id,
                )
                continue
            results.append(
                SearchHit(
                    keyframe_id=keyframe_id,
                    video_id=hit["video_id"],
                    score=hit["score"],
                    frame_idx=mongo_doc["frame_idx"],
                    timestamp=hit["timestamp"],
                    thumbnail_path=mongo_doc["thumbnail_path"],
                )
            )
        return results

    @staticmethod
    def _ocr_unavailable_error() -> SearchServiceError:
        return SearchServiceError(
            "OCR search is unavailable: Elasticsearch is unreachable "
            "(semantic search is unaffected)",
            error_code="ocr_unavailable",
            status_code=503,
        )

    @staticmethod
    def _asr_unavailable_error() -> SearchServiceError:
        return SearchServiceError(
            "ASR search is unavailable: transcript retrieval is unreachable",
            error_code="asr_unavailable",
            status_code=503,
        )


def _load_kis_config() -> KisConfig:
    with _SEARCH_CONFIG_PATH.open(encoding="utf-8") as f:
        return KisConfig.from_mapping(yaml.safe_load(f))


def _assert_kis_fg_clip_encoder(encoder: MultimodalEncoder) -> None:
    model_id = str(getattr(encoder, "model_id", ""))
    encoder_kind = getattr(encoder, "encoder_kind", None)
    if encoder_kind is not None:
        if str(encoder_kind).lower() == _EXPECTED_KIS_ENCODER_KIND:
            return
        raise SearchServiceError(
            "KIS semantic encoder must be FG-CLIP Base ViT-B/16; "
            f"got encoder_kind={encoder_kind!r}, model_id={model_id!r}",
            error_code="kis_encoder_mismatch",
            status_code=503,
        )

    normalized_model_id = model_id.lower().replace("_", "")
    if "fgclip" in normalized_model_id or normalized_model_id.startswith("fake"):
        return
    raise SearchServiceError(
        "KIS semantic encoder must be FG-CLIP Base ViT-B/16; "
        f"could not verify model_id={model_id!r} as FG-CLIP",
        error_code="kis_encoder_mismatch",
        status_code=503,
    )


def _kis_query_is_active(query: KisQuery) -> bool:
    if query.type == QueryType.ABE:
        return any(pair.attribute.strip() and pair.object.strip() for pair in query.pairs)
    return bool(query.text.strip())


def _union_candidate_ids(
    text_hits_by_query: dict[int, list[dict]],
    ocr_hits_by_query: dict[int, list[dict]],
) -> list[str]:
    candidate_ids: list[str] = []
    seen: set[str] = set()
    for hits_by_query in (text_hits_by_query, ocr_hits_by_query):
        for query_index in sorted(hits_by_query):
            for hit in hits_by_query[query_index]:
                keyframe_id = hit["keyframe_id"]
                if keyframe_id not in seen:
                    seen.add(keyframe_id)
                    candidate_ids.append(keyframe_id)
    return candidate_ids


def _union_stage1_candidate_ids(
    text_candidate_ids: Sequence[str],
    ocr_candidate_ids: Sequence[str],
) -> list[str]:
    candidate_ids: list[str] = []
    seen: set[str] = set()
    for source_ids in (text_candidate_ids, ocr_candidate_ids):
        for keyframe_id in source_ids:
            if keyframe_id not in seen:
                seen.add(keyframe_id)
                candidate_ids.append(keyframe_id)
    return candidate_ids


def _metadata_from_milvus_hit(hit: dict) -> dict:
    entity = hit.get("entity", {})
    return {
        "video_id": entity.get("video_id", ""),
        "frame_idx": int(entity.get("frame_idx", 0)),
        "timestamp": float(entity.get("timestamp", 0.0)),
    }


def _stage1_candidate_metadata(
    keyframe_id: str,
    text_metadata_by_id: dict[str, dict],
    mongo_docs: dict[str, dict],
) -> dict | None:
    text_metadata = text_metadata_by_id.get(keyframe_id)
    mongo_doc = mongo_docs.get(keyframe_id)
    if text_metadata is not None:
        return {
            "video_id": text_metadata["video_id"] or (mongo_doc or {}).get("video_id", ""),
            "frame_idx": int(
                text_metadata["frame_idx"]
                if text_metadata["frame_idx"] is not None
                else (mongo_doc or {}).get("frame_idx", 0)
            ),
            "timestamp": float(
                text_metadata["timestamp"]
                if text_metadata["timestamp"] is not None
                else (mongo_doc or {}).get("timestamp", 0.0)
            ),
            "thumbnail_path": mongo_doc["thumbnail_path"] if mongo_doc else "",
        }
    if mongo_doc is None:
        return None
    return {
        "video_id": mongo_doc["video_id"],
        "frame_idx": int(mongo_doc["frame_idx"]),
        "timestamp": float(mongo_doc["timestamp"]),
        "thumbnail_path": mongo_doc["thumbnail_path"],
    }


def _stage1_display_score(
    text_score: Stage1TextScore | None,
    ocr_score: Stage1OcrScore | None,
) -> float:
    if text_score is not None:
        return text_score.raw_cosine
    if ocr_score is not None:
        return ocr_score.query_score
    return 0.0


def _stage1_ocr_reference(ocr_scores_by_id: dict[str, Stage1OcrScore]) -> float:
    for score in ocr_scores_by_id.values():
        return score.bm25_reference
    return 0.0


def _stage1_fused_scores(
    candidate_ids: Sequence[str],
    text_scores_by_id: dict[str, float] | None,
    ocr_scores_by_id: dict[str, float] | None,
    config: KisConfig,
) -> dict[str, float]:
    modality_percentiles: list[dict[str, RankedScore]] = []
    if text_scores_by_id is not None:
        modality_percentiles.append(rank_percentiles(text_scores_by_id, candidate_ids))
    if ocr_scores_by_id is not None:
        modality_percentiles.append(rank_percentiles(ocr_scores_by_id, candidate_ids))
    if not modality_percentiles:
        return {}
    if len(modality_percentiles) == 1:
        return {
            keyframe_id: modality_percentiles[0][keyframe_id].rank_percentile
            for keyframe_id in candidate_ids
        }
    return {
        keyframe_id: geometric_mean_score(
            [percentiles[keyframe_id].rank_percentile for percentiles in modality_percentiles],
            config,
        )
        for keyframe_id in candidate_ids
    }


def _candidate_metadata(
    keyframe_id: str,
    mongo_docs: dict[str, dict],
    vector_rows_by_id: dict[str, dict],
) -> dict | None:
    vector_row = vector_rows_by_id.get(keyframe_id)
    mongo_doc = mongo_docs.get(keyframe_id)
    if vector_row is not None:
        return {
            "video_id": vector_row["video_id"],
            "frame_idx": int(vector_row["frame_idx"]),
            "timestamp": float(vector_row["timestamp"]),
            "thumbnail_path": mongo_doc["thumbnail_path"] if mongo_doc else "",
        }
    if mongo_doc is None:
        return None
    return {
        "video_id": mongo_doc["video_id"],
        "frame_idx": int(mongo_doc["frame_idx"]),
        "timestamp": float(mongo_doc["timestamp"]),
        "thumbnail_path": mongo_doc["thumbnail_path"],
    }


def _candidate_metadata_by_id(
    candidate_ids: Sequence[str],
    mongo_docs: dict[str, dict],
    vector_rows_by_id: dict[str, dict],
) -> dict[str, dict]:
    metadata_by_id: dict[str, dict] = {}
    for keyframe_id in candidate_ids:
        metadata = _candidate_metadata(keyframe_id, mongo_docs, vector_rows_by_id)
        if metadata is None:
            logger.warning(
                "search_kis: dropping keyframe_id=%s -- no vector or MongoDB metadata",
                keyframe_id,
            )
            continue
        metadata_by_id[keyframe_id] = metadata
    return metadata_by_id


def _rank_text_queries(
    candidate_pool_ids: Sequence[str],
    text_queries: Sequence[KisQuery],
    text_vectors: np.ndarray,
    text_vector_index: dict[int, int],
    vector_rows_by_id: dict[str, dict],
) -> dict[int, dict[str, RankedScore]]:
    image_vectors_by_id = {
        keyframe_id: np.asarray(vector_row["embedding"], dtype=np.float32)
        for keyframe_id, vector_row in vector_rows_by_id.items()
        if keyframe_id in candidate_pool_ids
    }
    rankings: dict[int, dict[str, RankedScore]] = {}
    for query in text_queries:
        query_vector = np.asarray(
            text_vectors[text_vector_index[query.query_index]], dtype=np.float32
        )
        scores_by_id = {}
        for keyframe_id in candidate_pool_ids:
            image_vector = image_vectors_by_id.get(keyframe_id)
            scores_by_id[keyframe_id] = (
                _cosine_similarity(query_vector, image_vector)
                if image_vector is not None
                else _MISSING_TEXT_RAW_SCORE
            )
        rankings[query.query_index] = rank_percentiles(scores_by_id, candidate_pool_ids)
    return rankings


def _cosine_similarity(query_vector: np.ndarray, image_vector: np.ndarray) -> float:
    query_norm = float(np.linalg.norm(query_vector))
    image_norm = float(np.linalg.norm(image_vector))
    if query_norm <= 0 or image_norm <= 0:
        return _MISSING_TEXT_RAW_SCORE
    return float(np.dot(query_vector, image_vector) / (query_norm * image_norm))


def _build_text_modality_score(
    keyframe_id: str,
    text_queries: Sequence[KisQuery],
    text_rankings: dict[int, dict[str, RankedScore]],
    config: KisConfig,
) -> TextModalityScore:
    details = [
        TextQueryScore(
            query_index=query.query_index,
            query=query.text,
            raw_cosine=text_rankings[query.query_index][keyframe_id].raw_score,
            rank=text_rankings[query.query_index][keyframe_id].rank,
            rank_percentile=text_rankings[query.query_index][keyframe_id].rank_percentile,
        )
        for query in text_queries
    ]
    score = geometric_mean_score([detail.rank_percentile for detail in details], config)
    return TextModalityScore(geometric_mean=score, score=score, queries=details)


def _rank_ocr_queries(
    candidate_pool_ids: Sequence[str],
    ocr_queries: Sequence[KisQuery],
    ocr_texts_by_id: dict[str, str],
    bm25_by_query: dict[int, dict[str, float]],
    bm25_references: dict[int, float],
    config: KisConfig,
) -> tuple[dict[int, dict[str, _OcrQueryComponents]], dict[int, dict[str, RankedScore]]]:
    components_by_query: dict[int, dict[str, _OcrQueryComponents]] = {}
    rankings: dict[int, dict[str, RankedScore]] = {}
    for query in ocr_queries:
        components_by_id: dict[str, _OcrQueryComponents] = {}
        scores_by_id: dict[str, float] = {}
        reference = bm25_references.get(query.query_index, 0.0)
        query_bm25 = bm25_by_query.get(query.query_index, {})
        for keyframe_id in candidate_pool_ids:
            raw_bm25 = query_bm25.get(keyframe_id, 0.0)
            bm25_normalized = normalize_bm25(raw_bm25, reference)
            coverage = ocr_coverage(query.text, ocr_texts_by_id.get(keyframe_id, ""), config)
            query_score = clamp01(bm25_normalized * coverage)
            components_by_id[keyframe_id] = _OcrQueryComponents(
                raw_bm25=float(raw_bm25),
                bm25_reference=float(reference),
                bm25_normalized=float(bm25_normalized),
                coverage=float(coverage),
                query_score=float(query_score),
            )
            scores_by_id[keyframe_id] = float(query_score)
        components_by_query[query.query_index] = components_by_id
        rankings[query.query_index] = rank_percentiles(scores_by_id, candidate_pool_ids)
    return components_by_query, rankings


def _build_ocr_modality_score(
    keyframe_id: str,
    ocr_queries: Sequence[KisQuery],
    ocr_components_by_query: dict[int, dict[str, _OcrQueryComponents]],
    ocr_rankings: dict[int, dict[str, RankedScore]],
    config: KisConfig,
) -> OcrModalityScore:
    details = []
    for query in ocr_queries:
        components = ocr_components_by_query[query.query_index][keyframe_id]
        ranked = ocr_rankings[query.query_index][keyframe_id]
        details.append(
            OcrQueryScore(
                query_index=query.query_index,
                query=query.text,
                raw_bm25=components.raw_bm25,
                bm25_reference=components.bm25_reference,
                bm25_normalized=components.bm25_normalized,
                coverage=components.coverage,
                query_score=components.query_score,
                rank=ranked.rank,
                rank_percentile=ranked.rank_percentile,
            )
        )

    score = geometric_mean_score([detail.rank_percentile for detail in details], config)
    return OcrModalityScore(geometric_mean=score, score=score, queries=details)


def _build_abe_modality_score(
    keyframe_id: str,
    abe_queries: Sequence[KisQuery],
    abe_ranked_by_query: dict[int, AbeRankedQuery],
    config: KisConfig,
) -> AbeModalityScore:
    details: list[AbeQueryScore] = []
    for query in abe_queries:
        ranked_query = abe_ranked_by_query[query.query_index]
        components = ranked_query.components_by_id[keyframe_id]
        ranked = ranked_query.rankings[keyframe_id]
        details.append(
            AbeQueryScore(
                query_index=query.query_index,
                pairs=components.pairs,
                base_local=components.base_local,
                refined_local=components.refined_local,
                binding_delta=components.binding_delta,
                local_score=components.local_score,
                global_abe=components.global_abe,
                raw_score=components.raw_score,
                rank=ranked.rank,
                rank_percentile=ranked.rank_percentile,
                patch_heatmap=components.patch_heatmap,
            )
        )

    score = geometric_mean_score([detail.rank_percentile for detail in details], config)
    return AbeModalityScore(geometric_mean=score, score=score, queries=details)


def _reciprocal_rank_fusion(
    weighted_channels: Sequence[tuple[Sequence[SearchHit], float]],
    top_k: int,
    k: int = _DEFAULT_RRF_K,
) -> list[SearchHit]:
    """`RRF(d) = Σ_c weight_c / (k + rank_c(d))` (plan §P1.3). `rank_c` is the
    1-based rank of `d` within channel `c`'s own hit list; a document absent
    from a channel simply contributes 0 for it. Ties (equal fused score) keep
    first-seen order across `weighted_channels` for determinism -- `dict`
    preserves insertion order and `sorted()` is stable."""
    scores: dict[str, float] = {}
    representative: dict[str, SearchHit] = {}
    for hits, weight in weighted_channels:
        for rank, hit in enumerate(hits, start=1):
            scores[hit.keyframe_id] = scores.get(hit.keyframe_id, 0.0) + weight / (k + rank)
            representative.setdefault(hit.keyframe_id, hit)

    ranked_ids = sorted(representative, key=lambda keyframe_id: -scores[keyframe_id])
    fused = [
        SearchHit(
            keyframe_id=keyframe_id,
            video_id=representative[keyframe_id].video_id,
            score=scores[keyframe_id],
            frame_idx=representative[keyframe_id].frame_idx,
            timestamp=representative[keyframe_id].timestamp,
            thumbnail_path=representative[keyframe_id].thumbnail_path,
        )
        for keyframe_id in ranked_ids
    ]
    return fused[:top_k]
