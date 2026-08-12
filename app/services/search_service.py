"""SearchService — the first end-to-end query bridge (plan §P0.8).

`mode` is a parameter of one function, not copy-pasted code paths (CLAUDE.md
§4): `semantic` embeds the query with the injected encoder and searches
Milvus; `ocr` full-text-searches Elasticsearch and must join to MongoDB
afterward because ES OCR hits carry no `frame_idx` at all (CLAUDE.md §3 only
requires `frame_idx` in Milvus/MongoDB).

`semantic` translates the query VI->EN before encoding when a
`QueryTranslator` is injected (plan §P0.9): BEiT-3's text tower is trained on
English pairs, so a Vietnamese query is off-distribution for it even though
the image tower is untouched -- confirmed 2026-08-11 by the `semantic@en`
eval channel (AIC Final Score 0.0451 -> 0.6208 on the same collection). `ocr`
never translates -- the ES index holds Vietnamese OCR text, so an English
query there would measure translation loss, not the retrieval channel.

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
use: fusing two semantic encoders of comparable quality (e.g. BEiT-3+envit5
⊕ SigLIP2, per docs/research-encoder-vietnamese.md §8.3) once a second
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
from typing import Literal

import numpy as np
from PIL import Image

from app.encoders.base import MultimodalEncoder
from app.services.query_translator import QueryTranslator
from app.stores.es_client import EsStore
from app.stores.milvus_client import MilvusStore
from app.stores.mongo_client import MongoStore

logger = logging.getLogger(__name__)

SearchMode = Literal["semantic", "ocr"]
_DEFAULT_RRF_K = 60  # plan §P1.3: RRF's own well-known literature default,
# kept only as _reciprocal_rank_fusion's own default -- no caller wires it
# today (module docstring: hybrid mode removed 2026-08-12).


class SearchServiceError(Exception):
    """Raised for a caller error, e.g. an unknown search mode."""


@dataclass(frozen=True)
class SearchHit:
    keyframe_id: str
    video_id: str
    score: float
    frame_idx: int
    timestamp: float
    thumbnail_path: str


class SearchService:
    def __init__(
        self,
        encoder: MultimodalEncoder,
        milvus_store: MilvusStore,
        es_store: EsStore,
        mongo_store: MongoStore,
        query_translator: QueryTranslator | None = None,
    ) -> None:
        self._encoder = encoder
        self._milvus_store = milvus_store
        self._es_store = es_store
        self._mongo_store = mongo_store
        self._query_translator = query_translator

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

    async def _search_semantic(self, query: str, top_k: int) -> list[SearchHit]:
        if self._query_translator is not None:
            query = await asyncio.to_thread(self._query_translator.translate, query)
        vector = await asyncio.to_thread(self._encoder.encode_text, [query])
        return await self._search_milvus_vector(vector[0], top_k)

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
        hits = await self._es_store.search(query, top_k)
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
