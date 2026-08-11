"""SearchService — the first end-to-end query bridge (plan §P0.8).

`mode` is a parameter of one function, not copy-pasted code paths (CLAUDE.md
§4): `semantic` embeds the query with the injected encoder and searches
Milvus; `ocr` full-text-searches Elasticsearch and must join to MongoDB
afterward because ES OCR hits carry no `frame_idx` at all (CLAUDE.md §3 only
requires `frame_idx` in Milvus/MongoDB). `hybrid` (RRF) is P1.3 — it needs
both channels measured independently first.

`semantic` translates the query VI->EN before encoding when a
`QueryTranslator` is injected (plan §P0.9): BEiT-3's text tower is trained on
English pairs, so a Vietnamese query is off-distribution for it even though
the image tower is untouched -- confirmed 2026-08-11 by the `semantic@en`
eval channel (AIC Final Score 0.0451 -> 0.6208 on the same collection). `ocr`
never translates -- the ES index holds Vietnamese OCR text, so an English
query there would measure translation loss, not the retrieval channel.

Milvus search, encoder inference, and translation are synchronous, CPU/GPU-
bound calls; they run off the event loop via `asyncio.to_thread` so this
async method never blocks other requests (.claude/rules/ecc/python/fastapi.md).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Literal

from app.encoders.base import MultimodalEncoder
from app.services.query_translator import QueryTranslator
from app.stores.es_client import EsStore
from app.stores.milvus_client import MilvusStore
from app.stores.mongo_client import MongoStore

logger = logging.getLogger(__name__)

SearchMode = Literal["semantic", "ocr"]


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

    async def _search_semantic(self, query: str, top_k: int) -> list[SearchHit]:
        if self._query_translator is not None:
            query = await asyncio.to_thread(self._query_translator.translate, query)
        vector = await asyncio.to_thread(self._encoder.encode_text, [query])
        hits = await asyncio.to_thread(self._milvus_store.search, vector[0], top_k)
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
                    "search_semantic: keyframe_id=%s has no MongoDB record, "
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
