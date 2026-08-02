"""EsStore — the only layer that talks to Elasticsearch directly (architecture.md §4.2).

Mapping uses the `vi_analyzer` plugin (Vietnamese word-segmentation, not
syllable-based) with `asciifolding`/`preserve_original` so a query typed
without diacritics still matches — a common shortcut under contest time
pressure (references.md §4.3). Bulk indexing goes through `helpers.async_bulk`,
never one `index()` call per document.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from elasticsearch import AsyncElasticsearch
from elasticsearch.helpers import async_bulk

_ANALYZER_NAME = "vi_analyzer"

# Shared between EsStore (OCR) and AsrEsStore (ASR transcripts, P0.11a) --
# both indexes need the same Vietnamese word-segmentation + no-diacritics
# fallback (references.md §4.3); only the mappings differ per document shape.
_VI_ANALYSIS_SETTINGS = {
    "analysis": {
        "analyzer": {
            _ANALYZER_NAME: {
                "tokenizer": "vi_tokenizer",
                "filter": ["lowercase", "ascii_folding"],
            }
        },
        "filter": {"ascii_folding": {"type": "asciifolding", "preserve_original": True}},
    }
}


@dataclass(frozen=True)
class OcrDocument:
    keyframe_id: str
    ocr_text: str
    video_id: str
    timestamp: float


class EsStore:
    """OCR text store — one index for all keyframes' recognized text."""

    def __init__(self, url: str, index_name: str, api_key: str | None = None) -> None:
        self._client = AsyncElasticsearch(hosts=[url], api_key=api_key)
        self._index_name = index_name

    @property
    def index_name(self) -> str:
        return self._index_name

    async def ensure_index(self) -> None:
        if await self._client.indices.exists(index=self._index_name):
            return
        await self._client.indices.create(
            index=self._index_name,
            settings=_VI_ANALYSIS_SETTINGS,
            mappings={
                "properties": {
                    "keyframe_id": {"type": "keyword"},
                    "ocr_text": {"type": "text", "analyzer": _ANALYZER_NAME},
                    "video_id": {"type": "keyword"},
                    "timestamp": {"type": "float"},
                }
            },
        )

    async def bulk_index(self, docs: Sequence[OcrDocument]) -> None:
        if not docs:
            return
        actions = (
            {
                "_index": self._index_name,
                "_id": doc.keyframe_id,
                "_source": {
                    "keyframe_id": doc.keyframe_id,
                    "ocr_text": doc.ocr_text,
                    "video_id": doc.video_id,
                    "timestamp": doc.timestamp,
                },
            }
            for doc in docs
        )
        await async_bulk(self._client, actions)

    async def search(self, query_text: str, top_k: int, video_id: str | None = None) -> list[dict]:
        """Full-text OCR search. `video_id` filter runs inside ES, never after the fact."""
        query: dict = {"match": {"ocr_text": query_text}}
        if video_id is not None:
            query = {"bool": {"must": [query], "filter": [{"term": {"video_id": video_id}}]}}

        response = await self._client.search(index=self._index_name, query=query, size=top_k)
        return [hit["_source"] | {"score": hit["_score"]} for hit in response["hits"]["hits"]]

    async def close(self) -> None:
        await self._client.close()


@dataclass(frozen=True)
class AsrSegmentDocument:
    """One spoken-dialogue segment (P0.11a) — indexed at SEGMENT
    granularity, never forced onto keyframe boundaries (a sentence can span
    several shots). `start_frame_idx`/`end_frame_idx` are a *retrieval
    window* derived via floor/ceil from `start_time`/`end_time` — never the
    BTC submission `frame_id`, which always comes from a keyframe's own
    `frame_idx` (CLAUDE.md T2)."""

    segment_id: str
    video_id: str
    start_time: float
    end_time: float
    text: str
    start_frame_idx: int
    end_frame_idx: int


class AsrEsStore:
    """ASR transcript store — deliberately separate from `EsStore`: that
    class hardcodes the `ocr_text`/`keyframe_id` mapping and can't be reused
    for segment-granularity ASR documents carrying a frame *window* instead
    of a single `frame_idx`."""

    def __init__(self, url: str, index_name: str, api_key: str | None = None) -> None:
        self._client = AsyncElasticsearch(hosts=[url], api_key=api_key)
        self._index_name = index_name

    @property
    def index_name(self) -> str:
        return self._index_name

    async def ensure_index(self) -> None:
        if await self._client.indices.exists(index=self._index_name):
            return
        await self._client.indices.create(
            index=self._index_name,
            settings=_VI_ANALYSIS_SETTINGS,
            mappings={
                "properties": {
                    "segment_id": {"type": "keyword"},
                    "video_id": {"type": "keyword"},
                    "start_time": {"type": "float"},
                    "end_time": {"type": "float"},
                    "text": {"type": "text", "analyzer": _ANALYZER_NAME},
                    "start_frame_idx": {"type": "integer"},
                    "end_frame_idx": {"type": "integer"},
                }
            },
        )

    async def bulk_index(self, docs: Sequence[AsrSegmentDocument]) -> None:
        if not docs:
            return
        actions = (
            {
                "_index": self._index_name,
                "_id": doc.segment_id,
                "_source": {
                    "segment_id": doc.segment_id,
                    "video_id": doc.video_id,
                    "start_time": doc.start_time,
                    "end_time": doc.end_time,
                    "text": doc.text,
                    "start_frame_idx": doc.start_frame_idx,
                    "end_frame_idx": doc.end_frame_idx,
                },
            }
            for doc in docs
        )
        await async_bulk(self._client, actions)

    async def search(self, query_text: str, top_k: int, video_id: str | None = None) -> list[dict]:
        """Full-text ASR search. `video_id` filter runs inside ES, never after the fact."""
        query: dict = {"match": {"text": query_text}}
        if video_id is not None:
            query = {"bool": {"must": [query], "filter": [{"term": {"video_id": video_id}}]}}

        response = await self._client.search(index=self._index_name, query=query, size=top_k)
        return [hit["_source"] | {"score": hit["_score"]} for hit in response["hits"]["hits"]]

    async def close(self) -> None:
        await self._client.close()
