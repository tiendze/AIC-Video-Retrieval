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
            settings={
                "analysis": {
                    "analyzer": {
                        _ANALYZER_NAME: {
                            "tokenizer": "vi_tokenizer",
                            "filter": ["lowercase", "ascii_folding"],
                        }
                    },
                    "filter": {
                        "ascii_folding": {"type": "asciifolding", "preserve_original": True}
                    },
                }
            },
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
