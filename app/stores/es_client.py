"""EsStore — the only layer that talks to Elasticsearch directly (architecture.md §4.2).

Mapping uses the `vi_analyzer` plugin (Vietnamese word-segmentation, not
syllable-based) with `asciifolding`/`preserve_original` so a query typed
without diacritics still matches — a common shortcut under contest time
pressure (references.md §4.3). Bulk indexing goes through `helpers.async_bulk`,
never one `index()` call per document.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any

from elasticsearch import AsyncElasticsearch
from elasticsearch.helpers import async_bulk, async_scan

_ANALYZER_NAME = "vi_analyzer"
_OCR_FUZZINESS = 1
_OCR_FUZZY_PREFIX_LENGTH = 1
_OCR_FUZZY_MAX_EXPANSIONS = 50

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

    async def ping(self) -> bool:
        """Connectivity check for `/health` (plan §P0.8). `AsyncElasticsearch.ping()`
        already catches transport/API errors itself and returns False --
        nothing to wrap here (CLAUDE.md §4 degraded mode: unreachable is
        expected data, not a bug)."""
        return await self._client.ping()

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

    async def get_by_ids(self, keyframe_ids: Sequence[str]) -> dict[str, str]:
        """Batch OCR-text lookup by exact `keyframe_id`, via `mget` -- used
        by `eval/metrics/leakage.py` (plan §P0.9), which needs the OCR text
        of specific answer keyframes, not a `search()` full-text match. A
        missing id is silently omitted, mirroring
        `MongoStore.get_keyframes_by_ids`'s convention: no OCR text for a
        keyframe is expected data (OCR ingest is still partial), not an
        error."""
        if not keyframe_ids:
            return {}
        response = await self._client.mget(index=self._index_name, ids=list(keyframe_ids))
        return {
            doc["_id"]: doc["_source"]["ocr_text"] for doc in response["docs"] if doc.get("found")
        }

    async def search(
        self,
        query_text: str,
        top_k: int,
        video_ids: Sequence[str] | None = None,
    ) -> list[dict]:
        """Full-text OCR search. Video filters run inside ES, never after the fact."""
        query = _ocr_match_query(query_text)
        if video_ids is not None:
            unique_video_ids = list(dict.fromkeys(video_ids))
            if not unique_video_ids:
                return []
            query = {
                "bool": {
                    "must": [query],
                    "filter": [{"terms": {"video_id": unique_video_ids}}],
                }
            }

        response = await self._client.search(index=self._index_name, query=query, size=top_k)
        return [hit["_source"] | {"score": hit["_score"]} for hit in response["hits"]["hits"]]

    async def score_by_ids(self, query_text: str, keyframe_ids: Sequence[str]) -> dict[str, float]:
        """BM25 scores for one OCR query against an existing candidate pool.

        Elasticsearch returns only candidates that match the `match` query.
        Callers assign raw score 0 to every requested id omitted here.
        """
        if not keyframe_ids:
            return {}
        unique_ids = list(dict.fromkeys(keyframe_ids))
        query = {
            "bool": {
                "must": [_ocr_match_query(query_text)],
                "filter": [{"ids": {"values": unique_ids}}],
            }
        }
        response = await self._client.search(
            index=self._index_name,
            query=query,
            size=len(unique_ids),
            source_includes=["keyframe_id"],
        )
        return {
            hit["_source"]["keyframe_id"]: float(hit["_score"]) for hit in response["hits"]["hits"]
        }

    async def close(self) -> None:
        await self._client.close()


def _ocr_match_query(query_text: str) -> dict[str, dict[str, dict[str, object]]]:
    """OCR candidate generation tolerates one-character OCR mistakes.

    The KIS coverage step still rescored matched candidates afterward, so
    this fuzzy query only widens the pool enough for cases like OpenAI/OpenAl.
    """
    return {
        "match": {
            "ocr_text": {
                "query": query_text,
                "fuzziness": _OCR_FUZZINESS,
                "prefix_length": _OCR_FUZZY_PREFIX_LENGTH,
                "max_expansions": _OCR_FUZZY_MAX_EXPANSIONS,
            }
        }
    }


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


@dataclass(frozen=True)
class AsrSegmentSearchHit:
    segment_id: str
    video_id: str
    text: str
    start_time: float
    end_time: float
    score: float


@dataclass(frozen=True)
class AsrSegmentSearchResult:
    total_hits: int
    hits: list[AsrSegmentSearchHit]


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

    async def search(self, query_text: str, top_k: int) -> AsrSegmentSearchResult:
        """Retrieve BM25-ranked ASR transcript segments with an exact hit count."""
        response = await self._client.search(
            index=self._index_name,
            query={"match": {"text": query_text}},
            size=top_k,
            track_total_hits=True,
        )
        return AsrSegmentSearchResult(
            total_hits=int(response["hits"]["total"]["value"]),
            hits=[
                AsrSegmentSearchHit(
                    segment_id=hit["_source"]["segment_id"],
                    video_id=hit["_source"]["video_id"],
                    text=hit["_source"]["text"],
                    start_time=float(hit["_source"]["start_time"]),
                    end_time=float(hit["_source"]["end_time"]),
                    score=float(hit["_score"]),
                )
                for hit in response["hits"]["hits"]
            ],
        )

    async def refresh(self) -> None:
        """Makes just-indexed segments visible to search/scroll immediately.
        Elasticsearch is near-real-time (~1s refresh interval by default), so a
        `scroll_all()` fired right after `bulk_index()` would silently miss the
        last video's segments — an export that looks complete but isn't."""
        await self._client.indices.refresh(index=self._index_name)

    async def delete_by_video(self, video_id: str) -> None:
        """Deletes every segment for one video, in-engine (`delete_by_query`)
        — never fetch-then-filter-then-delete client-side (CLAUDE.md §4)."""
        await self._client.delete_by_query(
            index=self._index_name, query={"term": {"video_id": video_id}}
        )

    async def recreate_index(self) -> None:
        """Drops the whole index and rebuilds it empty — used before
        re-running ASR under a different `model_id` (research-asr-
        phowhisper-batched.plan.md F5/R2), so segments from the old model can
        never linger and get scrolled back into the same JSON export
        alongside the new model's segments. `ignore_unavailable=True`: a
        first-ever run has no index yet, and that isn't an error here."""
        await self._client.indices.delete(index=self._index_name, ignore_unavailable=True)
        await self.ensure_index()

    async def scroll_all(self) -> AsyncIterator[dict[str, Any]]:
        """Every indexed segment, across all videos — P0.11b reads
        `transcript_text` back through here to embed, instead of re-running
        Whisper. Uses `helpers.async_scan` (the scroll API), not `search()`
        with a large `size`, since the segment count can exceed any single
        page Elasticsearch is willing to return."""
        query = {"query": {"match_all": {}}}
        async for hit in async_scan(self._client, index=self._index_name, query=query):
            yield hit["_source"]

    async def close(self) -> None:
        await self._client.close()
