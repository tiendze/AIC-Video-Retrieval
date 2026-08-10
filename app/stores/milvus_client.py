"""MilvusStore — the only layer that talks to Milvus directly (architecture.md §4.1).

Collection name embeds `model_id` so switching the embedding model can never
silently mix vectors from two different models into one collection — this is
enforced at the design level (CLAUDE.md §3), not by caller discipline.

Uses `MilvusClient` (pymilvus >=2.4), never the legacy ORM API.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from pymilvus import DataType, MilvusClient
from pymilvus.exceptions import MilvusException

_METRIC_TYPE = "COSINE"
_INDEX_TYPE = "HNSW"
_KEYFRAME_ID_MAX_LENGTH = 64
_VIDEO_ID_MAX_LENGTH = 32
_AUTHOR_MAX_LENGTH = 128
_BATCH_MAX_LENGTH = 8
_SEGMENT_ID_MAX_LENGTH = 64


@dataclass(frozen=True)
class KeyframeVector:
    """One row to upsert into Milvus. `embedding` must already be L2-normalized."""

    keyframe_id: str
    embedding: np.ndarray
    video_id: str
    global_index: int
    frame_idx: int
    timestamp: float
    author: str
    batch: str


class MilvusStore:
    """Vector store for keyframe embeddings, one collection per `model_id`."""

    def __init__(
        self,
        uri: str,
        token: str,
        model_id: str,
        embedding_dim: int,
        hnsw_m: int,
        hnsw_ef_construction: int,
        search_ef: int,
    ) -> None:
        self._client = MilvusClient(uri=uri, token=token)
        self._collection_name = f"keyframes_{model_id}"
        self._search_ef = search_ef
        self._embedding_dim = embedding_dim
        self._hnsw_m = hnsw_m
        self._hnsw_ef_construction = hnsw_ef_construction
        self._ensure_collection(embedding_dim, hnsw_m, hnsw_ef_construction)

    @property
    def collection_name(self) -> str:
        return self._collection_name

    def ping(self) -> bool:
        """Connectivity check for `/health` (plan §P0.8). A store being
        unreachable is expected degraded-mode data (CLAUDE.md §4), not a
        bug -- returns False instead of letting `MilvusException` propagate."""
        try:
            self._client.list_collections()
            return True
        except MilvusException:
            return False

    def drop_collection(self) -> None:
        """Deletes the collection outright (not a row-level delete) — used to
        wipe a keyframe source that's being replaced wholesale, so its rows
        can never mix with the next source's under the same `model_id`
        (CLAUDE.md §3, plan §9.5). Re-creates an empty collection immediately
        so callers can keep using this store without re-instantiating it."""
        if self._client.has_collection(self._collection_name):
            self._client.drop_collection(self._collection_name)
        self._ensure_collection(self._embedding_dim, self._hnsw_m, self._hnsw_ef_construction)

    def _ensure_collection(
        self, embedding_dim: int, hnsw_m: int, hnsw_ef_construction: int
    ) -> None:
        if self._client.has_collection(self._collection_name):
            return

        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(
            field_name="keyframe_id",
            datatype=DataType.VARCHAR,
            is_primary=True,
            max_length=_KEYFRAME_ID_MAX_LENGTH,
        )
        schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=embedding_dim)
        schema.add_field(
            field_name="video_id", datatype=DataType.VARCHAR, max_length=_VIDEO_ID_MAX_LENGTH
        )
        schema.add_field(field_name="global_index", datatype=DataType.INT64)
        # The competition answer format is `<video_id>, <frame_id>` and TRAKE
        # accepts windows often narrower than 10 frames, so the absolute frame
        # index has to travel with the search hit — deriving it from
        # `timestamp * fps` at submission time would round into the wrong frame.
        schema.add_field(field_name="frame_idx", datatype=DataType.INT64)
        schema.add_field(field_name="timestamp", datatype=DataType.FLOAT)
        schema.add_field(
            field_name="author", datatype=DataType.VARCHAR, max_length=_AUTHOR_MAX_LENGTH
        )
        schema.add_field(
            field_name="batch", datatype=DataType.VARCHAR, max_length=_BATCH_MAX_LENGTH
        )

        index_params = self._client.prepare_index_params()
        index_params.add_index(field_name="keyframe_id", index_type="AUTOINDEX")
        index_params.add_index(
            field_name="embedding",
            index_type=_INDEX_TYPE,
            metric_type=_METRIC_TYPE,
            params={"M": hnsw_m, "efConstruction": hnsw_ef_construction},
        )

        self._client.create_collection(
            collection_name=self._collection_name, schema=schema, index_params=index_params
        )

    def upsert(self, vectors: Sequence[KeyframeVector]) -> None:
        """Insert-or-replace by `keyframe_id` — re-running ingest never duplicates rows."""
        if not vectors:
            return
        rows = [
            {
                "keyframe_id": v.keyframe_id,
                "embedding": v.embedding.tolist(),
                "video_id": v.video_id,
                "global_index": v.global_index,
                "frame_idx": v.frame_idx,
                "timestamp": v.timestamp,
                "author": v.author,
                "batch": v.batch,
            }
            for v in vectors
        ]
        self._client.upsert(collection_name=self._collection_name, data=rows)

    def search(self, query_vector: np.ndarray, top_k: int, filter_expr: str = "") -> list[dict]:
        """ANN search. `filter_expr` runs inside Milvus — never filter results after the fact.

        `consistency_level="Strong"`: without it Milvus's default "Bounded"
        consistency can make a search miss a write from moments earlier
        (this store's own idempotent-upsert invariant depends on searches
        reliably seeing prior writes, not just eventually).
        """
        results = self._client.search(
            collection_name=self._collection_name,
            anns_field="embedding",
            data=[query_vector.tolist()],
            limit=top_k,
            search_params={"params": {"ef": self._search_ef}},
            filter=filter_expr,
            output_fields=[
                "video_id",
                "global_index",
                "frame_idx",
                "timestamp",
                "author",
                "batch",
            ],
            consistency_level="Strong",
        )
        return results[0]


@dataclass(frozen=True)
class TranscriptVector:
    """One row to upsert into the ASR transcript collection (P0.11b).
    `embedding` must already be L2-normalized. `start_frame_idx`/
    `end_frame_idx` are the same retrieval window computed in
    `pipelines/asr.py` — never the BTC submission `frame_id`."""

    segment_id: str
    embedding: np.ndarray
    video_id: str
    start_frame_idx: int
    end_frame_idx: int
    start_time: float
    end_time: float


class TranscriptMilvusStore:
    """Vector store for ASR transcript embeddings (P0.11b), one collection
    per `model_id` (e.g. `transcripts_bge_m3`) — deliberately separate from
    `MilvusStore`: the schemas differ (no `frame_idx`/`author`/`batch`; a
    frame *window* instead), and this is a structurally different channel
    (text-to-text, not image-to-text)."""

    def __init__(
        self,
        uri: str,
        token: str,
        model_id: str,
        embedding_dim: int,
        hnsw_m: int,
        hnsw_ef_construction: int,
        search_ef: int,
    ) -> None:
        self._client = MilvusClient(uri=uri, token=token)
        self._collection_name = f"transcripts_{model_id}"
        self._search_ef = search_ef
        self._embedding_dim = embedding_dim
        self._hnsw_m = hnsw_m
        self._hnsw_ef_construction = hnsw_ef_construction
        self._ensure_collection(embedding_dim, hnsw_m, hnsw_ef_construction)

    @property
    def collection_name(self) -> str:
        return self._collection_name

    def drop_collection(self) -> None:
        """Deletes the collection outright (not a row-level delete) — mirrors
        `MilvusStore.drop_collection()`. Used before re-embedding transcripts
        produced by a different ASR `model_id` (research-asr-phowhisper-
        batched.plan.md F5/R2), so segment vectors from the old model can
        never mix with the new model's under the same collection. Re-creates
        an empty collection immediately so callers can keep using this store
        without re-instantiating it."""
        if self._client.has_collection(self._collection_name):
            self._client.drop_collection(self._collection_name)
        self._ensure_collection(self._embedding_dim, self._hnsw_m, self._hnsw_ef_construction)

    def _ensure_collection(
        self, embedding_dim: int, hnsw_m: int, hnsw_ef_construction: int
    ) -> None:
        if self._client.has_collection(self._collection_name):
            return

        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(
            field_name="segment_id",
            datatype=DataType.VARCHAR,
            is_primary=True,
            max_length=_SEGMENT_ID_MAX_LENGTH,
        )
        schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=embedding_dim)
        schema.add_field(
            field_name="video_id", datatype=DataType.VARCHAR, max_length=_VIDEO_ID_MAX_LENGTH
        )
        schema.add_field(field_name="start_frame_idx", datatype=DataType.INT64)
        schema.add_field(field_name="end_frame_idx", datatype=DataType.INT64)
        schema.add_field(field_name="start_time", datatype=DataType.FLOAT)
        schema.add_field(field_name="end_time", datatype=DataType.FLOAT)

        index_params = self._client.prepare_index_params()
        index_params.add_index(field_name="segment_id", index_type="AUTOINDEX")
        index_params.add_index(
            field_name="embedding",
            index_type=_INDEX_TYPE,
            metric_type=_METRIC_TYPE,
            params={"M": hnsw_m, "efConstruction": hnsw_ef_construction},
        )

        self._client.create_collection(
            collection_name=self._collection_name, schema=schema, index_params=index_params
        )

    def upsert(self, vectors: Sequence[TranscriptVector]) -> None:
        """Insert-or-replace by `segment_id` — re-running embedding never duplicates rows."""
        if not vectors:
            return
        rows = [
            {
                "segment_id": v.segment_id,
                "embedding": v.embedding.tolist(),
                "video_id": v.video_id,
                "start_frame_idx": v.start_frame_idx,
                "end_frame_idx": v.end_frame_idx,
                "start_time": v.start_time,
                "end_time": v.end_time,
            }
            for v in vectors
        ]
        self._client.upsert(collection_name=self._collection_name, data=rows)

    def search(self, query_vector: np.ndarray, top_k: int, filter_expr: str = "") -> list[dict]:
        """ANN search. See `MilvusStore.search` for the `consistency_level` rationale."""
        results = self._client.search(
            collection_name=self._collection_name,
            anns_field="embedding",
            data=[query_vector.tolist()],
            limit=top_k,
            search_params={"params": {"ef": self._search_ef}},
            filter=filter_expr,
            output_fields=[
                "video_id",
                "start_frame_idx",
                "end_frame_idx",
                "start_time",
                "end_time",
            ],
            consistency_level="Strong",
        )
        return results[0]
