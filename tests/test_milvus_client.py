"""RED reproducer for TranscriptMilvusStore.drop_collection() (F5 in
research-asr-phowhisper-batched.plan.md) -- fakes `MilvusClient` itself so
this runs without Docker; see tests/integration/test_milvus_client.py for
the real-server counterpart.
"""

from __future__ import annotations

import numpy as np
from pymilvus.exceptions import MilvusException

from app.stores.milvus_client import MilvusStore, TranscriptMilvusStore


class _FakeSchema:
    def __init__(self, **kwargs) -> None:
        self.fields: list[dict] = []

    def add_field(self, **kwargs):
        self.fields.append(kwargs)


class _FakeIndexParams:
    def __init__(self) -> None:
        self.indexes: list[dict] = []

    def add_index(self, **kwargs):
        self.indexes.append(kwargs)


class _FakeMilvusClient:
    """Stands in for pymilvus.MilvusClient. `create_schema` is called on the
    class itself (a classmethod in the real SDK) -- verified via
    `inspect.signature(MilvusClient.create_schema)` before writing this fake."""

    def __init__(self, uri, token) -> None:
        self.uri = uri
        self.token = token
        self.collections: set[str] = set()
        self.drop_collection_calls: list[str] = []
        self.create_collection_calls: list[str] = []
        self.raise_on_list_collections = False
        self.search_calls: list[dict] = []

    def search(self, **kwargs):
        self.search_calls.append(kwargs)
        return [[]]

    @classmethod
    def create_schema(cls, **kwargs):
        return _FakeSchema(**kwargs)

    def prepare_index_params(self):
        return _FakeIndexParams()

    def has_collection(self, collection_name):
        return collection_name in self.collections

    def create_collection(self, collection_name, schema, index_params):
        self.collections.add(collection_name)
        self.create_collection_calls.append(collection_name)

    def drop_collection(self, collection_name):
        self.collections.discard(collection_name)
        self.drop_collection_calls.append(collection_name)

    def list_collections(self):
        if self.raise_on_list_collections:
            raise MilvusException("connection refused")
        return list(self.collections)


def _store(monkeypatch) -> TranscriptMilvusStore:
    monkeypatch.setattr("app.stores.milvus_client.MilvusClient", _FakeMilvusClient)
    return TranscriptMilvusStore(
        uri="http://unused",
        token="unused",
        model_id="bge_m3",
        embedding_dim=1024,
        hnsw_m=16,
        hnsw_ef_construction=200,
        search_ef=64,
    )


def test_collection_name_embeds_model_id(monkeypatch):
    store = _store(monkeypatch)

    assert store.collection_name == "transcripts_bge_m3"


def test_transcript_search_clamps_ef_to_top_k_when_configured_ef_is_lower(monkeypatch):
    """Same HNSW ef>=k invariant as MilvusStore.search() -- TranscriptMilvusStore
    has a structurally identical search() and must not reproduce the same crash."""
    store = _store(monkeypatch)  # search_ef=64

    store.search(np.zeros(1024, dtype=np.float32), top_k=100)

    sent_ef = store._client.search_calls[0]["search_params"]["params"]["ef"]
    assert sent_ef == 100


def test_drop_collection_drops_then_recreates_the_same_collection_name(monkeypatch):
    store = _store(monkeypatch)
    assert store._client.has_collection("transcripts_bge_m3") is True  # created in __init__

    store.drop_collection()

    assert store._client.drop_collection_calls == ["transcripts_bge_m3"]
    # re-created immediately -- caller can keep using this store instance
    assert store._client.has_collection("transcripts_bge_m3") is True
    assert store._client.create_collection_calls == ["transcripts_bge_m3", "transcripts_bge_m3"]


def test_drop_collection_is_a_no_op_on_the_underlying_drop_call_when_collection_absent(
    monkeypatch,
):
    store = _store(monkeypatch)
    store._client.collections.discard("transcripts_bge_m3")  # simulate already gone

    store.drop_collection()

    assert store._client.drop_collection_calls == []
    assert store._client.has_collection("transcripts_bge_m3") is True


def _keyframe_store(monkeypatch) -> MilvusStore:
    monkeypatch.setattr("app.stores.milvus_client.MilvusClient", _FakeMilvusClient)
    return MilvusStore(
        uri="http://unused",
        token="unused",
        model_id="beit3_base_itc",
        embedding_dim=768,
        hnsw_m=16,
        hnsw_ef_construction=200,
        search_ef=64,
    )


def test_ping_returns_true_when_the_connection_is_healthy(monkeypatch):
    store = _keyframe_store(monkeypatch)

    assert store.ping() is True


def test_ping_returns_false_instead_of_raising_when_milvus_is_unreachable(monkeypatch):
    store = _keyframe_store(monkeypatch)
    store._client.raise_on_list_collections = True

    assert store.ping() is False


def test_search_clamps_ef_to_top_k_when_configured_ef_is_lower(monkeypatch):
    """Milvus HNSW rejects any search where ef < k (top_k) -- eval/run.py
    hit this in production requesting top_k=100 against search_ef=64
    (MilvusException 'ef(64) should be larger than k(100)'). The store must
    never send a search that violates this invariant, regardless of what
    search_ef the config was set to."""
    store = _keyframe_store(monkeypatch)  # search_ef=64

    store.search(np.zeros(768, dtype=np.float32), top_k=100)

    sent_ef = store._client.search_calls[0]["search_params"]["params"]["ef"]
    assert sent_ef == 100


def test_search_keeps_configured_ef_when_it_already_covers_top_k(monkeypatch):
    store = _keyframe_store(monkeypatch)  # search_ef=64

    store.search(np.zeros(768, dtype=np.float32), top_k=10)

    sent_ef = store._client.search_calls[0]["search_params"]["params"]["ef"]
    assert sent_ef == 64
