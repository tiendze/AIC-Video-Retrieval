"""RED reproducer for the R2/F5 deletion API added to app/stores/es_client.py
(research-asr-phowhisper-batched.plan.md T6) -- fakes `AsyncElasticsearch`
itself so this runs without Docker; see tests/integration/test_es_client.py
for the real-server counterpart.
"""

from __future__ import annotations

from app.stores.es_client import AsrEsStore


class _FakeIndicesClient:
    def __init__(self) -> None:
        self.exists_result = False
        self.create_calls: list[dict] = []
        self.delete_calls: list[dict] = []
        self.refresh_calls: list[str] = []

    async def exists(self, index):
        return self.exists_result

    async def create(self, index, settings, mappings):
        self.create_calls.append({"index": index, "settings": settings, "mappings": mappings})
        self.exists_result = True

    async def delete(self, index, ignore_unavailable=False):
        self.delete_calls.append({"index": index, "ignore_unavailable": ignore_unavailable})
        self.exists_result = False

    async def refresh(self, index):
        self.refresh_calls.append(index)


class _FakeAsyncElasticsearch:
    def __init__(self, *args, **kwargs) -> None:
        self.indices = _FakeIndicesClient()
        self.delete_by_query_calls: list[dict] = []

    async def delete_by_query(self, index, query):
        self.delete_by_query_calls.append({"index": index, "query": query})


def _store(monkeypatch) -> AsrEsStore:
    monkeypatch.setattr("app.stores.es_client.AsyncElasticsearch", _FakeAsyncElasticsearch)
    return AsrEsStore(url="http://unused", index_name="asr_segments")


async def test_delete_by_video_runs_delete_by_query_in_engine_not_client_side(monkeypatch):
    store = _store(monkeypatch)

    await store.delete_by_video("L21_V001")

    assert store._client.delete_by_query_calls == [
        {"index": "asr_segments", "query": {"term": {"video_id": "L21_V001"}}}
    ]


async def test_recreate_index_deletes_then_rebuilds_the_same_index_name(monkeypatch):
    store = _store(monkeypatch)

    await store.recreate_index()

    assert store._client.indices.delete_calls == [
        {"index": "asr_segments", "ignore_unavailable": True}
    ]
    assert len(store._client.indices.create_calls) == 1
    assert store._client.indices.create_calls[0]["index"] == "asr_segments"


async def test_recreate_index_leaves_the_index_usable_afterwards(monkeypatch):
    store = _store(monkeypatch)

    await store.recreate_index()

    # ensure_index() called again right after would be a no-op, not a second
    # create -- proves recreate_index() left the store in a normal "exists" state.
    await store.ensure_index()
    assert len(store._client.indices.create_calls) == 1
