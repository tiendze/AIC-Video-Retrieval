"""RED reproducer for the R2/F5 deletion API added to app/stores/es_client.py
(research-asr-phowhisper-batched.plan.md T6) -- fakes `AsyncElasticsearch`
itself so this runs without Docker; see tests/integration/test_es_client.py
for the real-server counterpart.
"""

from __future__ import annotations

from app.stores.es_client import AsrEsStore, EsStore


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
        self.search_calls: list[dict] = []
        self.search_hits: list[dict] = []
        self.ping_result = True
        self.mget_docs: dict[str, dict] = {}

    async def delete_by_query(self, index, query):
        self.delete_by_query_calls.append({"index": index, "query": query})

    async def ping(self):
        return self.ping_result

    async def mget(self, index, ids):
        return {
            "docs": [
                (
                    {"_id": doc_id, "found": True, "_source": self.mget_docs[doc_id]}
                    if doc_id in self.mget_docs
                    else {"_id": doc_id, "found": False}
                )
                for doc_id in ids
            ]
        }

    async def search(self, **kwargs):
        self.search_calls.append(kwargs)
        return {"hits": {"hits": self.search_hits}}


def _store(monkeypatch) -> AsrEsStore:
    monkeypatch.setattr("app.stores.es_client.AsyncElasticsearch", _FakeAsyncElasticsearch)
    return AsrEsStore(url="http://unused", index_name="asr_segments")


def _ocr_store(monkeypatch) -> EsStore:
    monkeypatch.setattr("app.stores.es_client.AsyncElasticsearch", _FakeAsyncElasticsearch)
    return EsStore(url="http://unused", index_name="ocr_keyframes")


async def test_ping_returns_true_when_the_connection_is_healthy(monkeypatch):
    store = _ocr_store(monkeypatch)

    assert await store.ping() is True


async def test_ping_returns_false_when_elasticsearch_is_unreachable(monkeypatch):
    store = _ocr_store(monkeypatch)
    store._client.ping_result = False

    assert await store.ping() is False


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


async def test_get_by_ids_returns_ocr_text_keyed_by_keyframe_id(monkeypatch):
    # eval/metrics/leakage.py (plan §P0.9) needs the OCR text of specific
    # answer keyframes, not a full-text search() match.
    store = _ocr_store(monkeypatch)
    store._client.mget_docs = {
        "L21_V001_10": {"ocr_text": "tin tuc", "video_id": "L21_V001", "timestamp": 0.4},
    }

    result = await store.get_by_ids(["L21_V001_10"])

    assert result == {"L21_V001_10": "tin tuc"}


async def test_get_by_ids_omits_ids_not_found_in_elasticsearch(monkeypatch):
    # No OCR text for a keyframe is expected data (OCR ingest is still
    # partial, plan §P0.7) -- omitted, not raised.
    store = _ocr_store(monkeypatch)
    store._client.mget_docs = {"found": {"ocr_text": "x", "video_id": "v", "timestamp": 0.0}}

    result = await store.get_by_ids(["found", "missing"])

    assert set(result.keys()) == {"found"}


async def test_get_by_ids_returns_empty_dict_for_empty_input(monkeypatch):
    store = _ocr_store(monkeypatch)

    assert await store.get_by_ids([]) == {}


async def test_ocr_search_uses_fuzzy_match_for_candidate_generation(monkeypatch):
    store = _ocr_store(monkeypatch)

    await store.search("OpenAI", top_k=5)

    assert store._client.search_calls == [
        {
            "index": "ocr_keyframes",
            "query": {
                "match": {
                    "ocr_text": {
                        "query": "OpenAI",
                        "fuzziness": 1,
                        "prefix_length": 1,
                        "max_expansions": 50,
                    }
                }
            },
            "size": 5,
        }
    ]


async def test_score_by_ids_scores_one_ocr_query_inside_candidate_pool(monkeypatch):
    store = _ocr_store(monkeypatch)
    store._client.search_hits = [
        {"_source": {"keyframe_id": "A"}, "_score": 12.5},
        {"_source": {"keyframe_id": "B"}, "_score": 3.0},
    ]

    result = await store.score_by_ids("VINAMILK", ["A", "B", "A"])

    assert result == {"A": 12.5, "B": 3.0}
    assert store._client.search_calls == [
        {
            "index": "ocr_keyframes",
            "query": {
                "bool": {
                    "must": [
                        {
                            "match": {
                                "ocr_text": {
                                    "query": "VINAMILK",
                                    "fuzziness": 1,
                                    "prefix_length": 1,
                                    "max_expansions": 50,
                                }
                            }
                        }
                    ],
                    "filter": [{"ids": {"values": ["A", "B"]}}],
                }
            },
            "size": 2,
            "source_includes": ["keyframe_id"],
        }
    ]


async def test_score_by_ids_returns_empty_dict_for_empty_input(monkeypatch):
    store = _ocr_store(monkeypatch)

    assert await store.score_by_ids("VINAMILK", []) == {}
    assert store._client.search_calls == []
