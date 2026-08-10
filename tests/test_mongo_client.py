"""RED reproducer for MongoStore.get_keyframes_by_ids (P0.8) -- fakes
`AsyncMongoClient` itself so this runs without Docker; see
tests/integration/test_mongo_client.py for the real-server counterpart.

get_keyframes_by_ids is the join SearchService (app/services/search_service.py)
uses to attach frame_idx/thumbnail_path onto search hits: Milvus already
carries frame_idx for semantic hits, but Elasticsearch OCR hits don't (CLAUDE.md
§3 only requires frame_idx in Milvus/MongoDB), so OCR-mode results need this
lookup to get a submittable frame_idx at all.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pymongo.errors import ServerSelectionTimeoutError

from app.stores.mongo_client import MongoStore


class _FakeCursor:
    def __init__(self, docs: Sequence[dict[str, Any]]) -> None:
        self._docs = list(docs)

    def __aiter__(self):
        return self._iter()

    async def _iter(self):
        for doc in self._docs:
            yield doc


class _FakeKeyframesCollection:
    def __init__(self, docs: Sequence[dict[str, Any]]) -> None:
        self._docs = list(docs)
        self.find_calls: list[Mapping[str, Any]] = []

    def find(self, filter: Mapping[str, Any] | None = None):
        filter = filter or {}
        self.find_calls.append(filter)
        wanted = set(filter.get("keyframe_id", {}).get("$in", []))
        matched = [d for d in self._docs if d["keyframe_id"] in wanted]
        return _FakeCursor(matched)


class _FakeDB:
    def __init__(self, keyframes_docs: Sequence[dict[str, Any]]) -> None:
        self._collections = {
            "keyframes": _FakeKeyframesCollection(keyframes_docs),
            "videos": _FakeKeyframesCollection([]),
            "asr_runs": _FakeKeyframesCollection([]),
        }

    def __getitem__(self, name: str):
        return self._collections[name]


class _FakeAdmin:
    def __init__(self) -> None:
        self.raise_on_command = False

    async def command(self, name: str):
        if self.raise_on_command:
            raise ServerSelectionTimeoutError("no server available")
        return {"ok": 1.0}


class _FakeAsyncMongoClient:
    def __init__(self, keyframes_docs: Sequence[dict[str, Any]]) -> None:
        self._db = _FakeDB(keyframes_docs)
        self.admin = _FakeAdmin()

    def __getitem__(self, name: str):
        return self._db

    async def close(self) -> None:
        pass


def _store(monkeypatch, keyframes_docs: Sequence[dict[str, Any]]) -> MongoStore:
    monkeypatch.setattr(
        "app.stores.mongo_client.AsyncMongoClient",
        lambda uri: _FakeAsyncMongoClient(keyframes_docs),
    )
    return MongoStore(uri="mongodb://fake")


async def test_get_keyframes_by_ids_returns_mapping_keyed_by_keyframe_id(monkeypatch):
    docs = [
        {
            "keyframe_id": "L21_V001_0",
            "video_id": "L21_V001",
            "frame_idx": 10,
            "thumbnail_path": "t0.jpg",
        },
        {
            "keyframe_id": "L21_V001_4",
            "video_id": "L21_V001",
            "frame_idx": 40,
            "thumbnail_path": "t4.jpg",
        },
    ]
    store = _store(monkeypatch, docs)

    result = await store.get_keyframes_by_ids(["L21_V001_0", "L21_V001_4"])

    assert result["L21_V001_0"]["frame_idx"] == 10
    assert result["L21_V001_4"]["thumbnail_path"] == "t4.jpg"


async def test_get_keyframes_by_ids_returns_empty_dict_for_empty_input(monkeypatch):
    store = _store(monkeypatch, [])

    result = await store.get_keyframes_by_ids([])

    assert result == {}


async def test_get_keyframes_by_ids_omits_ids_not_found_in_mongo(monkeypatch):
    docs = [
        {"keyframe_id": "a", "video_id": "v", "frame_idx": 1, "thumbnail_path": "a.jpg"},
    ]
    store = _store(monkeypatch, docs)

    result = await store.get_keyframes_by_ids(["a", "missing"])

    assert set(result.keys()) == {"a"}


async def test_ping_returns_true_when_the_connection_is_healthy(monkeypatch):
    store = _store(monkeypatch, [])

    assert await store.ping() is True


async def test_ping_returns_false_instead_of_raising_when_mongo_is_unreachable(monkeypatch):
    store = _store(monkeypatch, [])
    store._client.admin.raise_on_command = True

    assert await store.ping() is False
