"""HealthService — reports connectivity to all three stores (plan §P0.8,
architecture.md §7). A store being unreachable is expected degraded-mode
data (CLAUDE.md §4: turning off Elasticsearch must still leave semantic
search usable), so `check()` never raises -- it reports `status="degraded"`
instead.

`MilvusStore.ping()` is synchronous (same reasoning as SearchService: Milvus
calls run off the event loop via `asyncio.to_thread`); `EsStore`/`MongoStore`
already expose async `ping()`.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Literal

from app.stores.es_client import EsStore
from app.stores.milvus_client import MilvusStore
from app.stores.mongo_client import MongoStore


@dataclass(frozen=True)
class HealthReport:
    status: Literal["ok", "degraded"]
    milvus_ok: bool
    elasticsearch_ok: bool
    mongodb_ok: bool


class HealthService:
    def __init__(
        self, milvus_store: MilvusStore, es_store: EsStore, mongo_store: MongoStore
    ) -> None:
        self._milvus_store = milvus_store
        self._es_store = es_store
        self._mongo_store = mongo_store

    async def check(self) -> HealthReport:
        milvus_ok, es_ok, mongo_ok = await asyncio.gather(
            asyncio.to_thread(self._milvus_store.ping),
            self._es_store.ping(),
            self._mongo_store.ping(),
        )
        status = "ok" if milvus_ok and es_ok and mongo_ok else "degraded"
        return HealthReport(
            status=status, milvus_ok=milvus_ok, elasticsearch_ok=es_ok, mongodb_ok=mongo_ok
        )
