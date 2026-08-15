"""`Depends` providers for encoder/store clients (architecture.md §6) --
routers inject through these, never instantiate a client themselves
(.claude/rules/ecc/python/fastapi.md). Each provider is a thin read off
`request.app.state`; the actual instances are built once in `app/main.py`'s
lifespan, not here.
"""

from __future__ import annotations

from fastapi import Depends, Request

from app.encoders.base import MultimodalEncoder
from app.services.health_service import HealthService
from app.services.query_translator import QueryTranslator
from app.services.quest_service import QueryRewriteService, QuestService
from app.services.search_service import SearchService
from app.services.thumbnail_service import ThumbnailService
from app.services.trake_service import TrakeService
from app.stores.es_client import EsStore
from app.stores.milvus_client import MilvusStore
from app.stores.mongo_client import MongoStore


def get_encoder(request: Request) -> MultimodalEncoder:
    return request.app.state.encoder


def get_query_translator(request: Request) -> QueryTranslator:
    return request.app.state.query_translator


def get_milvus_store(request: Request) -> MilvusStore:
    return request.app.state.milvus_store


def get_es_store(request: Request) -> EsStore:
    return request.app.state.es_store


def get_mongo_store(request: Request) -> MongoStore:
    return request.app.state.mongo_store


def get_thumbnail_service(request: Request) -> ThumbnailService:
    return request.app.state.thumbnail_service


def get_quest_service(request: Request) -> QuestService:
    return request.app.state.quest_service


def get_query_rewrite_service(request: Request) -> QueryRewriteService:
    return request.app.state.query_rewrite_service


def get_health_service(
    milvus_store: MilvusStore = Depends(get_milvus_store),
    es_store: EsStore = Depends(get_es_store),
    mongo_store: MongoStore = Depends(get_mongo_store),
) -> HealthService:
    return HealthService(milvus_store=milvus_store, es_store=es_store, mongo_store=mongo_store)


def get_trake_service(
    encoder: MultimodalEncoder = Depends(get_encoder),
    milvus_store: MilvusStore = Depends(get_milvus_store),
    mongo_store: MongoStore = Depends(get_mongo_store),
    query_translator: QueryTranslator = Depends(get_query_translator),
) -> TrakeService:
    return TrakeService(
        encoder=encoder,
        milvus_store=milvus_store,
        mongo_store=mongo_store,
        query_translator=query_translator,
    )


def get_search_service(
    encoder: MultimodalEncoder = Depends(get_encoder),
    milvus_store: MilvusStore = Depends(get_milvus_store),
    es_store: EsStore = Depends(get_es_store),
    mongo_store: MongoStore = Depends(get_mongo_store),
    query_translator: QueryTranslator = Depends(get_query_translator),
) -> SearchService:
    return SearchService(
        encoder=encoder,
        milvus_store=milvus_store,
        es_store=es_store,
        mongo_store=mongo_store,
        query_translator=query_translator,
    )
