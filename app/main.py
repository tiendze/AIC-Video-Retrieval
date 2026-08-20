import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import yaml
from ddgs import DDGS
from elasticsearch import ConnectionError as EsConnectionError
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.errors import register_error_handling
from app.api.health import router as health_router
from app.api.keyframes import router as keyframes_router
from app.api.qa import router as qa_router
from app.api.quest import router as quest_router
from app.api.search import register_upload_guard
from app.api.search import router as search_router
from app.api.trake import router as trake_router
from app.api.videos import router as videos_router
from app.config import CorsSettings, Settings
from app.encoders.base import MultimodalEncoder
from app.encoders.factory import build_encoder
from app.services.query_translator import LocalNmtTranslator
from app.services.quest_service import GeminiQueryRewriter, QueryRewriteService, QuestService
from app.services.search_service import SearchService
from app.services.thumbnail_service import ThumbnailService
from app.services.trake_embedding_cache import TrakeEmbeddingCache
from app.services.vqa_service import (
    GeminiVqaClient,
    QaService,
    ThumbnailImageLoader,
    VqaConfig,
    VqaService,
)
from app.stores.es_client import EsStore
from app.stores.milvus_client import MilvusStore
from app.stores.mongo_client import MongoStore

_REPO_ROOT = Path(__file__).resolve().parent.parent
_FG_CLIP_CONFIG_PATH = _REPO_ROOT / "configs" / "model" / "fg_clip.yaml"
_QUERY_TRANSLATOR_CONFIG_PATH = _REPO_ROOT / "configs" / "model" / "query_translator.yaml"
_MILVUS_CONFIG_PATH = _REPO_ROOT / "configs" / "stores" / "milvus.yaml"
_SEARCH_CONFIG_PATH = _REPO_ROOT / "configs" / "search" / "search.yaml"
_QUEST_CONFIG_PATH = _REPO_ROOT / "configs" / "search" / "quest.yaml"
_QA_CONFIG_PATH = _REPO_ROOT / "configs" / "qa" / "vqa.yaml"
# Mirrors scripts/run_ingest.py's _ES_OCR_INDEX_NAME -- the OCR branch writes
# to this same index name, so the API must read from it, not a new default.
_ES_OCR_INDEX_NAME = "ocr_keyframes"

logger = logging.getLogger(__name__)


def _load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_encoder() -> MultimodalEncoder:
    return build_encoder(_FG_CLIP_CONFIG_PATH)


def _build_milvus_store(settings: Settings, encoder: MultimodalEncoder) -> MilvusStore:
    config = _load_yaml(_MILVUS_CONFIG_PATH)
    return MilvusStore(
        uri=settings.milvus_uri,
        token=settings.milvus_token.get_secret_value(),
        model_id=encoder.model_id,
        embedding_dim=encoder.embedding_dim,
        hnsw_m=config["hnsw_m"],
        hnsw_ef_construction=config["hnsw_ef_construction"],
        search_ef=config["search_ef"],
    )


def _build_es_store(settings: Settings) -> EsStore:
    return EsStore(
        url=settings.elasticsearch_url,
        index_name=_ES_OCR_INDEX_NAME,
        api_key=settings.elastic_api_key.get_secret_value() if settings.elastic_api_key else None,
    )


def _build_mongo_store(settings: Settings) -> MongoStore:
    return MongoStore(uri=settings.mongo_uri)


def _build_thumbnail_service(mongo_store: MongoStore) -> ThumbnailService:
    config = _load_yaml(_SEARCH_CONFIG_PATH)
    return ThumbnailService(
        mongo_store=mongo_store,
        cache_dir=_REPO_ROOT / config["thumbnail_cache_dir"],
        max_px=config["thumbnail_max_px"],
        jpeg_quality=config["thumbnail_jpeg_quality"],
        cache_max_age_seconds=config["thumbnail_cache_max_age_seconds"],
    )


def _build_trake_embedding_cache(
    milvus_store: MilvusStore, mongo_store: MongoStore, encoder: MultimodalEncoder
) -> TrakeEmbeddingCache:
    return TrakeEmbeddingCache(
        milvus_store=milvus_store,
        mongo_store=mongo_store,
        expected_embedding_dim=encoder.embedding_dim,
    )


def _build_quest_service(
    search_service: SearchService, http_client: httpx.AsyncClient
) -> QuestService:
    config = _load_yaml(_QUEST_CONFIG_PATH)
    return QuestService(
        search_service=search_service,
        image_search_client=DDGS(),
        http_client=http_client,
        max_candidates=config["max_candidates"],
        image_candidates_timeout_sec=config["image_candidates_timeout_sec"],
        image_download_timeout_sec=config["image_download_timeout_sec"],
        candidate_cache_ttl_sec=config["candidate_cache_ttl_sec"],
    )


def _build_query_rewrite_service(settings: Settings) -> QueryRewriteService:
    config = _load_yaml(_QUEST_CONFIG_PATH)
    rewriter = GeminiQueryRewriter(
        api_key=settings.gemini_api_key.get_secret_value(),
        model=config["query_rewrite_model"],
        max_retries=config["query_rewrite_max_retries"],
        retry_base_delay_seconds=config["query_rewrite_retry_base_delay_sec"],
    )
    return QueryRewriteService(
        query_rewriter=rewriter, timeout_sec=config["query_rewrite_timeout_sec"]
    )


def _build_qa_service(
    settings: Settings, search_service: SearchService, thumbnail_service: ThumbnailService
) -> QaService:
    config = VqaConfig(**_load_yaml(_QA_CONFIG_PATH))
    return QaService(
        search_service=search_service,
        # Thumbnails, not full-res sources: VQA sends several images per
        # question and the corpus is 221,903 full-res JPEGs
        # (app/services/vqa_service.py:ThumbnailImageLoader).
        image_loader=ThumbnailImageLoader(thumbnail_service),
        vqa_service=VqaService(
            vqa_client=GeminiVqaClient(
                api_key=settings.gemini_api_key.get_secret_value(), config=config
            ),
            timeout_sec=config.timeout_seconds,
        ),
        max_keyframes_to_model=config.max_keyframes_to_model,
    )


def _build_query_translator() -> LocalNmtTranslator:
    config = _load_yaml(_QUERY_TRANSLATOR_CONFIG_PATH)
    return LocalNmtTranslator(
        model_id=config["model_id"],
        device=config["device"],
        source_prefix=config["source_prefix"],
        target_prefix=config["target_prefix"],
        max_new_tokens=config["max_new_tokens"],
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = Settings()
    app.state.settings = settings
    app.state.encoder = _build_encoder()
    app.state.milvus_store = _build_milvus_store(settings, app.state.encoder)
    app.state.es_store = _build_es_store(settings)
    app.state.mongo_store = _build_mongo_store(settings)
    app.state.query_translator = _build_query_translator()
    app.state.thumbnail_service = _build_thumbnail_service(app.state.mongo_store)
    app.state.trake_embedding_cache = _build_trake_embedding_cache(
        app.state.milvus_store, app.state.mongo_store, app.state.encoder
    )
    app.state.quest_http_client = httpx.AsyncClient()
    # One SearchService shared by QUEST's fallback path and QA's retrieval
    # step -- both must search the same vector space as /api/search
    # (CLAUDE.md §3: one encoder, one collection, no second code path).
    search_service = SearchService(
        encoder=app.state.encoder,
        milvus_store=app.state.milvus_store,
        es_store=app.state.es_store,
        mongo_store=app.state.mongo_store,
        query_translator=app.state.query_translator,
    )
    app.state.search_service = search_service
    app.state.quest_service = _build_quest_service(search_service, app.state.quest_http_client)
    app.state.query_rewrite_service = _build_query_rewrite_service(settings)
    app.state.qa_service = _build_qa_service(settings, search_service, app.state.thumbnail_service)
    await app.state.mongo_store.ensure_indexes()
    try:
        await app.state.es_store.ensure_index()
    except EsConnectionError:
        # plan §P1.8, PRD T5: Elasticsearch unreachable at startup must
        # degrade to semantic-only, not take the whole app down --
        # GET /health already reports elasticsearch_ok=False so the
        # frontend can show this (app/services/health_service.py).
        logger.warning(
            "Elasticsearch unreachable at startup -- starting in semantic-only "
            "degraded mode (OCR search will return a clear error until it recovers)"
        )
    # Loaded once per FastAPI process. With multiple uvicorn workers, each
    # worker owns one RAM copy of the TRAKE corpus cache.
    await app.state.trake_embedding_cache.load()
    yield
    await app.state.quest_http_client.aclose()
    await app.state.es_store.close()
    await app.state.mongo_store.close()


def create_app() -> FastAPI:
    app = FastAPI(title="AIC Video Retrieval", lifespan=lifespan)
    # Built from CorsSettings, not Settings: `frontend/`'s dev server needs
    # CORS configured at app-construction time, before `lifespan` runs, and
    # Settings() requires GEMINI_API_KEY/OPENAI_API_KEY that CORS setup has
    # no business depending on (app/config.py's CorsSettings docstring).
    cors_settings = CorsSettings()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_settings.cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    register_error_handling(app)
    register_upload_guard(app)
    app.include_router(health_router)
    app.include_router(search_router)
    app.include_router(trake_router)
    app.include_router(keyframes_router)
    app.include_router(videos_router)
    app.include_router(quest_router)
    app.include_router(qa_router)
    return app


app = create_app()
