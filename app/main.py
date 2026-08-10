from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from fastapi import FastAPI

from app.api.errors import register_error_handling
from app.api.health import router as health_router
from app.api.search import router as search_router
from app.config import Settings
from app.encoders.base import MultimodalEncoder
from app.encoders.beit3 import BEiT3Encoder
from app.stores.es_client import EsStore
from app.stores.milvus_client import MilvusStore
from app.stores.mongo_client import MongoStore

_REPO_ROOT = Path(__file__).resolve().parent.parent
_BEIT3_CONFIG_PATH = _REPO_ROOT / "configs" / "model" / "beit3.yaml"
_MILVUS_CONFIG_PATH = _REPO_ROOT / "configs" / "stores" / "milvus.yaml"
# Mirrors scripts/run_ingest.py's _ES_OCR_INDEX_NAME -- the OCR branch writes
# to this same index name, so the API must read from it, not a new default.
_ES_OCR_INDEX_NAME = "ocr_keyframes"


def _load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_encoder() -> BEiT3Encoder:
    config = _load_yaml(_BEIT3_CONFIG_PATH)
    return BEiT3Encoder(
        checkpoint_path=str(_REPO_ROOT / config["checkpoint_path"]),
        sentencepiece_model_path=str(_REPO_ROOT / config["sentencepiece_model_path"]),
        device=config["device"],
        batch_size=config["batch_size"],
        model_id=config["model_id"],
    )


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


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = Settings()
    app.state.settings = settings
    app.state.encoder = _build_encoder()
    app.state.milvus_store = _build_milvus_store(settings, app.state.encoder)
    app.state.es_store = _build_es_store(settings)
    app.state.mongo_store = _build_mongo_store(settings)
    await app.state.mongo_store.ensure_indexes()
    await app.state.es_store.ensure_index()
    yield
    await app.state.es_store.close()
    await app.state.mongo_store.close()


def create_app() -> FastAPI:
    app = FastAPI(title="AIC Video Retrieval", lifespan=lifespan)
    register_error_handling(app)
    app.include_router(health_router)
    app.include_router(search_router)
    return app


app = create_app()
