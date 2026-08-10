from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import app.main as main_module
from app.dependencies import get_search_service
from app.main import create_app
from app.services.search_service import SearchHit


class _FakeEncoder:
    embedding_dim = 4
    model_id = "fake-model"

    def encode_image(self, images):  # pragma: no cover - unused here
        raise NotImplementedError

    def encode_text(self, texts):  # pragma: no cover - unused here
        raise NotImplementedError


class _FakeMilvusStore:
    def ping(self) -> bool:
        return True


class _FakeEsStore:
    def __init__(self) -> None:
        self.ensure_index_called = False
        self.closed = False

    async def ensure_index(self) -> None:
        self.ensure_index_called = True

    async def ping(self) -> bool:
        return True

    async def close(self) -> None:
        self.closed = True


class _FakeMongoStore:
    def __init__(self) -> None:
        self.ensure_indexes_called = False
        self.closed = False

    async def ensure_indexes(self) -> None:
        self.ensure_indexes_called = True

    async def ping(self) -> bool:
        return True

    async def close(self) -> None:
        self.closed = True


def _stub_real_infra(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every lifespan test below needs the app to start without a GPU or
    Docker -- these stub out exactly the four factory functions main.py
    calls, leaving lifespan's own wiring (order, app.state assignment,
    ensure_index/ensure_indexes, close on shutdown) genuinely exercised."""
    monkeypatch.setattr(main_module, "_build_encoder", lambda: _FakeEncoder())
    monkeypatch.setattr(
        main_module, "_build_milvus_store", lambda settings, encoder: _FakeMilvusStore()
    )
    monkeypatch.setattr(main_module, "_build_es_store", lambda settings: _FakeEsStore())
    monkeypatch.setattr(main_module, "_build_mongo_store", lambda settings: _FakeMongoStore())


def test_create_app_returns_configured_fastapi_instance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)  # isolate from the repo's real .env
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-openai-key")

    app = create_app()

    assert app.title == "AIC Video Retrieval"


def test_lifespan_loads_settings_onto_app_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)  # isolate from the repo's real .env
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-openai-key")
    _stub_real_infra(monkeypatch)

    app = create_app()

    with TestClient(app) as client:
        assert client.app.state.settings.gemini_api_key.get_secret_value() == "fake-gemini-key"


def test_lifespan_raises_clear_error_when_api_key_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)  # isolate from the repo's real .env
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "fake-openai-key")
    # No _stub_real_infra: Settings() validation must fail before any
    # factory function is even called.

    app = create_app()

    with pytest.raises(ValidationError):
        with TestClient(app):
            pass


def test_lifespan_builds_encoder_and_all_three_stores_onto_app_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-openai-key")
    _stub_real_infra(monkeypatch)

    app = create_app()

    with TestClient(app) as client:
        state = client.app.state
        assert isinstance(state.encoder, _FakeEncoder)
        assert isinstance(state.milvus_store, _FakeMilvusStore)
        assert isinstance(state.es_store, _FakeEsStore)
        assert isinstance(state.mongo_store, _FakeMongoStore)
        assert state.mongo_store.ensure_indexes_called is True
        assert state.es_store.ensure_index_called is True


def test_lifespan_closes_es_and_mongo_stores_on_shutdown(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-openai-key")
    _stub_real_infra(monkeypatch)

    app = create_app()

    with TestClient(app) as client:
        es_store = client.app.state.es_store
        mongo_store = client.app.state.mongo_store

    assert es_store.closed is True
    assert mongo_store.closed is True


def test_health_endpoint_is_reachable_end_to_end(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-openai-key")
    _stub_real_infra(monkeypatch)

    app = create_app()

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "milvus_ok": True,
        "elasticsearch_ok": True,
        "mongodb_ok": True,
    }


def test_search_endpoint_is_reachable_end_to_end(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-openai-key")
    _stub_real_infra(monkeypatch)

    class _FakeSearchService:
        async def search(self, query, mode, top_k):
            return [
                SearchHit(
                    keyframe_id="L21_V001_f10",
                    video_id="L21_V001",
                    score=0.9,
                    frame_idx=10,
                    timestamp=0.4,
                    thumbnail_path="t.jpg",
                )
            ]

    app = create_app()
    app.dependency_overrides[get_search_service] = lambda: _FakeSearchService()

    with TestClient(app) as client:
        response = client.post("/api/search", json={"query": "thu tuong", "mode": "semantic"})

    assert response.status_code == 200
    assert response.json()["results"][0]["keyframe_id"] == "L21_V001_f10"
