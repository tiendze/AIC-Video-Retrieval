from pathlib import Path

import pytest
from elasticsearch import ConnectionError as EsConnectionError
from elasticsearch import ConnectionTimeout as EsConnectionTimeout
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


class _FakeAsrEsStore(_FakeEsStore):
    index_name = "asr_segments"

    async def index_exists(self) -> bool:
        return True

    async def count_segments(self) -> int:
        return 873


class FailingAsrStore:
    index_name = "asr_segments"

    def __init__(self, error: Exception | None = None) -> None:
        self.closed = False
        self._error = error or EsConnectionError("simulated ASR index failure")

    async def ensure_index(self) -> None:
        raise self._error

    async def index_exists(self) -> bool:
        raise self._error

    async def count_segments(self) -> int:
        raise AssertionError("count must not run after index probe failure")

    async def close(self) -> None:
        self.closed = True


class _FakeAsrRetrievalService:
    def __init__(self, store) -> None:
        self.store = store


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


class _FakeTrakeEmbeddingCache:
    def __init__(self) -> None:
        self.load_called = False
        self.load_count = 0

    async def load(self) -> None:
        self.load_called = True
        self.load_count += 1


class _FakeEsStoreDown(_FakeEsStore):
    """Simulates Elasticsearch unreachable at startup (plan §P1.8, PRD T5:
    'hệ thống chuyển sang semantic-only' -- ensure_index failing must not
    take the whole app down with it)."""

    async def ensure_index(self) -> None:
        self.ensure_index_called = True
        raise EsConnectionError("simulated ES down")


class _FakeQueryTranslator:
    model_id = "fake-translator"

    def translate(self, vietnamese: str) -> str:  # pragma: no cover - unused here
        raise NotImplementedError


def _stub_real_infra(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every lifespan test below needs the app to start without a GPU or
    Docker -- these stub out exactly the five factory functions main.py
    calls, leaving lifespan's own wiring (order, app.state assignment,
    ensure_index/ensure_indexes, close on shutdown) genuinely exercised."""
    monkeypatch.setattr(main_module, "_build_encoder", lambda: _FakeEncoder())
    monkeypatch.setattr(
        main_module, "_build_milvus_store", lambda settings, encoder: _FakeMilvusStore()
    )
    monkeypatch.setattr(main_module, "_build_es_store", lambda settings: _FakeEsStore())
    monkeypatch.setattr(main_module, "_build_asr_es_store", lambda settings: _FakeAsrEsStore())
    monkeypatch.setattr(
        main_module,
        "_build_asr_retrieval_service",
        lambda asr_store: _FakeAsrRetrievalService(asr_store),
    )
    monkeypatch.setattr(main_module, "_build_mongo_store", lambda settings: _FakeMongoStore())
    monkeypatch.setattr(
        main_module, "_build_query_translator", lambda settings: _FakeQueryTranslator()
    )
    monkeypatch.setattr(
        main_module,
        "_build_trake_embedding_cache",
        lambda milvus_store, mongo_store, encoder: _FakeTrakeEmbeddingCache(),
    )


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


def test_build_encoder_uses_fg_clip_config(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[Path] = []
    fake_encoder = _FakeEncoder()

    def fake_build_encoder(path: Path):
        calls.append(path)
        return fake_encoder

    monkeypatch.setattr(main_module, "build_encoder", fake_build_encoder)

    assert main_module._build_encoder() is fake_encoder
    assert calls == [main_module._FG_CLIP_CONFIG_PATH]


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
        assert isinstance(state.asr_es_store, _FakeAsrEsStore)
        assert isinstance(state.mongo_store, _FakeMongoStore)
        assert isinstance(state.query_translator, _FakeQueryTranslator)
        assert isinstance(state.trake_embedding_cache, _FakeTrakeEmbeddingCache)
        assert state.mongo_store.ensure_indexes_called is True
        assert state.es_store.ensure_index_called is True
        assert state.asr_es_store.ensure_index_called is True
        assert state.trake_embedding_cache.load_called is True


def test_lifespan_survives_es_unreachable_at_startup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """plan §P1.8: Elasticsearch down at startup must degrade to
    semantic-only, not take the whole app down -- `ensure_index()` raising a
    connection error must not propagate out of `lifespan`."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-openai-key")
    monkeypatch.setattr(main_module, "_build_encoder", lambda: _FakeEncoder())
    _stub_real_infra(monkeypatch)
    monkeypatch.setattr(main_module, "_build_es_store", lambda settings: _FakeEsStoreDown())

    app = create_app()

    with TestClient(app) as client:  # must not raise despite ensure_index() failing
        assert client.app.state.es_store.ensure_index_called is True


def test_search_endpoint_still_works_when_es_was_unreachable_at_startup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The core degraded-mode promise: semantic search survives ES being
    down, not just app startup surviving it."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-openai-key")
    _stub_real_infra(monkeypatch)
    monkeypatch.setattr(main_module, "_build_es_store", lambda settings: _FakeEsStoreDown())

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
        asr_es_store = client.app.state.asr_es_store
        mongo_store = client.app.state.mongo_store

    assert es_store.closed is True
    assert asr_es_store.closed is True
    assert mongo_store.closed is True


def test_lifespan_builds_and_closes_one_asr_store(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-openai-key")
    _stub_real_infra(monkeypatch)

    with TestClient(create_app()) as client:
        state = client.app.state
        assert state.asr_es_store.index_name == "asr_segments"
        assert state.asr_retrieval_service is state.search_service._asr_retrieval_service
        assert state.asr_retrieval_service.store is state.asr_es_store
        asr_store = state.asr_es_store

    assert asr_store.closed is True


def test_lifespan_survives_asr_ensure_index_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-openai-key")
    _stub_real_infra(monkeypatch)
    monkeypatch.setattr(main_module, "_build_asr_es_store", lambda settings: FailingAsrStore())

    with TestClient(create_app()) as client:
        response = client.get("/health")
        asr_store = client.app.state.asr_es_store

    assert response.status_code == 200
    assert response.json()["asr_status"] == "unavailable"
    assert "ASR index unavailable" in caplog.text
    assert asr_store.closed is True


def test_lifespan_survives_asr_ensure_index_timeout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-openai-key")
    _stub_real_infra(monkeypatch)
    monkeypatch.setattr(
        main_module,
        "_build_asr_es_store",
        lambda settings: FailingAsrStore(EsConnectionTimeout("simulated ASR timeout")),
    )

    with TestClient(create_app()) as client:
        assert client.app.state.trake_embedding_cache.load_called is True

    assert "ASR index unavailable" in caplog.text


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
        "asr_status": "ready",
        "asr_ready": True,
        "asr_segment_count": 873,
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


def test_cors_preflight_allows_configured_frontend_origin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The Vite dev server (`:5173`, `app/config.py`'s `cors_origins` default)
    must be able to call the API from the browser -- without `CORSMiddleware`
    wired up, every `frontend/` request is blocked before it reaches a route
    (plan §P1.6 backend gap 1.1)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-openai-key")
    _stub_real_infra(monkeypatch)

    app = create_app()

    with TestClient(app) as client:
        response = client.options(
            "/api/search",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "POST",
            },
        )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_cors_rejects_unconfigured_origin(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-openai-key")
    _stub_real_infra(monkeypatch)

    app = create_app()

    with TestClient(app) as client:
        response = client.options(
            "/api/search",
            headers={
                "Origin": "http://evil.example.com",
                "Access-Control-Request-Method": "POST",
            },
        )

    assert "access-control-allow-origin" not in response.headers


def test_create_app_does_not_require_secrets_for_cors_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`create_app()` itself must not raise before the lifespan even starts --
    CORS middleware setup only needs `cors_origins`/`app_env`, never the
    Gemini/OpenAI secrets that `Settings()` requires (those are validated
    later, inside `lifespan`, per `test_lifespan_raises_clear_error_when_api_key_missing`)."""
    monkeypatch.chdir(tmp_path)  # isolate from the repo's real .env

    app = create_app()  # must not raise, even with no GEMINI_API_KEY/OPENAI_API_KEY set

    assert app.title == "AIC Video Retrieval"
