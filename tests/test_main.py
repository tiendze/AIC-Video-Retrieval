from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.main import create_app


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

    app = create_app()

    with TestClient(app) as client:
        assert client.app.state.settings.gemini_api_key.get_secret_value() == "fake-gemini-key"


def test_lifespan_raises_clear_error_when_api_key_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)  # isolate from the repo's real .env
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "fake-openai-key")

    app = create_app()

    with pytest.raises(ValidationError):
        with TestClient(app):
            pass
