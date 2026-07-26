import pytest
from pydantic import ValidationError

from app.config import Settings


def test_settings_loads_from_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-openai-key")

    settings = Settings(_env_file=None)

    assert settings.gemini_api_key.get_secret_value() == "fake-gemini-key"
    assert settings.openai_api_key.get_secret_value() == "fake-openai-key"
    assert settings.milvus_uri == "http://localhost:19530"
    assert settings.mongo_uri == "mongodb://localhost:27017"
    assert settings.elasticsearch_url == "http://localhost:9200"


def test_settings_raises_when_gemini_api_key_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "fake-openai-key")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_settings_raises_when_openai_api_key_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_settings_masks_secrets_in_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-openai-key")

    settings = Settings(_env_file=None)

    assert "fake-gemini-key" not in repr(settings)
    assert "fake-openai-key" not in repr(settings)
