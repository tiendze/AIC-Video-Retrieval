from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class CorsSettings(BaseSettings):
    """The subset of app settings `create_app()` needs before `lifespan`
    ever runs (plan §P1.6 backend gap 1.1) -- CORS middleware must be
    registered at app-construction time, but `Settings` below requires
    `gemini_api_key`/`openai_api_key` and is deliberately built later, inside
    `lifespan`, so a missing key fails loudly at startup rather than mid-run
    (see `Settings`'s own docstring). Splitting this subset out means
    `create_app()` never has to construct the secret-requiring `Settings`
    just to read `cors_origins`."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    cors_origins: list[str] = ["http://localhost:5173"]
    app_env: str = "development"


class Settings(CorsSettings):
    """Application settings, loaded from environment variables / `.env`.

    All secrets use `SecretStr` so log lines and repr never leak the value.
    A missing `gemini_api_key` or `openai_api_key` must fail loudly at
    startup, not silently mid-way through a multi-hour ingest run.
    """

    gemini_api_key: SecretStr
    openai_api_key: SecretStr

    milvus_uri: str = "http://localhost:19530"
    milvus_token: SecretStr = SecretStr("root:Milvus")

    elasticsearch_url: str = "http://localhost:9200"
    elastic_api_key: SecretStr | None = None

    mongo_uri: str = "mongodb://localhost:27017"
