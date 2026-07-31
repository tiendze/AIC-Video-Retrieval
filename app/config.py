from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, loaded from environment variables / `.env`.

    All secrets use `SecretStr` so log lines and repr never leak the value.
    A missing `gemini_api_key` or `openai_api_key` must fail loudly at
    startup, not silently mid-way through a multi-hour ingest run.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    gemini_api_key: SecretStr
    openai_api_key: SecretStr

    milvus_uri: str = "http://localhost:19540"
    milvus_token: SecretStr = SecretStr("root:Milvus")

    elasticsearch_url: str = "http://localhost:9201"
    elastic_api_key: SecretStr | None = None

    mongo_uri: str = "mongodb://localhost:27017"

    cors_origins: list[str] = ["http://localhost:5173"]
    app_env: str = "development"
