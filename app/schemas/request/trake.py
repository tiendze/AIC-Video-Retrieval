"""Request shape for `POST /api/trake` (plan §P2.2). `top_k` bounds come
from configs/search/trake.yaml, never hardcoded (CLAUDE.md §6)."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent / "configs" / "search" / "trake.yaml"
)


def _load_config() -> dict:
    with _CONFIG_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


_config = _load_config()
DEFAULT_TOP_K: int = _config["default_top_k"]
MAX_TOP_K: int = _config["max_top_k"]


class TrakeRequest(BaseModel):
    events: list[str] = Field(min_length=1)  # N event descriptions, in time order
    lambda_penalty: float | None = None  # None = use dante.yaml's configured default
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=MAX_TOP_K)
