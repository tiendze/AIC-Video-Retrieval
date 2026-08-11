"""Request shape for `POST /api/search` (plan §P0.8). `top_k` bounds come
from configs/search/search.yaml, never hardcoded (CLAUDE.md §6)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent / "configs" / "search" / "search.yaml"
)


def _load_config() -> dict:
    with _CONFIG_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


_config = _load_config()
DEFAULT_TOP_K: int = _config["default_top_k"]
MAX_TOP_K: int = _config["max_top_k"]
MAX_UPLOAD_SIZE_BYTES: int = _config["max_upload_size_bytes"]
ALLOWED_UPLOAD_MIME_TYPES: frozenset[str] = frozenset(_config["allowed_upload_mime_types"])
# hybrid mode RRF fusion (plan §P1.3) -- read here (not hardcoded in
# SearchService) so app/dependencies.py and eval/run.py wire the SAME
# configured value a live query and an eval run both see.
HYBRID_RRF_K: int = _config["hybrid_rrf_k"]
HYBRID_SEMANTIC_WEIGHT: float = _config["hybrid_semantic_weight"]
HYBRID_OCR_WEIGHT: float = _config["hybrid_ocr_weight"]


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    mode: Literal["semantic", "ocr", "hybrid"]
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=MAX_TOP_K)
