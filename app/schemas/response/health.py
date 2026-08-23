"""Response shape for `GET /health` (plan §P0.8)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    milvus_ok: bool
    elasticsearch_ok: bool
    mongodb_ok: bool
    asr_status: Literal["unreachable", "unavailable", "empty", "ready"]
    asr_ready: bool
    asr_segment_count: int | None
