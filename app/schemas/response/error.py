"""Unified error response shape (architecture.md §7) -- every API error
answers with exactly these three fields, never a raw traceback (CLAUDE.md §7)."""

from __future__ import annotations

from pydantic import BaseModel


class ErrorResponse(BaseModel):
    detail: str
    error_code: str
    request_id: str
