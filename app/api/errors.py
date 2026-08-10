"""Unified API error handling (plan §P0.8, architecture.md §7) -- every
error out of the API answers with `{detail, error_code, request_id}` so a
person under contest time pressure never has to read a raw traceback
(CLAUDE.md §7). `register_error_handling()` is called once from
`create_app()` so every router gets this for free without importing it.
"""

from __future__ import annotations

import logging
import uuid

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.schemas.response.error import ErrorResponse

logger = logging.getLogger(__name__)

_REQUEST_ID_HEADER = "X-Request-ID"


class APIError(Exception):
    """Raised deliberately by a router/service for an expected failure
    (e.g. an unknown search mode) -- not a bug, so it carries its own HTTP
    status and a machine-readable `error_code` a client can branch on,
    distinct from the human-readable `detail`."""

    def __init__(
        self, detail: str, error_code: str, status_code: int = status.HTTP_400_BAD_REQUEST
    ) -> None:
        super().__init__(detail)
        self.detail = detail
        self.error_code = error_code
        self.status_code = status_code


async def _request_id_middleware(request: Request, call_next):
    """Reuses a caller-supplied `X-Request-ID` if present, otherwise mints
    one -- either way every response from this point on can be correlated
    back to a single request, in logs and in error bodies alike."""
    request_id = request.headers.get(_REQUEST_ID_HEADER) or str(uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers[_REQUEST_ID_HEADER] = request_id
    return response


def _request_id_of(request: Request) -> str:
    return getattr(request.state, "request_id", None) or str(uuid.uuid4())


def _error_response(status_code: int, body: ErrorResponse) -> JSONResponse:
    # The request-id middleware's post-call_next header injection never runs
    # for a raised exception (call_next itself raises), so every error
    # response must stamp the header here too, not rely on the middleware.
    return JSONResponse(
        status_code=status_code,
        content=body.model_dump(),
        headers={_REQUEST_ID_HEADER: body.request_id},
    )


async def _handle_api_error(request: Request, exc: APIError) -> JSONResponse:
    body = ErrorResponse(
        detail=exc.detail, error_code=exc.error_code, request_id=_request_id_of(request)
    )
    return _error_response(exc.status_code, body)


async def _handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    body = ErrorResponse(
        detail="Invalid request", error_code="validation_error", request_id=_request_id_of(request)
    )
    return _error_response(status.HTTP_422_UNPROCESSABLE_CONTENT, body)


async def _handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    request_id = _request_id_of(request)
    logger.exception("Unhandled error for request_id=%s", request_id)
    body = ErrorResponse(
        detail="Internal server error", error_code="internal_error", request_id=request_id
    )
    return _error_response(status.HTTP_500_INTERNAL_SERVER_ERROR, body)


def register_error_handling(app: FastAPI) -> None:
    app.middleware("http")(_request_id_middleware)
    app.add_exception_handler(APIError, _handle_api_error)
    app.add_exception_handler(RequestValidationError, _handle_validation_error)
    app.add_exception_handler(Exception, _handle_unexpected_error)
