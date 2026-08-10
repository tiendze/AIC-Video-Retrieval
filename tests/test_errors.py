"""RED reproducer for the unified API error schema (plan §P0.8,
architecture.md §7): every error out of the API must answer with
`{detail, error_code, request_id}` so a person under contest time pressure
never has to read a raw traceback (CLAUDE.md §7).

Builds a minimal throwaway FastAPI app with `register_error_handling()`
wired in, then hits it with `TestClient` -- no real router/store needed.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel

from app.api.errors import APIError, register_error_handling

_REQUEST_ID_HEADER = "X-Request-ID"


class _Body(BaseModel):
    query: str


def _app() -> FastAPI:
    app = FastAPI()
    register_error_handling(app)

    @app.get("/boom-api-error")
    async def boom_api_error():
        raise APIError(detail="unknown mode", error_code="invalid_mode", status_code=400)

    @app.get("/boom-unexpected")
    async def boom_unexpected():
        raise RuntimeError("something broke")

    @app.post("/validate")
    async def validate(body: _Body):
        return {"query": body.query}

    return app


def test_api_error_returns_unified_schema_with_its_own_status_code():
    client = TestClient(_app(), raise_server_exceptions=False)

    response = client.get("/boom-api-error")

    assert response.status_code == 400
    assert response.json() == {
        "detail": "unknown mode",
        "error_code": "invalid_mode",
        "request_id": response.headers[_REQUEST_ID_HEADER],
    }


def test_validation_error_returns_422_with_unified_schema():
    client = TestClient(_app(), raise_server_exceptions=False)

    bad_response = client.post("/validate", json={})  # missing required field

    assert bad_response.status_code == 422
    body = bad_response.json()
    assert body["error_code"] == "validation_error"
    assert set(body.keys()) == {"detail", "error_code", "request_id"}


def test_unhandled_exception_returns_500_with_unified_schema_and_no_traceback():
    client = TestClient(_app(), raise_server_exceptions=False)

    response = client.get("/boom-unexpected")

    assert response.status_code == 500
    body = response.json()
    assert body == {
        "detail": "Internal server error",
        "error_code": "internal_error",
        "request_id": response.headers[_REQUEST_ID_HEADER],
    }
    assert "RuntimeError" not in response.text
    assert "Traceback" not in response.text


def test_request_id_header_is_generated_when_absent():
    client = TestClient(_app(), raise_server_exceptions=False)

    response = client.get("/boom-api-error")

    assert response.headers[_REQUEST_ID_HEADER]


def test_incoming_request_id_header_is_echoed_back_not_replaced():
    client = TestClient(_app(), raise_server_exceptions=False)

    response = client.get("/boom-api-error", headers={_REQUEST_ID_HEADER: "client-supplied-id"})

    assert response.headers[_REQUEST_ID_HEADER] == "client-supplied-id"
    assert response.json()["request_id"] == "client-supplied-id"
