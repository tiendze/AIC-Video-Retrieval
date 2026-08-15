"""RED reproducer for `QueryRewriteService`/`GeminiQueryRewriter` (plan
§P2.5, branch 1 -- QUEST rewriting q0 into a more visually descriptive q_r
for an Out-of-Knowledge entity, PRD §F5).

Per chủ dự án's decision (2026-08-15): the operator picks which of q0/q_r
to actually search with -- `QueryRewriteService.rewrite_query()` never calls
`SearchService` itself, it only returns both strings for the UI to display
(CLAUDE.md §3: interactive system, don't automate the step the paper left
for the operator to decide). Fallback -> `rewritten_query=None`, `q0` is
still always returned so the operator can search with it unchanged.

`GeminiQueryRewriter`'s retry policy mirrors `pipelines/ocr.py:GeminiOcrClient`
and `eval/build_eval_set.py:GeminiCaptionGenerator` exactly (see
tests/test_build_eval_set.py::TestGeminiCaptionGeneratorRetry) -- text-only
input this time, no image.
"""

from __future__ import annotations

import pytest

from app.services.quest_service import (
    GeminiQueryRewriter,
    QueryRewriteResult,
    QueryRewriteService,
)


class _FakeQueryRewriter:
    def __init__(self, rewritten: str | None = None, error: Exception | None = None) -> None:
        self._rewritten = rewritten
        self._error = error
        self.rewrite_calls: list[str] = []

    def rewrite(self, query: str) -> str:
        self.rewrite_calls.append(query)
        if self._error is not None:
            raise self._error
        return self._rewritten


class _SlowFakeQueryRewriter:
    def __init__(self, sleep_seconds: float) -> None:
        self._sleep_seconds = sleep_seconds
        self.rewrite_calls: list[str] = []

    def rewrite(self, query: str) -> str:
        self.rewrite_calls.append(query)
        import time

        time.sleep(self._sleep_seconds)
        return "unreachable"


def _service(rewriter, timeout_sec: float = 8.0) -> QueryRewriteService:
    return QueryRewriteService(query_rewriter=rewriter, timeout_sec=timeout_sec)


class TestRewriteQuery:
    @pytest.mark.asyncio
    async def test_returns_rewritten_status_with_both_queries_on_success(self) -> None:
        rewriter = _FakeQueryRewriter(rewritten="một con búp bê đồ chơi tai thỏ màu hồng")
        service = _service(rewriter)

        result = await service.rewrite_query("Labubu")

        assert isinstance(result, QueryRewriteResult)
        assert result.status == "rewritten"
        assert result.original_query == "Labubu"
        assert result.rewritten_query == "một con búp bê đồ chơi tai thỏ màu hồng"
        assert rewriter.rewrite_calls == ["Labubu"]

    @pytest.mark.asyncio
    async def test_falls_back_when_gemini_raises_runtime_error_after_exhausted_retries(
        self,
    ) -> None:
        rewriter = _FakeQueryRewriter(error=RuntimeError("Gemini query rewrite failed"))
        service = _service(rewriter)

        result = await service.rewrite_query("Labubu")

        assert result.status == "fallback"
        assert result.original_query == "Labubu"
        assert result.rewritten_query is None

    @pytest.mark.asyncio
    async def test_falls_back_when_gemini_returns_malformed_structured_output(self) -> None:
        rewriter = _FakeQueryRewriter(error=ValueError("invalid JSON from structured output"))
        service = _service(rewriter)

        result = await service.rewrite_query("Labubu")

        assert result.status == "fallback"
        assert result.rewritten_query is None

    @pytest.mark.asyncio
    async def test_falls_back_when_the_call_exceeds_the_configured_timeout(self) -> None:
        rewriter = _SlowFakeQueryRewriter(sleep_seconds=0.1)
        service = _service(rewriter, timeout_sec=0.01)

        result = await service.rewrite_query("Labubu")

        assert result.status == "fallback"
        assert result.original_query == "Labubu"
        assert result.rewritten_query is None

    @pytest.mark.asyncio
    async def test_original_query_is_always_the_exact_q0_even_on_fallback(self) -> None:
        rewriter = _FakeQueryRewriter(error=RuntimeError("boom"))
        service = _service(rewriter)

        result = await service.rewrite_query("thủ tướng phát biểu")

        assert result.original_query == "thủ tướng phát biểu"


class TestGeminiQueryRewriterRetry:
    def test_retries_on_transient_error_then_succeeds(self, monkeypatch) -> None:
        rewriter = object.__new__(GeminiQueryRewriter)  # bypass __init__: no real Gemini client
        object.__setattr__(rewriter, "_max_retries", 3)
        object.__setattr__(rewriter, "_retry_base_delay_seconds", 0.0)

        calls = {"count": 0}

        def _flaky_call_once(query: str) -> str:
            calls["count"] += 1
            if calls["count"] < 3:
                raise ConnectionError("simulated transient network error")
            return "rewritten query"

        monkeypatch.setattr(rewriter, "_call_once", _flaky_call_once)

        result = rewriter.rewrite("Labubu")

        assert result == "rewritten query"
        assert calls["count"] == 3

    def test_raises_runtime_error_after_exhausting_retries(self, monkeypatch) -> None:
        rewriter = object.__new__(GeminiQueryRewriter)
        object.__setattr__(rewriter, "_max_retries", 2)
        object.__setattr__(rewriter, "_retry_base_delay_seconds", 0.0)

        def _always_fails(query: str) -> str:
            raise OSError("simulated permanent network error")

        monkeypatch.setattr(rewriter, "_call_once", _always_fails)

        with pytest.raises(RuntimeError, match="failed after 2 attempts"):
            rewriter.rewrite("Labubu")
