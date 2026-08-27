"""Behaviour lock for the LLM-backed VI->EN query translator.

Why this exists: measured on the 25 SOTUYEN1 contest queries at git 5191098,
`LocalNmtTranslator` (EnViT5) mistranslates exactly the nouns that carry the
retrieval signal -- `con lân` -> "phosphor", `cầm dù` -> "carrying parachutes",
`lia cam` -> "a single orange peel", `xanh ngọc` -> "yellow" -- and on one query
dropped a whole clause. None of those raise; they just return a fluent English
sentence that no keyframe matches.

So the tests that matter most are the ones that stop a *silent* wrong answer:
an empty translation must raise rather than be embedded, and a degraded LLM must
fall back to the local model rather than hang the search (CLAUDE.md §4 degraded
mode, §7 timeout + retry + fallback).
"""

from __future__ import annotations

import time

import pytest

from app.services.query_translator import (
    FallbackTranslator,
    GeminiTranslator,
    Translation,
    build_query_translator,
)

_VI = "Con lân do hai người điều khiển đang xoay vòng trên đỉnh cột."
_EN = "A lion dance costume operated by two performers spins on top of a pole."


class _FakeTranslator:
    """Stands in for either side of FallbackTranslator."""

    def __init__(
        self,
        *,
        english: str = "",
        error: Exception | None = None,
        sleep_seconds: float = 0.0,
        model: str = "fake-model",
    ) -> None:
        self._english = english
        self._error = error
        self._sleep_seconds = sleep_seconds
        self._model = model
        self.calls: list[str] = []

    @property
    def model_id(self) -> str:
        return self._model

    def translate(self, vietnamese: str) -> str:
        self.calls.append(vietnamese)
        if self._sleep_seconds:
            time.sleep(self._sleep_seconds)
        if self._error is not None:
            raise self._error
        return self._english


class _FakeInteractions:
    def __init__(self, output_text: str) -> None:
        self._output_text = output_text
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return type("_Interaction", (), {"output_text": self._output_text})()


class _FakeGenaiClient:
    def __init__(self, output_text: str) -> None:
        self.interactions = _FakeInteractions(output_text)


def _gemini(output_text: str, model: str = "gemini-3.6-flash") -> GeminiTranslator:
    """Builds a GeminiTranslator around a fake client, bypassing __init__ so no
    real `genai.Client` is constructed -- same pattern as
    tests/test_query_rewrite_service.py::TestGeminiQueryRewriterRetry."""
    translator = object.__new__(GeminiTranslator)
    object.__setattr__(translator, "_client", _FakeGenaiClient(output_text))
    object.__setattr__(translator, "_model", model)
    object.__setattr__(translator, "_max_retries", 3)
    object.__setattr__(translator, "_retry_base_delay_seconds", 0.0)
    return translator


class TestGeminiTranslatorCall:
    def test_returns_the_english_field_from_the_structured_response(self) -> None:
        translator = _gemini(Translation(english=_EN).model_dump_json())

        assert translator.translate(_VI) == _EN

    def test_sends_the_vietnamese_query_to_the_interactions_api(self) -> None:
        translator = _gemini(Translation(english=_EN).model_dump_json())

        translator.translate(_VI)

        call = translator._client.interactions.calls[0]
        assert call["model"] == "gemini-3.6-flash"
        assert _VI in call["input"]

    def test_requests_structured_output_against_the_translation_schema(self) -> None:
        # CLAUDE.md §5: structured output via model_json_schema(), never regex
        # over free text -- and gemini-2.5-flash is banned for this branch.
        translator = _gemini(Translation(english=_EN).model_dump_json())

        translator.translate(_VI)

        response_format = translator._client.interactions.calls[0]["response_format"]
        assert response_format["mime_type"] == "application/json"
        assert response_format["schema"] == Translation.model_json_schema()

    def test_strips_surrounding_whitespace(self) -> None:
        translator = _gemini(Translation(english=f"  {_EN}  ").model_dump_json())

        assert translator.translate(_VI) == _EN

    def test_raises_on_an_empty_translation_instead_of_embedding_it(self) -> None:
        # An empty query would be embedded and silently score ~0 against every
        # keyframe -- exactly the failure LocalNmtTranslator already guards.
        translator = _gemini(Translation(english="   ").model_dump_json())

        with pytest.raises(ValueError, match="empty translation"):
            translator.translate(_VI)

    def test_raises_on_a_malformed_response_rather_than_guessing(self) -> None:
        translator = _gemini("not json at all")

        with pytest.raises(ValueError):
            translator.translate(_VI)


class TestGeminiTranslatorRetry:
    def test_retries_on_transient_error_then_succeeds(self, monkeypatch) -> None:
        translator = _gemini(Translation(english=_EN).model_dump_json())
        calls = {"count": 0}

        def _flaky(vietnamese: str) -> str:
            calls["count"] += 1
            if calls["count"] < 3:
                raise ConnectionError("simulated transient network error")
            return _EN

        monkeypatch.setattr(translator, "_call_once", _flaky)

        assert translator.translate(_VI) == _EN
        assert calls["count"] == 3

    def test_raises_runtime_error_after_exhausting_retries(self, monkeypatch) -> None:
        translator = _gemini(Translation(english=_EN).model_dump_json())
        object.__setattr__(translator, "_max_retries", 2)

        def _always_fails(vietnamese: str) -> str:
            raise OSError("simulated permanent network error")

        monkeypatch.setattr(translator, "_call_once", _always_fails)

        with pytest.raises(RuntimeError, match="failed after 2 attempts"):
            translator.translate(_VI)

    def test_does_not_retry_a_schema_violation(self, monkeypatch) -> None:
        # A malformed payload will be malformed again; retrying just burns quota.
        translator = _gemini(Translation(english=_EN).model_dump_json())
        calls = {"count": 0}

        def _bad_payload(vietnamese: str) -> str:
            calls["count"] += 1
            raise ValueError("invalid JSON from structured output")

        monkeypatch.setattr(translator, "_call_once", _bad_payload)

        with pytest.raises(ValueError):
            translator.translate(_VI)
        assert calls["count"] == 1


class TestGeminiTranslatorProvenance:
    def test_model_id_reports_the_gemini_model(self) -> None:
        # Stamped onto sidecar rows (eval/query_translation.py) -- must name the
        # model that actually wrote the row.
        assert _gemini("{}", model="gemini-3.6-flash").model_id == "gemini-3.6-flash"


class TestFallbackTranslator:
    def test_uses_the_primary_when_it_succeeds(self) -> None:
        primary = _FakeTranslator(english=_EN)
        fallback = _FakeTranslator(english="local output")
        translator = FallbackTranslator(primary, fallback, timeout_sec=5.0)

        assert translator.translate(_VI) == _EN
        assert fallback.calls == []

    def test_falls_back_when_the_primary_raises(self) -> None:
        primary = _FakeTranslator(error=RuntimeError("Gemini failed after 3 attempts"))
        fallback = _FakeTranslator(english="local output")
        translator = FallbackTranslator(primary, fallback, timeout_sec=5.0)

        assert translator.translate(_VI) == "local output"
        assert fallback.calls == [_VI]

    def test_falls_back_when_the_primary_exceeds_the_timeout(self) -> None:
        # CLAUDE.md §4: an operator being timed must never watch a hung search.
        primary = _FakeTranslator(english=_EN, sleep_seconds=0.5)
        fallback = _FakeTranslator(english="local output")
        translator = FallbackTranslator(primary, fallback, timeout_sec=0.01)

        assert translator.translate(_VI) == "local output"

    def test_falls_back_when_the_primary_returns_an_empty_translation(self) -> None:
        primary = _FakeTranslator(error=ValueError("empty translation"))
        fallback = _FakeTranslator(english="local output")
        translator = FallbackTranslator(primary, fallback, timeout_sec=5.0)

        assert translator.translate(_VI) == "local output"

    def test_raises_when_both_sides_fail_instead_of_returning_empty(self) -> None:
        # CLAUDE.md §7: missing data fails loudly; returning "" here would embed
        # an empty vector and report a meaningless ranking.
        primary = _FakeTranslator(error=RuntimeError("primary down"))
        fallback = _FakeTranslator(error=RuntimeError("local model down"))
        translator = FallbackTranslator(primary, fallback, timeout_sec=5.0)

        with pytest.raises(RuntimeError, match="both translators failed"):
            translator.translate(_VI)

    def test_model_id_names_both_sides_so_provenance_is_not_a_single_model(self) -> None:
        primary = _FakeTranslator(english=_EN, model="gemini-3.6-flash")
        fallback = _FakeTranslator(english="x", model="VietAI/envit5-translation")
        translator = FallbackTranslator(primary, fallback, timeout_sec=5.0)

        assert translator.model_id == "gemini-3.6-flash|VietAI/envit5-translation"


class TestBuildQueryTranslator:
    _LOCAL_CONFIG = {
        "kind": "local_nmt",
        "model_id": "VietAI/envit5-translation",
        "device": "cpu",
        "source_prefix": "vi: ",
        "target_prefix": "en: ",
        "max_new_tokens": 128,
    }
    _LLM_CONFIG = {
        **_LOCAL_CONFIG,
        "kind": "llm",
        "llm": {
            "model": "gemini-3.6-flash",
            "max_retries": 3,
            "retry_base_delay_seconds": 1.0,
            "timeout_seconds": 8.0,
        },
    }

    def test_builds_the_local_model_when_kind_is_local_nmt(self, monkeypatch) -> None:
        built: dict = {}

        def _fake_local(**kwargs):
            built.update(kwargs)
            return _FakeTranslator(english="local")

        monkeypatch.setattr("app.services.query_translator.LocalNmtTranslator", _fake_local)

        build_query_translator(self._LOCAL_CONFIG)

        assert built["model_id"] == "VietAI/envit5-translation"
        assert built["device"] == "cpu"

    def test_defaults_to_the_local_model_when_kind_is_absent(self, monkeypatch) -> None:
        # Configs written before the LLM branch existed must keep working.
        calls = {"count": 0}

        def _fake_local(**kwargs):
            calls["count"] += 1
            return _FakeTranslator(english="local")

        monkeypatch.setattr("app.services.query_translator.LocalNmtTranslator", _fake_local)
        config = {k: v for k, v in self._LOCAL_CONFIG.items() if k != "kind"}

        build_query_translator(config)

        assert calls["count"] == 1

    def test_llm_kind_wraps_the_local_model_as_fallback(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "app.services.query_translator.LocalNmtTranslator",
            lambda **kwargs: _FakeTranslator(english="local", model="envit5"),
        )
        monkeypatch.setattr(
            "app.services.query_translator.GeminiTranslator",
            lambda **kwargs: _FakeTranslator(english=_EN, model="gemini-3.6-flash"),
        )

        translator = build_query_translator(self._LLM_CONFIG, api_key="k")

        assert isinstance(translator, FallbackTranslator)
        assert translator.translate(_VI) == _EN

    def test_with_fallback_false_returns_a_single_named_translator(self, monkeypatch) -> None:
        # The eval sidecar stamps translator_model per row, so it must never be
        # handed a composite that hides which model wrote the row.
        monkeypatch.setattr(
            "app.services.query_translator.GeminiTranslator",
            lambda **kwargs: _FakeTranslator(english=_EN, model="gemini-3.6-flash"),
        )

        translator = build_query_translator(self._LLM_CONFIG, api_key="k", with_fallback=False)

        assert not isinstance(translator, FallbackTranslator)
        assert translator.model_id == "gemini-3.6-flash"

    def test_llm_kind_without_an_api_key_fails_loudly(self) -> None:
        with pytest.raises(ValueError, match="api_key"):
            build_query_translator(self._LLM_CONFIG, api_key=None)

    def test_unknown_kind_is_rejected_rather_than_silently_defaulted(self) -> None:
        with pytest.raises(ValueError, match="unknown query translator kind"):
            build_query_translator({**self._LOCAL_CONFIG, "kind": "magic"})
