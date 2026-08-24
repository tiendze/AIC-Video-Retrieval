"""QueryTranslator — VI->EN query-time translation for the `semantic` search
channel.

The production vision-language text tower is English-caption oriented, so a
Vietnamese query is off-distribution for semantic retrieval even though the
image tower is unchanged. Translating the query at search time keeps the
existing collection valid -- no re-embedding, no new `model_id` (CLAUDE.md §3).

Shared by two callers so a sidecar generated for the eval diagnostic and a
live production query are translated identically: `app.services.search_service`
(production, `mode="semantic"` only -- the `ocr` index holds Vietnamese text,
so translating an OCR query would measure translation loss instead of the
retrieval channel) and `scripts/translate_eval_queries.py` (offline sidecar
generation for the `semantic@en` eval channel).

Two implementations satisfy the Protocol, selected by `kind` in
`configs/model/query_translator.yaml` through `build_query_translator()`:

- `LocalNmtTranslator` -- EnViT5, offline, no quota, no network.
- `GeminiTranslator` -- an LLM asked to translate *for retrieval*.

The LLM branch exists because EnViT5's failures are silent and land exactly on
the words that carry the retrieval signal. Measured on the 25 SOTUYEN1 contest
queries at git 5191098: `con lân` -> "phosphor", `cầm dù` -> "carrying
parachutes", `lia cam` -> "a single orange peel", `xanh ngọc` -> "yellow", and
one query lost a whole clause (65 Vietnamese words in, 36 English tokens out).
Each returns a fluent English sentence that no keyframe matches, and none of
them raise.

`build_query_translator()` is the single construction site for all three entry
points, so production and both eval paths cannot drift onto different models --
the property this module exists to protect.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Protocol

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class QueryTranslator(Protocol):
    @property
    def model_id(self) -> str:
        """Stamped onto sidecar rows for provenance (eval/query_translation.py
        `TranslatedQuery.translator_model`)."""
        ...

    def translate(self, vietnamese: str) -> str: ...


def strip_target_prefix(decoded: str, target_prefix: str) -> str:
    """EnViT5 echoes the target-language tag it was trained with, so a decode
    comes back as `"en: A mother donkey..."`. Leaving the tag in would make it
    part of the query text that gets embedded and searched.

    Only strips at the start and only the configured tag -- a caption that
    legitimately contains "en:" mid-sentence is left alone. Models that need no
    tag (VinAI-Translate) get `target_prefix=""` and pass through untouched.
    """
    if not target_prefix:
        return decoded.strip()
    text = decoded.strip()
    return text[len(target_prefix) :].strip() if text.startswith(target_prefix) else text


class LocalNmtTranslator:
    """Offline VI->EN translation via a local seq2seq model -- no API key, no
    quota, no network after the first HuggingFace download.

    Default is `VietAI/envit5-translation` (EnViT5, trained on MTet + PhoMT):
    a T5-base-sized seq2seq that reports state-of-the-art en<->vi BLEU while
    staying small enough to run on CPU. Greedy decoding (no sampling) keeps
    translation deterministic -- the same query must translate identically on
    every call, otherwise the same search issued twice could return different
    results.

    Model swap (e.g. to a more Vietnamese-specialized NMT model) is a config
    change only -- `configs/model/query_translator.yaml` -- since every
    caller talks to the `QueryTranslator` Protocol, not this class directly.
    """

    def __init__(
        self,
        model_id: str,
        device: str,
        source_prefix: str,
        target_prefix: str,
        max_new_tokens: int,
    ) -> None:
        import torch  # lazy: only needed when this class is used
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self._model_id = model_id
        self._torch = torch
        self._device = torch.device(device)
        self._source_prefix = source_prefix
        self._target_prefix = target_prefix
        self._max_new_tokens = max_new_tokens
        logger.info("Loading %s on %s ...", model_id, device)
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        self._model.to(self._device).eval()

    @property
    def model_id(self) -> str:
        return self._model_id

    def translate(self, vietnamese: str) -> str:
        prompt = f"{self._source_prefix}{vietnamese}"
        encoded = self._tokenizer(prompt, return_tensors="pt", padding=True)
        encoded = {key: value.to(self._device) for key, value in encoded.items()}
        with self._torch.no_grad():
            generated = self._model.generate(
                **encoded,
                max_new_tokens=self._max_new_tokens,
                num_beams=1,
                do_sample=False,
            )
        decoded = self._tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        english = strip_target_prefix(decoded, self._target_prefix)
        if not english:
            # An empty translation would be embedded as an empty query and
            # silently score 0 -- fail the search instead (CLAUDE.md §7).
            raise ValueError(f"empty translation for {vietnamese!r} (decoded={decoded!r})")
        return english


class Translation(BaseModel):
    """Structured-output schema for the LLM branch. Free-text parsing is banned
    (CLAUDE.md §5), so the model is pinned to this schema and the result is read
    from a validated field rather than scraped out of prose."""

    english: str = Field(
        description=(
            "The complete English translation of the Vietnamese query, preserving "
            "every scene, object, colour and count from the original."
        )
    )


# The rules below are not stylistic preferences -- each one names a class of
# failure measured on the SOTUYEN1 contest queries (see module docstring).
_TRANSLATION_PROMPT_TEMPLATE = (
    "Dịch truy vấn tìm kiếm video tiếng Việt sau sang tiếng Anh, để dùng cho "
    "tìm kiếm keyframe bằng mô hình vision-language.\n\n"
    "Quy tắc:\n"
    "1. Dịch ĐẦY ĐỦ. Không bỏ bớt, không tóm tắt, không gộp mệnh đề. Mỗi cảnh, "
    "mỗi vật thể, mỗi màu sắc và mỗi con số trong bản gốc phải xuất hiện trong "
    "bản dịch.\n"
    "2. Giữ nguyên số đếm và từ chỉ số lượng: 'hơn 5 người' là 'more than five "
    "people', không phải 'five people'.\n"
    "3. Khái niệm văn hoá Việt Nam phải dịch thành thứ NHÌN THẤY ĐƯỢC, kèm tên "
    "riêng nếu có. Ví dụ: 'con lân' là 'a lion dance costume', 'áo dài' là "
    "'ao dai, a long Vietnamese tunic dress', 'khăn quàng đỏ' là 'a red neck "
    "scarf'. Tuyệt đối không dịch theo nghĩa đen của từng chữ.\n"
    "4. Thuật ngữ quay phim giữ đúng nghĩa điện ảnh: 'lia cam' là 'camera pan', "
    "'quay chậm' là 'slow motion', 'cảnh cận' là 'close-up', 'zoom xa dần' là "
    "'zooming out'.\n"
    "5. Không thêm chi tiết không có trong bản gốc.\n\n"
    'Truy vấn: "{query}"'
)


class GeminiTranslator:
    """Wraps `client.interactions.create(...)` -- the current Gemini
    Interactions API (references.md §5.1), never `generate_content`. Mirrors
    `app.services.quest_service.GeminiQueryRewriter` exactly (retry policy, lazy
    import, structured-output schema pattern).

    `gemini-3.6-flash` is mandatory, not a default: `gemini-2.5-flash` ignores
    `response_format` outright and returns prose instead of schema-valid JSON
    (CLAUDE.md §5, references.md §5.1).

    Raises on failure rather than degrading, so an offline sidecar run fails
    loudly and can be resumed. Production wraps this in `FallbackTranslator`.
    """

    def __init__(
        self, api_key: str, model: str, max_retries: int, retry_base_delay_seconds: float
    ) -> None:
        from google import genai  # lazy: only needed when this class is used

        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._max_retries = max_retries
        self._retry_base_delay_seconds = retry_base_delay_seconds

    @property
    def model_id(self) -> str:
        return self._model

    def translate(self, vietnamese: str) -> str:
        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                return self._call_once(vietnamese)
            except (TimeoutError, ConnectionError, OSError) as exc:
                # Only transient transport errors are retried. A schema
                # violation would fail identically on every attempt, so it
                # propagates immediately instead of burning quota.
                last_exc = exc
                time.sleep(self._retry_base_delay_seconds * (2**attempt))
        raise RuntimeError(
            f"Gemini translation failed after {self._max_retries} attempts"
        ) from last_exc

    def _call_once(self, vietnamese: str) -> str:
        interaction = self._client.interactions.create(
            model=self._model,
            input=_TRANSLATION_PROMPT_TEMPLATE.format(query=vietnamese),
            response_format={
                "type": "text",
                "mime_type": "application/json",
                "schema": Translation.model_json_schema(),
            },
        )
        english = Translation.model_validate_json(interaction.output_text).english.strip()
        if not english:
            # Same guard as LocalNmtTranslator: an empty query embeds to a
            # meaningless vector and scores ~0 against everything (CLAUDE.md §7).
            raise ValueError(f"empty translation for {vietnamese!r}")
        return english


class FallbackTranslator:
    """Runs `primary`, falling back to `fallback` on timeout or failure.

    This is what keeps degraded mode real (CLAUDE.md §4): losing the network or
    the Gemini quota mid-contest must leave semantic search working, not hang an
    operator who is being timed. Both sides failing raises -- returning "" would
    embed an empty vector and report a confident, meaningless ranking.

    The catch around the primary call is intentionally `Exception`, not a
    specific list. `pipelines/ocr.py:GeminiOcrClient` and this module's own
    `GeminiTranslator` retry loop both narrow their catch to
    (TimeoutError, ConnectionError, OSError) with the documented caveat that
    google-genai's real exception hierarchy has not been verified against a
    live key -- so an unlisted exception type would skip the retry AND escape
    a narrow fallback catch here, crashing the search request instead of
    degrading to EnViT5. That failure mode is exactly what this class exists
    to prevent, so the fallback boundary must catch broadly: any primary
    failure, known type or not, degrades to `fallback` rather than propagating.
    `BaseException` (KeyboardInterrupt, SystemExit) is deliberately NOT caught.

    `model_id` deliberately names BOTH models. A sidecar built through this
    class records the *set* of possible translators, not the one that wrote each
    row, so `scripts/translate_eval_queries.py` builds its translator with
    `with_fallback=False` to keep per-row provenance exact.
    """

    def __init__(
        self, primary: QueryTranslator, fallback: QueryTranslator, timeout_sec: float
    ) -> None:
        self._primary = primary
        self._fallback = fallback
        self._timeout_sec = timeout_sec
        # Shared and bounded on purpose: a per-call executor would spin up an
        # unbounded new thread pool on every timeout, so a sustained Gemini
        # hang leaks one permanently-blocked thread per query for the life of
        # the process. Reusing a small fixed pool caps that leak instead of
        # letting it grow without limit; a stuck call still costs one worker
        # slot until the underlying HTTP call itself times out or the process
        # exits, but at most 4 slots are ever at risk, matching the bounded
        # (not per-call) shape of asyncio's own default executor that
        # `QueryRewriteService.rewrite_query` relies on
        # (app/services/quest_service.py:350-358).
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="query-translate")

    @property
    def model_id(self) -> str:
        return f"{self._primary.model_id}|{self._fallback.model_id}"

    def translate(self, vietnamese: str) -> str:
        try:
            return self._call_primary(vietnamese)
        except Exception as exc:  # noqa: BLE001 -- intentional, see class docstring
            logger.warning(
                "query translation fell back from %s to %s: query=%r reason=%r",
                self._primary.model_id,
                self._fallback.model_id,
                vietnamese,
                exc,
            )
            primary_error = exc

        try:
            return self._fallback.translate(vietnamese)
        except Exception as exc:  # noqa: BLE001 -- same rationale: must not crash the request
            raise RuntimeError(
                f"both translators failed for {vietnamese!r}: "
                f"{self._primary.model_id} -> {primary_error!r}, "
                f"{self._fallback.model_id} -> {exc!r}"
            ) from exc

    def _call_primary(self, vietnamese: str) -> str:
        # `translate` is synchronous by contract (the Protocol) and callers
        # already run it inside `asyncio.to_thread`, so the timeout has to
        # live here rather than as `asyncio.wait_for`.
        future = self._executor.submit(self._primary.translate, vietnamese)
        return future.result(timeout=self._timeout_sec)


def _build_local_nmt(config: Mapping[str, Any]) -> QueryTranslator:
    return LocalNmtTranslator(
        model_id=config["model_id"],
        device=config["device"],
        source_prefix=config["source_prefix"],
        target_prefix=config["target_prefix"],
        max_new_tokens=config["max_new_tokens"],
    )


def build_query_translator(
    config: Mapping[str, Any],
    api_key: str | None = None,
    *,
    with_fallback: bool = True,
) -> QueryTranslator:
    """Single construction site for every entry point that needs a translator.

    `app/main.py`, `eval/run.py` and `scripts/translate_eval_queries.py` all
    call this with the same `configs/model/query_translator.yaml`, so a query
    translated in production and the same query translated for a diagnostic can
    never come from different models.

    `with_fallback=False` returns the primary alone, for callers that stamp a
    per-row `translator_model` and need it to name exactly one model.
    """
    kind = str(config.get("kind", "local_nmt")).strip().lower()

    if kind == "local_nmt":
        return _build_local_nmt(config)

    if kind == "llm":
        if not api_key:
            raise ValueError(
                "query translator kind 'llm' requires an api_key -- refusing to "
                "start rather than silently falling back to the local model"
            )
        llm_config = config.get("llm") or {}
        primary = GeminiTranslator(
            api_key=api_key,
            model=llm_config["model"],
            max_retries=int(llm_config.get("max_retries", 3)),
            retry_base_delay_seconds=float(llm_config.get("retry_base_delay_seconds", 1.0)),
        )
        if not with_fallback:
            return primary
        return FallbackTranslator(
            primary,
            _build_local_nmt(config),
            timeout_sec=float(llm_config.get("timeout_seconds", 8.0)),
        )

    raise ValueError(
        f"unknown query translator kind: {kind!r} (expected 'local_nmt' or 'llm')"
    )
