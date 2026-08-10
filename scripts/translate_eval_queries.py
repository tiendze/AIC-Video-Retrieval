"""CLI: generate the VI->EN query sidecar for the `semantic@en` eval channel.

Translates the ground-truth CAPTIONS ONLY. `configs/eval/aic_eval_v1.json` and
its `.sha256` are never rewritten, so the resulting metrics stay directly
comparable to the 2026-08-10 baseline (CLAUDE.md §8).

Why this exists: BEiT-3's image-text contrastive stage was trained on English
pairs, so a Vietnamese query is off-distribution for the text tower while the
image tower is untouched. Scoring the SAME Milvus collection with English
queries tells us which tower is at fault before any GPU time is spent
re-embedding 873 videos under a new model.

Resumable: the sidecar is flushed every `checkpoint_every` translations and
re-read on startup, so a quota or rate-limit error never discards calls
already paid for (same discipline as eval/build_eval_set.py).

Per CLAUDE.md §10 this is prepared for the project owner to run -- it spends
Gemini quota:

  python -m scripts.translate_eval_queries --limit 10   # cheap smoke test
  python -m scripts.translate_eval_queries              # full 173 captions
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import yaml
from pydantic import BaseModel, Field

from app.config import Settings
from datasets.aic2026 import AIC2026Adapter
from datasets.base import GroundTruthSegment
from eval.query_translation import (
    TranslatedQuery,
    key_of,
    load_translations,
    models_used,
    to_payload,
)
from eval.run import filter_ground_truth_to_available, load_pilot_available

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = _REPO_ROOT / "configs" / "eval" / "query_translation.yaml"
_MAX_REPORTED_FAILURES = 10


@dataclass(frozen=True)
class TranslationConfig:
    """Mirrors `configs/eval/query_translation.yaml` field for field."""

    split: str
    dataset_root: str
    eval_set_path: str
    pilot_available_path: str
    output_path: str
    translator: str
    gemini_model: str
    local_model_id: str
    local_device: str
    local_source_prefix: str
    local_target_prefix: str
    local_max_new_tokens: int
    max_retries: int
    retry_base_delay_seconds: float
    checkpoint_every: int


class QueryTranslation(BaseModel):
    """Structured-output schema -- no regex over free text (CLAUDE.md §5)."""

    english_query: str = Field(
        description=(
            "Faithful English translation of the Vietnamese sentence, phrased as a "
            "natural image-caption style description. Preserve every visual detail "
            "(objects, people, actions, colours, setting). Do not add detail that is "
            "not in the Vietnamese sentence, and do not explain or comment."
        )
    )


class Translator(Protocol):
    @property
    def model_id(self) -> str:
        """Stamped onto every row this translator writes, so a sidecar built
        across a Gemini run and a local run records which model wrote what."""
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


class GeminiQueryTranslator:
    """Wraps `client.interactions.create(...)` -- the current Gemini
    Interactions API (references.md §5.1), never `generate_content`. Retry
    policy mirrors `pipelines/ocr.py:GeminiOcrClient`."""

    def __init__(self, api_key: str, config: TranslationConfig) -> None:
        from google import genai  # lazy: only needed when this class is used

        self._client = genai.Client(api_key=api_key)
        self._config = config

    @property
    def model_id(self) -> str:
        return self._config.gemini_model

    def translate(self, vietnamese: str) -> str:
        last_exc: Exception | None = None
        for attempt in range(self._config.max_retries):
            try:
                return self._call_once(vietnamese)
            except (TimeoutError, ConnectionError, OSError) as exc:
                last_exc = exc
                time.sleep(self._config.retry_base_delay_seconds * (2**attempt))
        raise RuntimeError(
            f"Gemini translation failed after {self._config.max_retries} attempts"
        ) from last_exc

    def _call_once(self, vietnamese: str) -> str:
        prompt = (
            "Dịch câu tiếng Việt sau sang tiếng Anh, giữ nguyên MỌI chi tiết hình "
            "ảnh (người, vật, hành động, màu sắc, bối cảnh). Viết theo văn phong "
            "mô tả ảnh tự nhiên. KHÔNG thêm chi tiết không có trong câu gốc, "
            "KHÔNG giải thích.\n\n"
            f"Câu tiếng Việt: {vietnamese}"
        )
        interaction = self._client.interactions.create(
            model=self._config.gemini_model,
            input=[{"type": "text", "text": prompt}],
            response_format={
                "type": "text",
                "mime_type": "application/json",
                "schema": QueryTranslation.model_json_schema(),
            },
        )
        return QueryTranslation.model_validate_json(interaction.output_text).english_query


class LocalNmtTranslator:
    """Offline VI->EN translation — no API key, no quota, no network after the
    first model download.

    Default is `VietAI/envit5-translation` (EnViT5, trained on MTet + PhoMT):
    a T5-base-sized seq2seq that reports state-of-the-art en<->vi BLEU while
    staying small enough to run on CPU, so it does not compete with BEiT-3 for
    the 8GB of VRAM (CLAUDE.md §1.4). Greedy decoding (no sampling) keeps the
    sidecar reproducible -- the same caption must translate identically on a
    re-run, otherwise resuming a partial file would mix two different English
    queries for the same label.
    """

    def __init__(self, config: TranslationConfig) -> None:
        import torch  # lazy: only needed when this class is used
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self._config = config
        self._torch = torch
        self._device = torch.device(config.local_device)
        logger.info("Loading %s on %s ...", config.local_model_id, config.local_device)
        self._tokenizer = AutoTokenizer.from_pretrained(config.local_model_id)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(config.local_model_id)
        self._model.to(self._device).eval()

    @property
    def model_id(self) -> str:
        return self._config.local_model_id

    def translate(self, vietnamese: str) -> str:
        prompt = f"{self._config.local_source_prefix}{vietnamese}"
        encoded = self._tokenizer(prompt, return_tensors="pt", padding=True)
        encoded = {key: value.to(self._device) for key, value in encoded.items()}
        with self._torch.no_grad():
            generated = self._model.generate(
                **encoded,
                max_new_tokens=self._config.local_max_new_tokens,
                num_beams=1,
                do_sample=False,
            )
        decoded = self._tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        english = strip_target_prefix(decoded, self._config.local_target_prefix)
        if not english:
            # An empty translation would be embedded as an empty query and
            # silently score 0 -- fail the row instead (CLAUDE.md §7).
            raise ValueError(f"empty translation for {vietnamese!r} (decoded={decoded!r})")
        return english


_TRANSLATOR_LOCAL = "local"
_TRANSLATOR_GEMINI = "gemini"


def build_translator(config: TranslationConfig) -> Translator:
    """Picks the implementation named by `config.translator`.

    `Settings()` is constructed only on the Gemini branch: it validates
    `gemini_api_key` at construction, and the offline path must run on a
    machine with no Gemini key and no quota at all.
    """
    if config.translator == _TRANSLATOR_LOCAL:
        return LocalNmtTranslator(config)
    if config.translator == _TRANSLATOR_GEMINI:
        return GeminiQueryTranslator(Settings().gemini_api_key.get_secret_value(), config)
    raise ValueError(
        f"unknown translator {config.translator!r} -- "
        f"expected {_TRANSLATOR_LOCAL!r} or {_TRANSLATOR_GEMINI!r}"
    )


def _load_config(path: Path) -> TranslationConfig:
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return TranslationConfig(**raw)


def _load_ground_truth(config: TranslationConfig) -> list[GroundTruthSegment]:
    """Same population eval/run.py scores -- reusing its loaders so the sidecar
    can never cover a different query set than the report that consumes it."""
    adapter = AIC2026Adapter(
        dataset_root=_REPO_ROOT / config.dataset_root,
        eval_set_path=_REPO_ROOT / config.eval_set_path,
    )
    available = load_pilot_available(_REPO_ROOT / config.pilot_available_path)
    return filter_ground_truth_to_available(adapter.load_ground_truth(config.split), available)


def _eval_set_sha256(config: TranslationConfig) -> str:
    """Reads the sibling `.sha256` rather than re-hashing, so the sidecar
    records exactly the identifier the eval report cites."""
    hash_path = _REPO_ROOT / (config.eval_set_path + ".sha256")
    return hash_path.read_text(encoding="utf-8").strip() if hash_path.exists() else "unknown"


def _load_existing(output_path: Path) -> dict[str, TranslatedQuery]:
    """Prior run's output, or empty on a first run. A malformed sidecar raises
    out of `load_translations` -- overwriting it silently would throw away
    translations already paid for."""
    if not output_path.exists():
        return {}
    existing = load_translations(output_path)
    logger.info("Resuming: %d translation(s) already on disk -- not re-billed", len(existing))
    return existing


def _write_sidecar(
    output_path: Path,
    translations: Sequence[TranslatedQuery],
    config: TranslationConfig,
    eval_set_sha256: str,
) -> None:
    payload: dict[str, Any] = to_payload(
        translations,
        eval_set_path=config.eval_set_path,
        eval_set_sha256=eval_set_sha256,
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def translate_all(
    ground_truth: Sequence[GroundTruthSegment],
    translator: Translator,
    existing: dict[str, TranslatedQuery],
    config: TranslationConfig,
    on_checkpoint: Callable[[Sequence[TranslatedQuery]], None],
) -> tuple[list[TranslatedQuery], list[str]]:
    """Translates every caption that has none yet. Returns
    `(all translations, failed keys)`.

    A per-caption failure is recorded and skipped rather than aborting the
    batch: one rate-limited call must not discard the other 172 (CLAUDE.md §7
    -- logged with its key, surfaced in the return value, never swallowed).
    """
    results = dict(existing)
    failed: list[str] = []
    pending = [gt for gt in ground_truth if key_of(gt) not in results]
    logger.info("%d caption(s) to translate (%d already done)", len(pending), len(results))

    for index, gt in enumerate(pending, start=1):
        key = key_of(gt)
        try:
            english = translator.translate(gt.caption)
        except Exception as exc:  # noqa: BLE001 -- see eval/build_eval_set.py
            # google-genai raises from two unrelated hierarchies (documented
            # `google.genai.errors.*` and `_gaos.lib.compat_errors.*`), neither
            # an OSError/RuntimeError subclass, so there is no narrower type to
            # catch without letting one bad call kill the whole batch.
            logger.error("%s: translation failed: %s", key, exc)
            failed.append(key)
            continue

        results[key] = TranslatedQuery(
            video_id=gt.video_id,
            order_index=gt.order_index,
            source_caption=gt.caption,
            translated_query=english,
            translator_model=translator.model_id,
        )
        if index % config.checkpoint_every == 0:
            on_checkpoint(list(results.values()))
            logger.info("checkpoint: %d/%d translated", index, len(pending))

    return list(results.values()), failed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=_CONFIG_PATH)
    parser.add_argument(
        "--limit", type=int, default=0, help="translate at most N captions (smoke test); 0 = all"
    )
    parser.add_argument(
        "--translator",
        choices=[_TRANSLATOR_LOCAL, _TRANSLATOR_GEMINI],
        default=None,
        help="override the translator in --config",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = _parse_args()
    config = _load_config(args.config)
    if args.translator is not None:
        config = replace(config, translator=args.translator)

    ground_truth = _load_ground_truth(config)
    if args.limit > 0:
        ground_truth = ground_truth[: args.limit]
    if not ground_truth:
        raise SystemExit(
            "no ground-truth record survived the pilot_available filter -- "
            "nothing to translate (check configs/eval/pilot_available_v1.txt)"
        )

    output_path = _REPO_ROOT / config.output_path
    eval_set_sha256 = _eval_set_sha256(config)
    existing = _load_existing(output_path)
    translator = build_translator(config)

    already_used = models_used(list(existing.values()))
    if already_used and translator.model_id not in already_used:
        logger.warning(
            "This sidecar already holds rows from %s; new rows will come from %s. "
            "The English channel will be a MIX of translators -- each row records "
            "its own translator_model, so check that column before reading the numbers.",
            ", ".join(already_used),
            translator.model_id,
        )

    def checkpoint(translations: Sequence[TranslatedQuery]) -> None:
        _write_sidecar(output_path, translations, config, eval_set_sha256)

    translations, failed = translate_all(ground_truth, translator, existing, config, checkpoint)
    _write_sidecar(output_path, translations, config, eval_set_sha256)

    logger.info("Wrote %d translation(s) to %s", len(translations), output_path)
    if failed:
        logger.warning(
            "%d caption(s) failed and have NO translation: %s -- re-run to retry them "
            "(eval/run.py refuses the sidecar until every scored query has one)",
            len(failed),
            ", ".join(failed[:_MAX_REPORTED_FAILURES]),
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
