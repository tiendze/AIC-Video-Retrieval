"""CLI: generate the VI->EN query sidecar for the `semantic@en` eval channel.

Translates the ground-truth CAPTIONS ONLY. `configs/eval/aic_eval_v1.json` and
its `.sha256` are never rewritten, so the resulting metrics stay directly
comparable to the 2026-08-10 baseline (CLAUDE.md §8).

Why this exists: BEiT-3's image-text contrastive stage was trained on English
pairs, so a Vietnamese query is off-distribution for the text tower while the
image tower is untouched. Scoring the SAME Milvus collection with English
queries tells us which tower is at fault before any GPU time is spent
re-embedding 873 videos under a new model. Confirmed 2026-08-11: AIC Final
Score went 0.0451 -> 0.6208 -- the text tower is the fault, and translating
the query at search time (app/services/query_translator.py, wired into
SearchService) is the production fix, not an encoder swap.

Translator is `LocalNmtTranslator` (app/services/query_translator.py) only --
no Gemini branch. Settings come from `configs/model/query_translator.yaml`,
the SAME file `app/main.py` reads for the production SearchService, so a
sidecar generated here and a live query translated in production never drift
apart.

Resumable: the sidecar is flushed every `checkpoint_every` translations and
re-read on startup, so an interrupted run never discards translations already
computed (same discipline as eval/build_eval_set.py).

Per CLAUDE.md §10 this is prepared for the project owner to run -- it loads a
local translation model:

  python -m scripts.translate_eval_queries --limit 10   # cheap smoke test
  python -m scripts.translate_eval_queries              # full 173 captions
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from app.config import Settings
from app.services.query_translator import QueryTranslator, build_query_translator
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
_QUERY_TRANSLATOR_CONFIG_PATH = _REPO_ROOT / "configs" / "model" / "query_translator.yaml"
_MAX_REPORTED_FAILURES = 10


@dataclass(frozen=True)
class TranslationConfig:
    """Mirrors `configs/eval/query_translation.yaml` field for field."""

    split: str
    dataset_root: str
    eval_set_path: str
    pilot_available_path: str
    output_path: str
    checkpoint_every: int


def build_translator() -> QueryTranslator:
    """Reads the SAME model config app/main.py uses for the production
    SearchService, through the SAME factory, so this sidecar and a live query
    are translated identically.

    `with_fallback=False`: see the module docstring -- per-row provenance must
    name exactly one model."""
    with _QUERY_TRANSLATOR_CONFIG_PATH.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)
    api_key = (
        Settings().gemini_api_key.get_secret_value()
        if str(config.get("kind", "local_nmt")).strip().lower() == "llm"
        else None
    )
    return build_query_translator(config, api_key=api_key, with_fallback=False)


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
    translator: QueryTranslator,
    existing: dict[str, TranslatedQuery],
    config: TranslationConfig,
    on_checkpoint: Callable[[Sequence[TranslatedQuery]], None],
) -> tuple[list[TranslatedQuery], list[str]]:
    """Translates every caption that has none yet. Returns
    `(all translations, failed keys)`.

    A per-caption failure is recorded and skipped rather than aborting the
    batch: one bad generation must not discard the other 172 (CLAUDE.md §7 --
    logged with its key, surfaced in the return value, never swallowed).
    """
    results = dict(existing)
    failed: list[str] = []
    pending = [gt for gt in ground_truth if key_of(gt) not in results]
    logger.info("%d caption(s) to translate (%d already done)", len(pending), len(results))

    for index, gt in enumerate(pending, start=1):
        key = key_of(gt)
        try:
            english = translator.translate(gt.caption)
        except (RuntimeError, ValueError) as exc:
            # RuntimeError: torch/CUDA generation failure. ValueError:
            # LocalNmtTranslator's own empty-decode guard.
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
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = _parse_args()
    config = _load_config(args.config)

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
    translator = build_translator()

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
