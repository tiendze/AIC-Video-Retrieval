"""Query-side VI->EN translation sidecar — labels stay frozen, only the query
text changes.

Why a sidecar and not an edited eval set: CLAUDE.md §8 forbids touching the
label set to move a number ("Không sửa bộ eval để cải thiện số"). Translating
the query is not a label edit — `video_id`, `start_sec`, `end_sec` and
`order_index` are untouched, so `aic_eval_v1.json` and its `.sha256` stay
byte-identical and every metric stays comparable to the 2026-08-10 baseline.
This module keeps translations in a SEPARATE file, joined to the ground truth
at search time.

What this buys, diagnostically: BEiT-3's ITC stage was trained on English
image-text pairs, so a Vietnamese query is off-distribution for the text tower
while the image tower is untouched. Running the SAME collection with English
queries isolates which tower is at fault, with zero re-embedding:

  - English >> Vietnamese -> image tower is fine, the text tower is the
    problem; swapping to a multilingual encoder should pay off (and
    translate-at-query-time is itself a legitimate production design, since
    the competition issues Vietnamese queries).
  - English ~= Vietnamese ~= random -> the fault is on the image/index side
    and no encoder swap will fix it.

`consistency_error` guards the join: a sidecar generated against an older eval
set would silently mistranslate queries, exactly the kind of quiet wrong
number CLAUDE.md §3 exists to prevent.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets.base import GroundTruthSegment

_MAX_REPORTED_KEYS = 5


class QueryTranslationError(Exception):
    """Raised when the sidecar cannot be joined to the ground truth safely."""


@dataclass(frozen=True)
class TranslatedQuery:
    """One ground-truth query restated in another language.

    `source_caption` is stored so the join can verify the sidecar was built
    against the caption actually being evaluated, not merely against a record
    that happens to share `(video_id, order_index)`.
    """

    video_id: str
    order_index: int
    source_caption: str
    translated_query: str
    # Which model produced this row. Per-row rather than per-file because a
    # sidecar is resumable: a run that starts on Gemini and finishes on a local
    # NMT model after the quota runs out is a MIX, and a report citing one
    # model for all 173 rows would be false provenance (CLAUDE.md §8 requires
    # the model checkpoint to travel with the numbers). Optional so sidecars
    # written before this field existed still load.
    translator_model: str = ""


def translation_key(video_id: str, order_index: int) -> str:
    """The join key. `video_id` alone is not unique — a TRAKE sequence
    contributes several records per video, distinguished by `order_index`."""
    return f"{video_id}#{order_index}"


def key_of(segment: GroundTruthSegment) -> str:
    return translation_key(segment.video_id, segment.order_index)


def models_used(translations: Sequence[TranslatedQuery]) -> list[str]:
    """Distinct `translator_model` values present, sorted. A list with more
    than one entry means the sidecar was produced by several models across
    resumed runs -- the caller should say so rather than quote one model."""
    return sorted({t.translator_model for t in translations if t.translator_model})


def to_payload(
    translations: Sequence[TranslatedQuery],
    *,
    eval_set_path: str,
    eval_set_sha256: str,
    generated_at: str,
) -> dict[str, Any]:
    """Serializable sidecar document.

    `eval_set_sha256` is recorded so a later reader can tell whether the labels
    moved underneath the translations. `translator_models` is derived from the
    rows themselves rather than passed in, so it cannot claim a model that did
    not actually write any row.
    """
    return {
        "eval_set_path": eval_set_path,
        "eval_set_sha256": eval_set_sha256,
        "translator_models": models_used(translations),
        "generated_at": generated_at,
        "translations": [
            {
                "video_id": t.video_id,
                "order_index": t.order_index,
                "source_caption": t.source_caption,
                "translated_query": t.translated_query,
                "translator_model": t.translator_model,
            }
            for t in translations
        ],
    }


def parse_payload(payload: Mapping[str, Any]) -> dict[str, TranslatedQuery]:
    """Sidecar document -> `{key: TranslatedQuery}`.

    Rejects duplicate keys rather than letting the last one win: two rows for
    the same `(video_id, order_index)` means the generator double-wrote, and
    silently picking one would make the run unreproducible.
    """
    rows = payload.get("translations")
    if not isinstance(rows, list):
        raise QueryTranslationError(
            "sidecar has no 'translations' list -- refusing to run with zero translated queries"
        )

    parsed: dict[str, TranslatedQuery] = {}
    for row in rows:
        try:
            translated = TranslatedQuery(
                video_id=str(row["video_id"]),
                order_index=int(row["order_index"]),
                source_caption=str(row["source_caption"]),
                translated_query=str(row["translated_query"]),
                translator_model=str(row.get("translator_model", "")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise QueryTranslationError(f"malformed translation row: {row!r}") from exc

        key = translation_key(translated.video_id, translated.order_index)
        if key in parsed:
            raise QueryTranslationError(f"duplicate translation for {key}")
        parsed[key] = translated
    return parsed


def load_translations(path: Path) -> dict[str, TranslatedQuery]:
    """Reads and parses the sidecar. A missing file raises — an eval run
    configured for translated queries must not silently fall back to the
    Vietnamese ones and report the result under the English channel name
    (CLAUDE.md §7: "thiếu dữ liệu phải fail rõ ràng, không trả rỗng")."""
    if not path.exists():
        raise QueryTranslationError(
            f"{path} not found -- generate it with scripts/translate_eval_queries.py "
            "before enabling query_translation_path in the eval config."
        )
    with path.open(encoding="utf-8") as f:
        return parse_payload(json.load(f))


def _format_keys(keys: Sequence[str]) -> str:
    shown = ", ".join(sorted(keys)[:_MAX_REPORTED_KEYS])
    remainder = len(keys) - _MAX_REPORTED_KEYS
    return shown + (f", ... (+{remainder} more)" if remainder > 0 else "")


def consistency_error(
    ground_truth: Sequence[GroundTruthSegment], translations: Mapping[str, TranslatedQuery]
) -> str | None:
    """Returns a human-readable reason the sidecar cannot be trusted, or None.

    Two independent failure modes, both fatal:
      - a scored query has no translation at all;
      - a translation exists but was generated from a DIFFERENT caption text,
        i.e. the eval set was regenerated after the sidecar was built.
    """
    missing = [key_of(gt) for gt in ground_truth if key_of(gt) not in translations]
    if missing:
        return (
            f"{len(missing)}/{len(ground_truth)} scored quer(ies) have no translation: "
            f"{_format_keys(missing)}"
        )

    stale = [
        key_of(gt) for gt in ground_truth if translations[key_of(gt)].source_caption != gt.caption
    ]
    if stale:
        return (
            f"{len(stale)}/{len(ground_truth)} translation(s) were generated from a different "
            "caption than the one being scored (eval set changed after translation?): "
            f"{_format_keys(stale)}"
        )
    return None


def build_query_transform(
    ground_truth: Sequence[GroundTruthSegment], translations: Mapping[str, TranslatedQuery]
) -> Callable[[GroundTruthSegment], str]:
    """Validates the join up front, then returns the per-query text lookup that
    `eval.run.score_channel` applies instead of `gt.caption`.

    Validation happens once here rather than per query so the run fails before
    burning 173 Milvus searches on a sidecar that was never going to line up.
    """
    error = consistency_error(ground_truth, translations)
    if error is not None:
        raise QueryTranslationError(error)

    def transform(segment: GroundTruthSegment) -> str:
        return translations[key_of(segment)].translated_query

    return transform
