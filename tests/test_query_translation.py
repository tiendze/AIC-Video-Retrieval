"""Behaviour lock for eval/query_translation.py.

The point of the sidecar is that it joins to the ground truth WITHOUT touching
the label set, so the tests that matter most are the refusal ones: a sidecar
that is incomplete, stale, or duplicated must stop the run loudly rather than
let `semantic@en` be scored on a mix of English and Vietnamese queries and
quietly report a meaningless number.
"""

from __future__ import annotations

import json

import pytest

from datasets.base import GroundTruthSegment
from eval.query_translation import (
    QueryTranslationError,
    TranslatedQuery,
    build_query_transform,
    consistency_error,
    key_of,
    load_translations,
    models_used,
    parse_payload,
    to_payload,
    translation_key,
)
from scripts.translate_eval_queries import strip_target_prefix

_VI_ONE = "Một con lừa mẹ và lừa con đang đứng cạnh nhau."
_VI_TWO = "Cận cảnh các tang vật gồm tiền mặt và một chiếc cốc màu đỏ."
_EN_ONE = "A mother donkey and a foal standing side by side."
_EN_TWO = "Close-up of evidence including cash and a red cup."


def _segment(video_id: str, order_index: int, caption: str) -> GroundTruthSegment:
    return GroundTruthSegment(
        video_id=video_id,
        start_sec=1.0,
        end_sec=2.0,
        order_index=order_index,
        caption=caption,
    )


def _translation(
    video_id: str,
    order_index: int,
    source: str,
    target: str,
    model: str = "VietAI/envit5-translation",
) -> TranslatedQuery:
    return TranslatedQuery(
        video_id=video_id,
        order_index=order_index,
        source_caption=source,
        translated_query=target,
        translator_model=model,
    )


def _ground_truth() -> list[GroundTruthSegment]:
    return [_segment("L21_V001", 0, _VI_ONE), _segment("L21_V001", 1, _VI_TWO)]


def _translations() -> dict[str, TranslatedQuery]:
    return {
        "L21_V001#0": _translation("L21_V001", 0, _VI_ONE, _EN_ONE),
        "L21_V001#1": _translation("L21_V001", 1, _VI_TWO, _EN_TWO),
    }


def _written_sidecar(tmp_path):
    path = tmp_path / "queries_en.json"
    path.write_text(
        json.dumps(
            to_payload(
                list(_translations().values()),
                eval_set_path="configs/eval/aic_eval_v1.json",
                eval_set_sha256="008f368e",
                generated_at="2026-08-10T16:10:49Z",
            ),
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


class TestTranslationKey:
    def test_key_includes_order_index_so_trake_rows_do_not_collide(self) -> None:
        # A TRAKE sequence contributes several records per video; keying on
        # video_id alone would map all of them to one translation.
        first = key_of(_segment("L21_V001", 0, _VI_ONE))
        second = key_of(_segment("L21_V001", 1, _VI_TWO))

        assert first != second
        assert first == translation_key("L21_V001", 0)


class TestPayloadRoundTrip:
    def test_to_payload_then_parse_payload_preserves_every_field(self) -> None:
        payload = to_payload(
            list(_translations().values()),
            eval_set_path="configs/eval/aic_eval_v1.json",
            eval_set_sha256="008f368e",
            generated_at="2026-08-10T16:10:49Z",
        )

        parsed = parse_payload(json.loads(json.dumps(payload)))

        assert parsed["L21_V001#0"].translated_query == _EN_ONE
        assert parsed["L21_V001#0"].source_caption == _VI_ONE
        assert parsed["L21_V001#1"].translated_query == _EN_TWO

    def test_payload_records_the_eval_set_hash_it_was_built_against(self) -> None:
        payload = to_payload(
            list(_translations().values()),
            eval_set_path="configs/eval/aic_eval_v1.json",
            eval_set_sha256="008f368e",
            generated_at="2026-08-10T16:10:49Z",
        )

        assert payload["eval_set_sha256"] == "008f368e"
        assert payload["generated_at"] == "2026-08-10T16:10:49Z"

    def test_rejects_duplicate_rows_instead_of_last_one_wins(self) -> None:
        rows = [
            {
                "video_id": "L21_V001",
                "order_index": 0,
                "source_caption": _VI_ONE,
                "translated_query": _EN_ONE,
            },
            {
                "video_id": "L21_V001",
                "order_index": 0,
                "source_caption": _VI_ONE,
                "translated_query": "a different translation",
            },
        ]

        with pytest.raises(QueryTranslationError, match="duplicate translation"):
            parse_payload({"translations": rows})

    def test_rejects_payload_without_a_translations_list(self) -> None:
        with pytest.raises(QueryTranslationError, match="no 'translations' list"):
            parse_payload({"generated_at": "2026-08-10T16:10:49Z"})

    def test_rejects_a_row_missing_a_required_field(self) -> None:
        with pytest.raises(QueryTranslationError, match="malformed translation row"):
            parse_payload({"translations": [{"video_id": "L21_V001", "order_index": 0}]})


class TestTranslatorProvenance:
    def test_rows_carry_the_model_that_wrote_them(self) -> None:
        payload = to_payload(
            list(_translations().values()),
            eval_set_path="configs/eval/aic_eval_v1.json",
            eval_set_sha256="008f368e",
            generated_at="2026-08-10T16:10:49Z",
        )

        assert payload["translations"][0]["translator_model"] == "VietAI/envit5-translation"
        assert payload["translator_models"] == ["VietAI/envit5-translation"]

    def test_a_sidecar_resumed_on_another_model_reports_both(self) -> None:
        # Real case: Gemini quota ran out mid-run and the rest was translated
        # offline. The document must not claim a single translator.
        mixed = [
            _translation("L21_V001", 0, _VI_ONE, _EN_ONE, model="gemini-3.6-flash"),
            _translation("L21_V001", 1, _VI_TWO, _EN_TWO, model="VietAI/envit5-translation"),
        ]

        assert models_used(mixed) == ["VietAI/envit5-translation", "gemini-3.6-flash"]

    def test_sidecars_written_before_provenance_existed_still_load(self) -> None:
        rows = [
            {
                "video_id": "L21_V001",
                "order_index": 0,
                "source_caption": _VI_ONE,
                "translated_query": _EN_ONE,
            }
        ]

        parsed = parse_payload({"translations": rows})

        assert parsed["L21_V001#0"].translator_model == ""


class TestStripTargetPrefix:
    def test_strips_the_language_tag_envit5_echoes_back(self) -> None:
        assert strip_target_prefix("en: A mother donkey and a foal.", "en: ") == (
            "A mother donkey and a foal."
        )

    def test_leaves_text_alone_when_the_tag_is_absent(self) -> None:
        assert strip_target_prefix("A mother donkey and a foal.", "en: ") == (
            "A mother donkey and a foal."
        )

    def test_only_strips_at_the_start_not_mid_sentence(self) -> None:
        assert strip_target_prefix("The sign reads en: closed", "en: ") == (
            "The sign reads en: closed"
        )

    def test_models_needing_no_tag_pass_through(self) -> None:
        # VinAI-Translate emits no language tag -> configured with prefix "".
        assert strip_target_prefix("  A mother donkey.  ", "") == "A mother donkey."


class TestLoadTranslations:
    def test_missing_file_raises_instead_of_returning_empty(self, tmp_path) -> None:
        # CLAUDE.md §7: falling back to {} here would score the Vietnamese
        # captions and label the result `semantic@en`.
        with pytest.raises(QueryTranslationError, match="not found"):
            load_translations(tmp_path / "__no_such_sidecar__.json")

    def test_reads_a_written_sidecar(self, tmp_path) -> None:
        parsed = load_translations(_written_sidecar(tmp_path))

        assert parsed["L21_V001#0"].translated_query == _EN_ONE
        assert parsed["L21_V001#1"].source_caption == _VI_TWO


class TestConsistencyError:
    def test_returns_none_when_every_query_has_a_matching_translation(self) -> None:
        assert consistency_error(_ground_truth(), _translations()) is None

    def test_flags_a_query_with_no_translation(self) -> None:
        partial = {"L21_V001#0": _translations()["L21_V001#0"]}

        error = consistency_error(_ground_truth(), partial)

        assert error is not None
        assert "no translation" in error
        assert "L21_V001#1" in error

    def test_flags_a_translation_built_from_a_different_caption(self) -> None:
        stale = dict(_translations())
        stale["L21_V001#1"] = _translation("L21_V001", 1, "một caption cũ đã bị thay", _EN_TWO)

        error = consistency_error(_ground_truth(), stale)

        assert error is not None
        assert "different caption" in error


class TestBuildQueryTransform:
    def test_transform_returns_the_english_query_for_each_segment(self) -> None:
        transform = build_query_transform(_ground_truth(), _translations())

        assert transform(_segment("L21_V001", 0, _VI_ONE)) == _EN_ONE
        assert transform(_segment("L21_V001", 1, _VI_TWO)) == _EN_TWO

    def test_never_falls_back_to_the_vietnamese_caption(self) -> None:
        transform = build_query_transform(_ground_truth(), _translations())

        assert transform(_segment("L21_V001", 0, _VI_ONE)) != _VI_ONE

    def test_refuses_to_build_from_an_incomplete_sidecar(self) -> None:
        partial = {"L21_V001#0": _translations()["L21_V001#0"]}

        with pytest.raises(QueryTranslationError, match="no translation"):
            build_query_transform(_ground_truth(), partial)

    def test_refuses_to_build_from_a_stale_sidecar(self) -> None:
        stale = dict(_translations())
        stale["L21_V001#0"] = _translation("L21_V001", 0, "caption đã đổi", _EN_ONE)

        with pytest.raises(QueryTranslationError, match="different caption"):
            build_query_transform(_ground_truth(), stale)
