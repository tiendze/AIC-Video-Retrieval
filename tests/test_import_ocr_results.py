"""RED reproducer for pipelines/import_ocr_results.py -- imports the team's
already-run Gemini OCR output (data/ocr_results.jsonl, 873 videos / 221,903
records, one record per keyframe from the same Kaggle embed_out set as
pipelines/import_embed_out.py) into Elasticsearch.

No real ES/Mongo here -- pure functions only, exercised with synthetic rows
and an in-memory keyframe_index built by the test itself. The join key is
frame_idx (this file's `frame_id`) -> (keyframe_id, timestamp), sourced from
already-imported MongoDB keyframe docs, never recomputed from fps here --
recomputing would duplicate pipelines/import_embed_out.py's fps lookup and
risks disagreeing with the timestamp already stored for that same keyframe.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.stores.es_client import OcrDocument
from pipelines.import_ocr_results import (
    build_keyframe_index,
    build_ocr_documents_for_video,
    load_ocr_results,
)


class TestLoadOcrResults:
    def test_groups_records_by_video_id_preserving_order(self, tmp_path: Path) -> None:
        path = tmp_path / "ocr_results.jsonl"
        path.write_text(
            "\n".join(
                json.dumps(row)
                for row in [
                    {"folder": "L21", "video_id": "L21_V001", "frame_id": 3, "ocr_text": "a"},
                    {"folder": "L21", "video_id": "L21_V002", "frame_id": 5, "ocr_text": "b"},
                    {"folder": "L21", "video_id": "L21_V001", "frame_id": 25, "ocr_text": "c"},
                ]
            ),
            encoding="utf-8",
        )

        result = load_ocr_results(path)

        assert list(result.keys()) == ["L21_V001", "L21_V002"]
        assert [r["frame_id"] for r in result["L21_V001"]] == [3, 25]
        assert [r["frame_id"] for r in result["L21_V002"]] == [5]

    def test_skips_blank_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "ocr_results.jsonl"
        path.write_text(
            json.dumps({"folder": "L21", "video_id": "L21_V001", "frame_id": 3, "ocr_text": "a"})
            + "\n\n",
            encoding="utf-8",
        )

        result = load_ocr_results(path)

        assert result == {"L21_V001": [{"folder": "L21", "video_id": "L21_V001", "frame_id": 3, "ocr_text": "a"}]}


class TestBuildKeyframeIndex:
    def test_maps_frame_idx_to_keyframe_id_and_timestamp(self) -> None:
        mongo_docs = [
            {"keyframe_id": "L21_V001_f3", "frame_idx": 3, "timestamp": 0.12},
            {"keyframe_id": "L21_V001_f25", "frame_idx": 25, "timestamp": 1.0},
        ]

        index = build_keyframe_index(mongo_docs)

        assert index == {3: ("L21_V001_f3", 0.12), 25: ("L21_V001_f25", 1.0)}

    def test_empty_docs_returns_empty_index(self) -> None:
        assert build_keyframe_index([]) == {}


class TestBuildOcrDocumentsForVideo:
    def test_matches_records_to_keyframe_id_and_timestamp_via_frame_id(self) -> None:
        records = [
            {"video_id": "L21_V001", "frame_id": 3, "ocr_text": "hello"},
            {"video_id": "L21_V001", "frame_id": 25, "ocr_text": "world"},
        ]
        keyframe_index = {3: ("L21_V001_f3", 0.12), 25: ("L21_V001_f25", 1.0)}

        docs, unmatched = build_ocr_documents_for_video("L21_V001", records, keyframe_index)

        assert docs == [
            OcrDocument(keyframe_id="L21_V001_f3", ocr_text="hello", video_id="L21_V001", timestamp=0.12),
            OcrDocument(keyframe_id="L21_V001_f25", ocr_text="world", video_id="L21_V001", timestamp=1.0),
        ]
        assert unmatched == []

    def test_returns_unmatched_frame_ids_when_missing_from_index(self) -> None:
        # OCR was run on the exact same embed_out keyframe set (CLAUDE.md T2:
        # frame_idx is the submission unit) -- a miss here means that video
        # hasn't been imported into Mongo yet, or the two sets disagree. Either
        # way this must be reported, never silently dropped (CLAUDE.md §7).
        records = [
            {"video_id": "L21_V001", "frame_id": 3, "ocr_text": "hello"},
            {"video_id": "L21_V001", "frame_id": 999, "ocr_text": "orphan"},
        ]
        keyframe_index = {3: ("L21_V001_f3", 0.12)}

        docs, unmatched = build_ocr_documents_for_video("L21_V001", records, keyframe_index)

        assert [d.keyframe_id for d in docs] == ["L21_V001_f3"]
        assert unmatched == [999]

    def test_empty_records_returns_empty(self) -> None:
        docs, unmatched = build_ocr_documents_for_video("L21_V001", [], {})

        assert docs == []
        assert unmatched == []

    def test_empty_index_marks_every_record_unmatched(self) -> None:
        # Whole-video miss (video not imported into Mongo yet at all) is
        # distinguishable from a per-record miss by the caller comparing
        # len(unmatched) to len(records) -- this test locks that shape in.
        records = [{"video_id": "L21_V001", "frame_id": 3, "ocr_text": "hello"}]

        docs, unmatched = build_ocr_documents_for_video("L21_V001", records, {})

        assert docs == []
        assert unmatched == [3]
