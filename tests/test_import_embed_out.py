"""RED reproducer for pipelines/import_embed_out.py — giai đoạn 2 of
.claude/plans/kaggle-offline-embed.plan.md: joining a Kaggle-produced
embed_out artifact (vectors/*.npy + *.jsonl) with local metadata
(data/shots.json for shot_id, VideoRef for fps/author/...) into the rows
Milvus + MongoDB expect. All pure-function tests — no Docker, no torch.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from datasets.base import VideoRef
from pipelines.import_embed_out import (
    STATUS_COMPLETE,
    ImportedKeyframe,
    ShotRange,
    VectorsOnlyKeyframe,
    align_metadata_to_video_ids,
    build_imported_keyframes,
    build_vectors_only_rows,
    build_video_document,
    discover_video_ids,
    is_video_already_imported,
    load_shot_index,
    load_video_vectors,
    resolve_shot_id,
    to_keyframe_document,
    to_milvus_vector,
    vectors_only_row_to_milvus_vector,
)


def _write_video_vectors(vectors_dir, video_id: str, rows: list[dict]) -> np.ndarray:
    vectors_dir.mkdir(parents=True, exist_ok=True)
    matrix = np.stack(
        [np.full(4, fill_value=float(i + 1), dtype=np.float32) for i in range(len(rows))]
    )
    np.save(vectors_dir / f"{video_id}.npy", matrix)
    lines = "\n".join(json.dumps(row) for row in rows)
    (vectors_dir / f"{video_id}.jsonl").write_text(lines + "\n", encoding="utf-8")
    return matrix


class TestDiscoverVideoIds:
    def test_returns_sorted_video_ids_with_matching_npy_and_jsonl(self, tmp_path):
        vectors_dir = tmp_path / "vectors"
        _write_video_vectors(
            vectors_dir, "L21_V002", [{"keyframe_id": "x", "video_id": "L21_V002", "frame_idx": 1}]
        )
        _write_video_vectors(
            vectors_dir, "L21_V001", [{"keyframe_id": "y", "video_id": "L21_V001", "frame_idx": 2}]
        )

        assert discover_video_ids(vectors_dir) == ["L21_V001", "L21_V002"]

    def test_raises_when_jsonl_missing_for_an_npy_file(self, tmp_path):
        vectors_dir = tmp_path / "vectors"
        vectors_dir.mkdir(parents=True)
        np.save(vectors_dir / "L21_V001.npy", np.zeros((1, 4), dtype=np.float32))

        with pytest.raises(ValueError, match="mismatched"):
            discover_video_ids(vectors_dir)


class TestAlignMetadataToVideoIds:
    def test_shifts_metadata_when_first_record_is_duplicated(self):
        metadata = {
            "L21_V001": {"watch_url": "https://yt/a", "title": "day 1"},
            "L21_V002": {"watch_url": "https://yt/a", "title": "day 1"},
            "L21_V003": {"watch_url": "https://yt/b", "title": "day 2"},
        }

        aligned = align_metadata_to_video_ids(["L21_V001", "L21_V002", "L21_V003"], metadata)

        assert aligned["L21_V001"] is metadata["L21_V001"]
        assert aligned["L21_V002"] is metadata["L21_V003"]
        assert aligned["L21_V003"] is metadata["L21_V003"]

    def test_keeps_filename_mapping_when_metadata_is_not_shifted(self):
        metadata = {
            "L21_V001": {"watch_url": "https://yt/a", "title": "day 1"},
            "L21_V002": {"watch_url": "https://yt/b", "title": "day 2"},
        }

        aligned = align_metadata_to_video_ids(["L21_V001", "L21_V002"], metadata)

        assert aligned["L21_V001"] is metadata["L21_V001"]
        assert aligned["L21_V002"] is metadata["L21_V002"]


class TestLoadVideoVectors:
    def test_loads_matrix_and_row_aligned_jsonl_rows(self, tmp_path):
        vectors_dir = tmp_path / "vectors"
        rows = [
            {"keyframe_id": "L21_V001_f10", "video_id": "L21_V001", "frame_idx": 10},
            {"keyframe_id": "L21_V001_f20", "video_id": "L21_V001", "frame_idx": 20},
        ]
        matrix = _write_video_vectors(vectors_dir, "L21_V001", rows)

        loaded_vectors, loaded_rows = load_video_vectors(vectors_dir, "L21_V001")

        np.testing.assert_array_equal(loaded_vectors, matrix)
        assert loaded_rows == rows

    def test_raises_on_row_count_mismatch(self, tmp_path):
        vectors_dir = tmp_path / "vectors"
        vectors_dir.mkdir(parents=True)
        np.save(vectors_dir / "L21_V001.npy", np.zeros((2, 4), dtype=np.float32))
        (vectors_dir / "L21_V001.jsonl").write_text(
            json.dumps({"keyframe_id": "a", "video_id": "L21_V001", "frame_idx": 1}) + "\n",
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="2 vector rows but 1 jsonl lines"):
            load_video_vectors(vectors_dir, "L21_V001")


class TestLoadShotIndex:
    def test_builds_sorted_ranges_keyed_by_video_id_without_mp4_suffix(self, tmp_path):
        shots_path = tmp_path / "shots.json"
        shots_path.write_text(
            json.dumps(
                {
                    "videos": [
                        {
                            "video_name": "L21_V001.mp4",
                            "shots": [
                                {"shot_id": 2, "start_frame": 9, "end_frame": 53},
                                {"shot_id": 1, "start_frame": 0, "end_frame": 8},
                            ],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

        index = load_shot_index(shots_path)

        assert index["L21_V001"] == [
            ShotRange(shot_id=1, start_frame=0, end_frame=8),
            ShotRange(shot_id=2, start_frame=9, end_frame=53),
        ]


class TestResolveShotId:
    _RANGES = [
        ShotRange(shot_id=1, start_frame=0, end_frame=8),
        ShotRange(shot_id=2, start_frame=9, end_frame=53),
        ShotRange(shot_id=3, start_frame=54, end_frame=342),
    ]

    @pytest.mark.parametrize(
        "frame_idx,expected_shot_id",
        [
            (0, "L21_V001_shot1"),
            (8, "L21_V001_shot1"),
            (9, "L21_V001_shot2"),
            (342, "L21_V001_shot3"),
        ],
    )
    def test_finds_shot_containing_frame_idx(self, frame_idx, expected_shot_id):
        assert resolve_shot_id(self._RANGES, "L21_V001", frame_idx) == expected_shot_id

    def test_raises_when_frame_idx_outside_every_range(self):
        with pytest.raises(ValueError, match="outside every shot range"):
            resolve_shot_id(self._RANGES, "L21_V001", 9999)


class TestBuildImportedKeyframes:
    _RANGES = [ShotRange(shot_id=1, start_frame=0, end_frame=100)]

    def test_assigns_contiguous_global_index_and_derived_fields(self):
        vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        rows = [
            {"keyframe_id": "L21_V001_f10", "video_id": "L21_V001", "frame_idx": 10},
            {"keyframe_id": "L21_V001_f20", "video_id": "L21_V001", "frame_idx": 20},
        ]

        records = build_imported_keyframes(
            "L21_V001", vectors, rows, self._RANGES, fps=25.0, global_index_start=100, author="ch"
        )

        assert len(records) == 2
        expected = [
            ("L21_V001_f10", "L21_V001_shot1", 10, 100, 0.4),
            ("L21_V001_f20", "L21_V001_shot1", 20, 101, 0.8),
        ]
        for record, (keyframe_id, shot_id, frame_idx, global_index, timestamp) in zip(
            records, expected, strict=True
        ):
            assert record.keyframe_id == keyframe_id
            assert record.video_id == "L21_V001"
            assert record.shot_id == shot_id
            assert record.frame_idx == frame_idx
            assert record.global_index == global_index
            assert record.timestamp == pytest.approx(timestamp)
            assert record.author == "ch"
            assert record.batch == "L21"
        np.testing.assert_array_equal(records[0].embedding, vectors[0])
        np.testing.assert_array_equal(records[1].embedding, vectors[1])

    def test_raises_when_jsonl_row_video_id_does_not_match(self):
        vectors = np.zeros((1, 2), dtype=np.float32)
        rows = [{"keyframe_id": "x", "video_id": "L21_V999", "frame_idx": 5}]

        with pytest.raises(ValueError, match="video_id"):
            build_imported_keyframes(
                "L21_V001", vectors, rows, self._RANGES, fps=25.0, global_index_start=0, author=""
            )

    def test_raises_when_frame_idx_not_strictly_ascending(self):
        vectors = np.zeros((2, 2), dtype=np.float32)
        rows = [
            {"keyframe_id": "L21_V001_f20", "video_id": "L21_V001", "frame_idx": 20},
            {"keyframe_id": "L21_V001_f10", "video_id": "L21_V001", "frame_idx": 10},
        ]

        with pytest.raises(ValueError, match="not ascending"):
            build_imported_keyframes(
                "L21_V001", vectors, rows, self._RANGES, fps=25.0, global_index_start=0, author=""
            )

    def test_raises_when_frame_idx_repeats(self):
        vectors = np.zeros((2, 2), dtype=np.float32)
        rows = [
            {"keyframe_id": "L21_V001_f10", "video_id": "L21_V001", "frame_idx": 10},
            {"keyframe_id": "L21_V001_f10", "video_id": "L21_V001", "frame_idx": 10},
        ]

        with pytest.raises(ValueError, match="not ascending"):
            build_imported_keyframes(
                "L21_V001", vectors, rows, self._RANGES, fps=25.0, global_index_start=0, author=""
            )


class TestToMilvusVectorAndKeyframeDocument:
    _KEYFRAME = ImportedKeyframe(
        keyframe_id="L21_V001_f10",
        video_id="L21_V001",
        shot_id="L21_V001_shot1",
        frame_idx=10,
        global_index=5,
        timestamp=0.4,
        embedding=np.array([1.0, 0.0], dtype=np.float32),
        author="ch",
        batch="L21",
    )

    def test_to_milvus_vector_carries_every_field(self):
        vector = to_milvus_vector(self._KEYFRAME)

        assert vector.keyframe_id == "L21_V001_f10"
        assert vector.video_id == "L21_V001"
        assert vector.global_index == 5
        assert vector.frame_idx == 10
        assert vector.timestamp == pytest.approx(0.4)
        assert vector.author == "ch"
        assert vector.batch == "L21"
        np.testing.assert_array_equal(vector.embedding, self._KEYFRAME.embedding)

    def test_to_keyframe_document_points_image_and_thumbnail_at_same_local_file(self, tmp_path):
        doc = to_keyframe_document(self._KEYFRAME, tmp_path)

        expected_path = str(tmp_path / "L21" / "L21_V001" / "L21_V001_10.jpg")
        assert doc.image_path == expected_path
        assert doc.thumbnail_path == expected_path
        assert doc.shot_id == "L21_V001_shot1"
        assert doc.global_index == 5


class TestIsVideoAlreadyImported:
    def test_false_when_no_document_exists(self):
        assert is_video_already_imported(None) is False

    def test_false_when_status_is_not_complete(self):
        assert is_video_already_imported({"ingest_status": "failed"}) is False

    def test_true_when_status_is_complete(self):
        assert is_video_already_imported({"ingest_status": STATUS_COMPLETE}) is True


class TestBuildVideoDocument:
    def test_computes_s_v_e_v_from_keyframe_global_indices(self):
        video = VideoRef(
            video_id="L21_V001",
            source_path="unused",
            duration_sec=12.5,
            category="60 Giay",
            title="t",
            publish_date="01/08/2024",
            watch_url="https://x",
            channel_id="UCx",
        )
        keyframes = [
            ImportedKeyframe(
                keyframe_id="a",
                video_id="L21_V001",
                shot_id="s",
                frame_idx=1,
                global_index=10,
                timestamp=0.1,
                embedding=np.zeros(2, dtype=np.float32),
                author="60 Giay",
                batch="L21",
            ),
            ImportedKeyframe(
                keyframe_id="b",
                video_id="L21_V001",
                shot_id="s",
                frame_idx=2,
                global_index=12,
                timestamp=0.2,
                embedding=np.zeros(2, dtype=np.float32),
                author="60 Giay",
                batch="L21",
            ),
        ]

        doc = build_video_document(video, keyframes, fps=25.0)

        assert doc.s_v == 10
        assert doc.e_v == 12
        assert doc.duration == 12
        assert doc.fps == pytest.approx(25.0)
        assert doc.num_keyframes == 2
        assert doc.ingest_status == STATUS_COMPLETE
        assert doc.author == "60 Giay"
        assert doc.batch == "L21"
        assert doc.title == "t"
        assert doc.publish_date == "01/08/2024"
        assert doc.watch_url == "https://x"
        assert doc.channel_id == "UCx"


class TestBuildVectorsOnlyRows:
    def test_joins_artifact_vectors_with_mongo_metadata(self):
        vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        jsonl_rows = [
            {"keyframe_id": "L21_V001_f10", "video_id": "L21_V001", "frame_idx": 10},
            {"keyframe_id": "L21_V001_f20", "video_id": "L21_V001", "frame_idx": 20},
        ]
        mongo_keyframe_docs = [
            {"keyframe_id": "L21_V001_f10", "global_index": 100, "frame_idx": 10, "timestamp": 0.4},
            {"keyframe_id": "L21_V001_f20", "global_index": 101, "frame_idx": 20, "timestamp": 0.8},
        ]

        rows = build_vectors_only_rows(
            "L21_V001", vectors, jsonl_rows, mongo_keyframe_docs, author="ch"
        )

        assert len(rows) == 2
        assert rows[0].keyframe_id == "L21_V001_f10"
        assert rows[0].video_id == "L21_V001"
        assert rows[0].global_index == 100
        assert rows[0].frame_idx == 10
        assert rows[0].timestamp == pytest.approx(0.4)
        assert rows[0].author == "ch"
        assert rows[0].batch == "L21"
        np.testing.assert_array_equal(rows[0].embedding, vectors[0])
        np.testing.assert_array_equal(rows[1].embedding, vectors[1])
        assert rows[1].global_index == 101

    def test_raises_when_artifact_has_keyframe_id_missing_from_mongo(self):
        vectors = np.zeros((1, 2), dtype=np.float32)
        jsonl_rows = [{"keyframe_id": "L21_V001_f99", "video_id": "L21_V001", "frame_idx": 99}]
        mongo_keyframe_docs = [
            {"keyframe_id": "L21_V001_f10", "global_index": 0, "frame_idx": 10, "timestamp": 0.4}
        ]

        with pytest.raises(ValueError, match="keyframe_id set mismatch"):
            build_vectors_only_rows("L21_V001", vectors, jsonl_rows, mongo_keyframe_docs, author="")

    def test_raises_when_mongo_has_keyframe_id_missing_from_artifact(self):
        vectors = np.zeros((1, 2), dtype=np.float32)
        jsonl_rows = [{"keyframe_id": "L21_V001_f10", "video_id": "L21_V001", "frame_idx": 10}]
        mongo_keyframe_docs = [
            {"keyframe_id": "L21_V001_f10", "global_index": 0, "frame_idx": 10, "timestamp": 0.4},
            {"keyframe_id": "L21_V001_f20", "global_index": 1, "frame_idx": 20, "timestamp": 0.8},
        ]

        with pytest.raises(ValueError, match="keyframe_id set mismatch"):
            build_vectors_only_rows("L21_V001", vectors, jsonl_rows, mongo_keyframe_docs, author="")


class TestVectorsOnlyRowToMilvusVector:
    def test_carries_every_field(self):
        row = VectorsOnlyKeyframe(
            keyframe_id="L21_V001_f10",
            video_id="L21_V001",
            global_index=5,
            frame_idx=10,
            timestamp=0.4,
            embedding=np.array([1.0, 0.0], dtype=np.float32),
            author="ch",
            batch="L21",
        )

        vector = vectors_only_row_to_milvus_vector(row)

        assert vector.keyframe_id == "L21_V001_f10"
        assert vector.video_id == "L21_V001"
        assert vector.global_index == 5
        assert vector.frame_idx == 10
        assert vector.timestamp == pytest.approx(0.4)
        assert vector.author == "ch"
        assert vector.batch == "L21"
        np.testing.assert_array_equal(vector.embedding, row.embedding)
