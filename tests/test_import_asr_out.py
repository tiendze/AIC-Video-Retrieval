"""RED reproducer for pipelines/import_asr_out.py -- the local side of the
Kaggle offline ASR split (.claude/plans/kaggle-asr-transcribe.plan.md, Nhip A
T1). No GPU, no real ES/Mongo -- file I/O uses tmp_path, hashing/manifest
checks use synthetic bytes written by the test itself.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pipelines.asr import AsrConfig, AsrSegment
from pipelines.import_asr_out import (
    AsrArtifactError,
    build_segments_for_video,
    discover_video_ids,
    load_raw_segment_rows,
    load_run_index,
    verify_manifest,
    verify_video_info_matches,
)
from pipelines.video_io import VideoInfo


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_checkpoint(checkpoint_dir: Path, model_bytes: bytes, vocab_bytes: bytes) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "model.bin").write_bytes(model_bytes)
    (checkpoint_dir / "vocabulary.json").write_bytes(vocab_bytes)


class TestVerifyManifest:
    def test_raises_when_manifest_model_id_differs_from_local_config(self, tmp_path):
        # research-asr-phowhisper-batched.plan.md F1, reopened via the Kaggle
        # path: importing an artifact stamped with a different model_id than
        # this machine's asr.yaml must be blocked, not silently accepted.
        checkpoint_dir = tmp_path / "checkpoint"
        _write_checkpoint(checkpoint_dir, b"model-bytes", b"vocab-bytes")
        manifest = {
            "model_id": "models/OtherModel-ct2",
            "checkpoint_sha256": _sha256_bytes(b"model-bytes"),
            "vocabulary_sha256": _sha256_bytes(b"vocab-bytes"),
        }
        local_config = AsrConfig(model_id="models/PhoWhisper-large-ct2")

        with pytest.raises(AsrArtifactError, match="model_id mismatch"):
            verify_manifest(manifest, local_config, checkpoint_dir)

    def test_raises_when_checkpoint_bytes_differ(self, tmp_path):
        checkpoint_dir = tmp_path / "checkpoint"
        _write_checkpoint(checkpoint_dir, b"model-bytes-DIFFERENT", b"vocab-bytes")
        manifest = {
            "model_id": "models/PhoWhisper-large-ct2",
            "checkpoint_sha256": _sha256_bytes(b"model-bytes"),
            "vocabulary_sha256": _sha256_bytes(b"vocab-bytes"),
        }
        local_config = AsrConfig(model_id="models/PhoWhisper-large-ct2")

        with pytest.raises(AsrArtifactError, match="checkpoint_sha256 mismatch"):
            verify_manifest(manifest, local_config, checkpoint_dir)

    def test_raises_when_vocabulary_bytes_differ(self, tmp_path):
        checkpoint_dir = tmp_path / "checkpoint"
        _write_checkpoint(checkpoint_dir, b"model-bytes", b"vocab-bytes-DIFFERENT")
        manifest = {
            "model_id": "models/PhoWhisper-large-ct2",
            "checkpoint_sha256": _sha256_bytes(b"model-bytes"),
            "vocabulary_sha256": _sha256_bytes(b"vocab-bytes"),
        }
        local_config = AsrConfig(model_id="models/PhoWhisper-large-ct2")

        with pytest.raises(AsrArtifactError, match="vocabulary_sha256 mismatch"):
            verify_manifest(manifest, local_config, checkpoint_dir)

    def test_passes_when_model_id_and_both_checksums_match(self, tmp_path):
        checkpoint_dir = tmp_path / "checkpoint"
        _write_checkpoint(checkpoint_dir, b"model-bytes", b"vocab-bytes")
        manifest = {
            "model_id": "models/PhoWhisper-large-ct2",
            "checkpoint_sha256": _sha256_bytes(b"model-bytes"),
            "vocabulary_sha256": _sha256_bytes(b"vocab-bytes"),
        }
        local_config = AsrConfig(model_id="models/PhoWhisper-large-ct2")

        verify_manifest(manifest, local_config, checkpoint_dir)  # does not raise


class TestDiscoverVideoIds:
    def test_returns_sorted_video_ids_from_jsonl_filenames(self, tmp_path):
        segments_dir = tmp_path / "segments"
        segments_dir.mkdir()
        (segments_dir / "L21_V003.jsonl").write_text("", encoding="utf-8")
        (segments_dir / "L21_V001.jsonl").write_text("", encoding="utf-8")
        (segments_dir / "L21_V001.jsonl.tmp").write_text("", encoding="utf-8")  # ignored

        assert discover_video_ids(segments_dir) == ["L21_V001", "L21_V003"]


class TestLoadRunIndex:
    def test_raises_when_runs_jsonl_missing(self, tmp_path):
        with pytest.raises(AsrArtifactError, match="not found"):
            load_run_index(tmp_path / "runs.jsonl")

    def test_last_line_wins_for_a_retried_video_id(self, tmp_path):
        # Kaggle-side resume contract: a video that failed then succeeded on
        # a later run appends a SECOND, newer line for the same video_id.
        runs_path = tmp_path / "runs.jsonl"
        runs_path.write_text(
            json.dumps({"video_id": "L21_V001", "status": "failed", "error": "boom"})
            + "\n"
            + json.dumps(
                {
                    "video_id": "L21_V001",
                    "status": "ok",
                    "num_raw_segments": 5,
                    "fps": 25.0,
                    "frame_count": 1000,
                }
            )
            + "\n",
            encoding="utf-8",
        )

        index = load_run_index(runs_path)

        assert index["L21_V001"]["status"] == "ok"
        assert index["L21_V001"]["num_raw_segments"] == 5


class TestVerifyVideoInfoMatches:
    def test_passes_within_tolerance(self):
        local = VideoInfo(fps=25.0, frame_count=1000)

        verify_video_info_matches(
            "L21_V001",
            local,
            kaggle_fps=25.005,
            kaggle_frame_count=1001,
            fps_tolerance=0.01,
            frame_count_tolerance=1,
        )  # does not raise

    def test_raises_when_fps_mismatch_exceeds_tolerance(self):
        # F1: the .mp4 on Kaggle may not be byte-identical to the local copy
        # -- a silent frame-window mismatch here is a lost TRAKE window, not
        # a crash, so this must fail loud with BOTH numbers in the message.
        local = VideoInfo(fps=25.0, frame_count=1000)

        with pytest.raises(AsrArtifactError, match="fps mismatch"):
            verify_video_info_matches(
                "L21_V001",
                local,
                kaggle_fps=29.97,
                kaggle_frame_count=1000,
                fps_tolerance=0.01,
                frame_count_tolerance=1,
            )

    def test_raises_when_frame_count_mismatch_exceeds_tolerance(self):
        local = VideoInfo(fps=25.0, frame_count=1000)

        with pytest.raises(AsrArtifactError, match="frame_count mismatch"):
            verify_video_info_matches(
                "L21_V001",
                local,
                kaggle_fps=25.0,
                kaggle_frame_count=1500,
                fps_tolerance=0.01,
                frame_count_tolerance=1,
            )


class TestLoadRawSegmentRows:
    def test_loads_rows_in_order(self, tmp_path):
        path = tmp_path / "L21_V001.jsonl"
        path.write_text(
            json.dumps(
                {
                    "index": 0,
                    "start": 0.0,
                    "end": 1.0,
                    "text": "a",
                    "no_speech_prob": 0.1,
                    "avg_logprob": -0.2,
                    "words": None,
                }
            )
            + "\n"
            + json.dumps(
                {
                    "index": 1,
                    "start": 1.0,
                    "end": 2.0,
                    "text": "b",
                    "no_speech_prob": 0.1,
                    "avg_logprob": -0.2,
                    "words": None,
                }
            )
            + "\n",
            encoding="utf-8",
        )

        rows = load_raw_segment_rows(path)

        assert [r["text"] for r in rows] == ["a", "b"]

    def test_raises_when_index_does_not_match_line_number(self, tmp_path):
        # plan "Hop dong artifact" bullet 1: segment_id derives from this
        # index (pipelines/asr.py's run_asr, f"{video_id}_seg{index}") -- a
        # mismatch here silently desyncs segment_id across re-imports.
        path = tmp_path / "L21_V001.jsonl"
        path.write_text(
            json.dumps(
                {
                    "index": 0,
                    "start": 0.0,
                    "end": 1.0,
                    "text": "a",
                    "no_speech_prob": 0.1,
                    "avg_logprob": -0.2,
                    "words": None,
                }
            )
            + "\n"
            + json.dumps(
                {
                    "index": 5,
                    "start": 1.0,
                    "end": 2.0,
                    "text": "b",
                    "no_speech_prob": 0.1,
                    "avg_logprob": -0.2,
                    "words": None,
                }
            )
            + "\n",
            encoding="utf-8",
        )

        with pytest.raises(AsrArtifactError, match="index 5 != line number 1"):
            load_raw_segment_rows(path)

    def test_video_with_zero_segments_returns_empty_list(self, tmp_path):
        # A silent video is a valid outcome (0 raw segments), not an error --
        # must not raise and must not be confused with "not yet processed".
        path = tmp_path / "L21_V999.jsonl"
        path.write_text("", encoding="utf-8")

        assert load_raw_segment_rows(path) == []


class TestBuildSegmentsForVideo:
    def test_keeps_original_segment_when_resegment_disabled(self):
        rows = [
            {
                "index": 0,
                "start": 0.0,
                "end": 30.0,
                "text": " long segment",
                "no_speech_prob": 0.1,
                "avg_logprob": -0.2,
                "words": None,
            },
        ]

        segments = build_segments_for_video(rows, resegment_min_pause_sec=None)

        assert segments == [
            AsrSegment(
                start=0.0, end=30.0, text=" long segment", no_speech_prob=0.1, avg_logprob=-0.2
            )
        ]

    def test_keeps_original_segment_when_row_has_no_words_even_if_resegment_enabled(self):
        # Graceful degradation: an artifact predating word_timestamps (or a
        # row that legitimately has none) must not block the whole video.
        rows = [
            {
                "index": 0,
                "start": 0.0,
                "end": 30.0,
                "text": " long segment",
                "no_speech_prob": 0.1,
                "avg_logprob": -0.2,
                "words": None,
            },
        ]

        segments = build_segments_for_video(rows, resegment_min_pause_sec=0.5)

        assert len(segments) == 1
        assert segments[0].text == " long segment"

    def test_resegments_by_pause_when_words_present_and_enabled(self):
        rows = [
            {
                "index": 0,
                "start": 0.0,
                "end": 2.5,
                "text": " xin chao ban",
                "no_speech_prob": 0.05,
                "avg_logprob": -0.1,
                "words": [
                    {"start": 0.0, "end": 0.5, "word": " xin", "probability": 0.9},
                    {"start": 0.5, "end": 1.0, "word": " chao", "probability": 0.9},
                    # 1.0s gap >= 0.5s threshold -> new segment
                    {"start": 2.0, "end": 2.5, "word": " ban", "probability": 0.9},
                ],
            },
        ]

        segments = build_segments_for_video(rows, resegment_min_pause_sec=0.5)

        assert len(segments) == 2
        assert segments[0].text == " xin chao"
        assert segments[1].text == " ban"
        # Confidence values carried unchanged from the original row (plan F2:
        # batched mode already coarsens these to one pair per VAD chunk).
        assert segments[0].no_speech_prob == 0.05
        assert segments[1].avg_logprob == -0.1

    def test_multiple_rows_processed_in_order(self):
        rows = [
            {
                "index": 0,
                "start": 0.0,
                "end": 1.0,
                "text": " first",
                "no_speech_prob": 0.1,
                "avg_logprob": -0.2,
                "words": None,
            },
            {
                "index": 1,
                "start": 1.0,
                "end": 2.0,
                "text": " second",
                "no_speech_prob": 0.1,
                "avg_logprob": -0.2,
                "words": None,
            },
        ]

        segments = build_segments_for_video(rows, resegment_min_pause_sec=None)

        assert [s.text for s in segments] == [" first", " second"]

    def test_empty_rows_returns_empty_segments(self):
        assert build_segments_for_video([], resegment_min_pause_sec=0.5) == []
