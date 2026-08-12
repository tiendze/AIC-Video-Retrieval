"""RED reproducer for eval/build_eval_set.py (P0.4', .claude/plans/
p0-4-build-eval-set.plan.md).

Sampling/allocation/filter/freeze functions are pure -- tested directly
against synthetic ShotRef universes, no Mongo/Gemini needed. Gemini and
MongoDB are Protocol-faked (tests/test_ocr.py:50-67 pattern) for the one
orchestration-level test that exercises `_generate_candidates`.
"""

from __future__ import annotations

import json
import random

import pytest
from PIL import Image

from datasets.aic2026 import AIC2026Adapter
from datasets.base import GroundTruthSegment, VideoRef
from eval.build_eval_set import (
    BuildEvalSetConfig,
    EvalSetBuildError,
    GeminiCaptionGenerator,
    ShotRef,
    VideoMeta,
    _generate_candidates,
    _load_config,
    _write_candidates_doc,
    allocate_by_sqrt,
    build_shot_refs,
    build_video_meta,
    compute_quality_report,
    extract_trake_sequences,
    flag_generic_captions,
    freeze_records,
    load_shots_json,
    sample_kis_shots,
    sample_trake_sequences,
    select_middle_keyframe,
    stratum_key,
)

_TEST_STOPWORDS = ("một", "người", "là", "và", "có", "đang", "này", "của", "trước", "buổi")


def _shot(
    video_id: str, shot_id: int, start_sec: float, end_sec: float, batch: str, author: str
) -> ShotRef:
    return ShotRef(
        video_id=video_id,
        shot_id=shot_id,
        start_sec=start_sec,
        end_sec=end_sec,
        batch=batch,
        author=author,
    )


def _multi_stratum_shots_by_video() -> dict[str, list[ShotRef]]:
    """3 strata (batch, author) x 5 videos x 5 shots = 75 shots, all >= the
    default 1.0s min duration -- big enough to exercise stratified sampling
    without being slow."""
    shots_by_video: dict[str, list[ShotRef]] = {}
    for batch, author in (("L21", "Author A"), ("L22", "Author B"), ("L23", "Author C")):
        for v in range(5):
            video_id = f"{batch}_V{v:03d}"
            shots_by_video[video_id] = [
                _shot(video_id, i, float(i) * 2.0, float(i) * 2.0 + 1.5, batch, author)
                for i in range(1, 6)
            ]
    return shots_by_video


class _FakeAdapter:
    """Duck-types `AIC2026Adapter.iter_videos` -- `build_video_meta` only
    ever calls that one method, so a real adapter (needing dataset files on
    disk) isn't needed."""

    def __init__(self, video_refs):
        self._video_refs = video_refs

    def iter_videos(self, split):
        return iter(self._video_refs)


class TestDataLoading:
    def test_load_shots_json_parses_video_name_and_shots(self, tmp_path):
        shots_path = tmp_path / "shots.json"
        shots_path.write_text(
            '{"videos": [{"video_name": "L21_V001.mp4", "shots": '
            '[{"shot_id": 1, "start_frame": 0, "end_frame": 8, '
            '"start_sec": 0.0, "end_sec": 0.3}]}]}',
            encoding="utf-8",
        )

        result = load_shots_json(shots_path)

        assert list(result) == ["L21_V001"]
        assert result["L21_V001"][0]["shot_id"] == 1

    def test_build_video_meta_maps_video_id_to_metadata(self):
        adapter = _FakeAdapter(
            [
                VideoRef(
                    video_id="L21_V001",
                    source_path="x.mp4",
                    duration_sec=10.0,
                    category="Author A",
                    title="Tin tuc",
                )
            ]
        )

        meta = build_video_meta(adapter, "all")

        assert meta["L21_V001"] == VideoMeta(batch="L21", author="Author A", title="Tin tuc")

    def test_build_video_meta_falls_back_to_unknown_author(self):
        adapter = _FakeAdapter(
            [VideoRef(video_id="L21_V001", source_path="x.mp4", duration_sec=10.0, category=None)]
        )

        meta = build_video_meta(adapter, "all")

        assert meta["L21_V001"].author == "unknown"

    def test_build_shot_refs_skips_videos_outside_split(self):
        shots_by_video_name = {
            "L21_V001": [{"shot_id": 1, "start_sec": 0.0, "end_sec": 1.0}],
            "L25_V001": [{"shot_id": 1, "start_sec": 0.0, "end_sec": 1.0}],  # not in video_meta
        }
        video_meta = {"L21_V001": VideoMeta(batch="L21", author="A", title="t")}

        result = build_shot_refs(shots_by_video_name, video_meta)

        assert list(result) == ["L21_V001"]
        assert result["L21_V001"][0].batch == "L21"

    def test_build_shot_refs_sorts_by_shot_id(self):
        shots_by_video_name = {
            "L21_V001": [
                {"shot_id": 2, "start_sec": 1.0, "end_sec": 2.0},
                {"shot_id": 1, "start_sec": 0.0, "end_sec": 1.0},
            ]
        }
        video_meta = {"L21_V001": VideoMeta(batch="L21", author="A", title="t")}

        result = build_shot_refs(shots_by_video_name, video_meta)

        assert [s.shot_id for s in result["L21_V001"]] == [1, 2]

    def test_build_shot_refs_skips_video_with_no_shots_on_record(self):
        video_meta = {"L21_V001": VideoMeta(batch="L21", author="A", title="t")}

        result = build_shot_refs({}, video_meta)  # data/shots.json has no entry for this video

        assert result == {}


class TestAllocateBySqrt:
    def test_sqrt_allocation_gives_small_stratum_more_than_proportional(self):
        sizes = {("big", "x"): 10_000, ("small", "y"): 100}

        quotas = allocate_by_sqrt(sizes, 110)

        proportional_small = 110 * 100 / 10_100  # ~1.09 under a pure-proportional split
        assert quotas[("small", "y")] > proportional_small + 5
        assert sum(quotas.values()) == 110

    def test_allocation_sums_to_requested_total(self):
        sizes = {("L21", "A"): 500, ("L22", "A"): 300, ("L23", "B"): 50}

        quotas = allocate_by_sqrt(sizes, 100)

        assert sum(quotas.values()) == 100
        assert set(quotas) == set(sizes)

    def test_allocation_never_exceeds_total_when_total_is_zero(self):
        assert allocate_by_sqrt({("a", "b"): 10}, 0) == {("a", "b"): 0}

    def test_allocation_is_all_zero_when_every_stratum_is_empty(self):
        assert allocate_by_sqrt({("a", "b"): 0, ("c", "d"): 0}, 100) == {
            ("a", "b"): 0,
            ("c", "d"): 0,
        }


class TestSampleKisShots:
    def test_eval_set_sampling_is_deterministic_given_seed(self):
        shots_by_video = _multi_stratum_shots_by_video()
        config = BuildEvalSetConfig(kis_sample_size=10)

        first = sample_kis_shots(shots_by_video, config, random.Random(config.seed))
        second = sample_kis_shots(shots_by_video, config, random.Random(config.seed))

        assert first == second

    def test_eval_set_is_stratified_across_batch_and_author(self):
        shots_by_video: dict[str, list[ShotRef]] = {}
        # Large stratum: 20 videos x 5 shots.
        for v in range(20):
            video_id = f"L21_V{v:03d}"
            shots_by_video[video_id] = [
                _shot(video_id, i, float(i) * 2.0, float(i) * 2.0 + 1.5, "L21", "Author A")
                for i in range(1, 6)
            ]
        # Small stratum: 1 video x 5 shots.
        shots_by_video["L23_V001"] = [
            _shot("L23_V001", i, float(i) * 2.0, float(i) * 2.0 + 1.5, "L23", "Author B")
            for i in range(1, 6)
        ]
        config = BuildEvalSetConfig(kis_sample_size=20)

        sampled = sample_kis_shots(shots_by_video, config, random.Random(42))

        strata_present = {stratum_key(s) for s in sampled}
        assert ("L21", "Author A") in strata_present
        assert ("L23", "Author B") in strata_present


class TestSampleTrakeSequences:
    def test_trake_sample_yields_contiguous_shot_ids(self):
        shots = [_shot("L21_V001", i, float(i), float(i) + 1.0, "L21", "A") for i in range(1, 11)]
        shots_by_video = {"L21_V001": shots}
        config = BuildEvalSetConfig(trake_sequence_count=1, trake_sequence_length=5)

        sequences = sample_trake_sequences(shots_by_video, config, random.Random(7))

        assert len(sequences) == 1
        shot_ids = [s.shot_id for s in sequences[0]]
        assert shot_ids == list(range(shot_ids[0], shot_ids[0] + 5))

    def test_skips_videos_shorter_than_sequence_length(self):
        shots_by_video = {"L21_V001": [_shot("L21_V001", 1, 0.0, 1.0, "L21", "A")]}
        config = BuildEvalSetConfig(trake_sequence_count=5, trake_sequence_length=5)

        sequences = sample_trake_sequences(shots_by_video, config, random.Random(1))

        assert sequences == []


class TestSelectMiddleKeyframe:
    def test_middle_keyframe_is_picked_by_frame_idx_order(self):
        docs = [
            {"frame_idx": 50, "keyframe_id": "k4"},
            {"frame_idx": 10, "keyframe_id": "k1"},
            {"frame_idx": 30, "keyframe_id": "k3"},
            {"frame_idx": 20, "keyframe_id": "k2"},
        ]

        middle = select_middle_keyframe(docs)

        assert middle["keyframe_id"] == "k3"  # sorted 10,20,30,50 -> index len//2=2 -> 30

    def test_raises_on_empty_keyframe_list(self):
        with pytest.raises(EvalSetBuildError, match="empty"):
            select_middle_keyframe([])


class _FakeCaptionGenerator:
    def __init__(self, fail_first_n: int = 0):
        self.calls = 0
        self._fail_first_n = fail_first_n

    def generate(self, image, title):
        self.calls += 1
        if self.calls <= self._fail_first_n:
            raise RuntimeError("simulated caption failure")
        return "một caption đầy đủ nội dung cụ thể cho shot này"


class _FakeKeyframeSource:
    def __init__(self, keyframes_by_video):
        self._keyframes_by_video = keyframes_by_video

    async def get_keyframes_for_video(self, video_id):
        return self._keyframes_by_video.get(video_id, [])


class TestGenerateCandidatesReportsGaps:
    async def test_shot_without_keyframes_is_reported_not_silently_skipped(self, tmp_path):
        shot_with_kf = _shot("L21_V001", 1, 0.0, 2.0, "L21", "A")
        shot_without_kf = _shot("L21_V001", 2, 2.0, 4.0, "L21", "A")
        shots_by_video = {"L21_V001": [shot_with_kf, shot_without_kf]}
        video_meta = {"L21_V001": VideoMeta(batch="L21", author="A", title="Tieu de")}

        image_path = tmp_path / "frame.jpg"
        Image.new("RGB", (10, 10)).save(image_path)
        keyframes_by_video = {
            "L21_V001": [
                {
                    "shot_id": "L21_V001_shot1",
                    "frame_idx": 5,
                    "keyframe_id": "k1",
                    "image_path": str(image_path),
                }
            ]
        }
        config = BuildEvalSetConfig(kis_sample_size=2, trake_sequence_count=0)

        result = await _generate_candidates(
            config,
            shots_by_video=shots_by_video,
            video_meta=video_meta,
            caption_generator=_FakeCaptionGenerator(),
            keyframe_source=_FakeKeyframeSource(keyframes_by_video),
        )

        assert result["skipped_no_keyframe"] == ["L21_V001_shot2"]
        assert len(result["records"]) == 1
        assert result["records"][0]["shot_id"] == 1

    async def test_caption_generation_failure_is_reported_not_silently_skipped(self, tmp_path):
        shot = _shot("L21_V001", 1, 0.0, 2.0, "L21", "A")
        shots_by_video = {"L21_V001": [shot]}
        video_meta = {"L21_V001": VideoMeta(batch="L21", author="A", title="Tieu de")}

        image_path = tmp_path / "frame.jpg"
        Image.new("RGB", (10, 10)).save(image_path)
        keyframes_by_video = {
            "L21_V001": [
                {
                    "shot_id": "L21_V001_shot1",
                    "frame_idx": 5,
                    "keyframe_id": "k1",
                    "image_path": str(image_path),
                }
            ]
        }
        config = BuildEvalSetConfig(kis_sample_size=1, trake_sequence_count=0)

        result = await _generate_candidates(
            config,
            shots_by_video=shots_by_video,
            video_meta=video_meta,
            caption_generator=_FakeCaptionGenerator(fail_first_n=1),
            keyframe_source=_FakeKeyframeSource(keyframes_by_video),
        )

        assert result["failed"] == ["L21_V001_shot1"]
        assert result["records"] == []

    async def test_limit_shots_truncates_combined_kis_and_trake_tasks(self, tmp_path):
        shots = [_shot("L21_V001", i, float(i), float(i) + 1.0, "L21", "A") for i in range(1, 11)]
        shots_by_video = {"L21_V001": shots}
        video_meta = {"L21_V001": VideoMeta(batch="L21", author="A", title="Tieu de")}

        image_path = tmp_path / "frame.jpg"
        Image.new("RGB", (10, 10)).save(image_path)
        keyframes_by_video = {
            "L21_V001": [
                {
                    "shot_id": f"L21_V001_shot{i}",
                    "frame_idx": i,
                    "keyframe_id": f"k{i}",
                    "image_path": str(image_path),
                }
                for i in range(1, 11)
            ]
        }
        # trake_sequence_count=1 exercises the "extend tasks with a trake
        # sequence" branch; limit_shots=1 exercises the truncation branch.
        config = BuildEvalSetConfig(
            kis_sample_size=10, trake_sequence_count=1, trake_sequence_length=5
        )

        result = await _generate_candidates(
            config,
            shots_by_video=shots_by_video,
            video_meta=video_meta,
            caption_generator=_FakeCaptionGenerator(),
            keyframe_source=_FakeKeyframeSource(keyframes_by_video),
            limit_shots=1,
        )

        assert len(result["records"]) == 1

    async def test_resume_skips_shots_already_in_existing_records(self, tmp_path):
        """A prior interrupted run already generated shot 1's caption --
        re-running with `existing_records` must not call Gemini for it again
        (this is what makes resuming after a Gemini quota error not re-bill
        already-paid-for shots)."""
        shots = [
            _shot("L21_V001", 1, 0.0, 1.0, "L21", "A"),
            _shot("L21_V001", 2, 1.0, 2.0, "L21", "A"),
        ]
        shots_by_video = {"L21_V001": shots}
        video_meta = {"L21_V001": VideoMeta(batch="L21", author="A", title="Tieu de")}

        image_path = tmp_path / "frame.jpg"
        Image.new("RGB", (10, 10)).save(image_path)
        keyframes_by_video = {
            "L21_V001": [
                {
                    "shot_id": f"L21_V001_shot{i}",
                    "frame_idx": i,
                    "keyframe_id": f"k{i}",
                    "image_path": str(image_path),
                }
                for i in (1, 2)
            ]
        }
        existing_records = [
            {
                "video_id": "L21_V001",
                "shot_id": 1,
                "start_sec": 0.0,
                "end_sec": 1.0,
                "order_index": 1,
                "caption": "caption từ lần chạy trước",
                "task": "kis",
                "keyframe_id": "k1",
                "image_path": str(image_path),
                "batch": "L21",
                "author": "A",
                "auto_filter": "kept",
                "approved": None,
            }
        ]
        config = BuildEvalSetConfig(kis_sample_size=2, trake_sequence_count=0)
        generator = _FakeCaptionGenerator()

        result = await _generate_candidates(
            config,
            shots_by_video=shots_by_video,
            video_meta=video_meta,
            caption_generator=generator,
            keyframe_source=_FakeKeyframeSource(keyframes_by_video),
            existing_records=existing_records,
        )

        assert generator.calls == 1  # only shot 2 -- shot 1 was NOT re-billed
        assert len(result["records"]) == 2
        by_shot = {r["shot_id"]: r for r in result["records"]}
        assert by_shot[1]["caption"] == "caption từ lần chạy trước"
        assert by_shot[2]["caption"] != "caption từ lần chạy trước"

    async def test_on_progress_checkpoints_after_each_video(self, tmp_path):
        shots_by_video = {
            "L21_V001": [_shot("L21_V001", 1, 0.0, 1.0, "L21", "A")],
            "L21_V002": [_shot("L21_V002", 1, 0.0, 1.0, "L21", "A")],
        }
        video_meta = {
            "L21_V001": VideoMeta(batch="L21", author="A", title="t1"),
            "L21_V002": VideoMeta(batch="L21", author="A", title="t2"),
        }
        image_path = tmp_path / "frame.jpg"
        Image.new("RGB", (10, 10)).save(image_path)
        keyframes_by_video = {
            video_id: [
                {
                    "shot_id": f"{video_id}_shot1",
                    "frame_idx": 1,
                    "keyframe_id": "k1",
                    "image_path": str(image_path),
                }
            ]
            for video_id in shots_by_video
        }
        config = BuildEvalSetConfig(kis_sample_size=2, trake_sequence_count=0)
        checkpoints: list[int] = []

        await _generate_candidates(
            config,
            shots_by_video=shots_by_video,
            video_meta=video_meta,
            caption_generator=_FakeCaptionGenerator(),
            keyframe_source=_FakeKeyframeSource(keyframes_by_video),
            on_progress=lambda records, skipped, failed: checkpoints.append(len(records)),
        )

        # 2 videos, 1 shot each -> checkpoints fire per shot, records accumulate: 1, then 2.
        assert checkpoints == [1, 2]

    async def test_on_progress_checkpoints_within_a_single_multi_shot_video(self, tmp_path):
        """Regression test for a real incident (2026-08-09): a video with
        several shots (e.g. a TRAKE sequence) hit a Gemini rate-limit error
        partway through -- checkpointing only after the whole video finished
        meant every caption already generated for that video was lost. Must
        checkpoint after EACH shot, not each video."""
        shots = [_shot("L21_V001", i, float(i), float(i) + 1.0, "L21", "A") for i in range(1, 4)]
        shots_by_video = {"L21_V001": shots}
        video_meta = {"L21_V001": VideoMeta(batch="L21", author="A", title="t1")}
        image_path = tmp_path / "frame.jpg"
        Image.new("RGB", (10, 10)).save(image_path)
        keyframes_by_video = {
            "L21_V001": [
                {
                    "shot_id": f"L21_V001_shot{i}",
                    "frame_idx": i,
                    "keyframe_id": f"k{i}",
                    "image_path": str(image_path),
                }
                for i in range(1, 4)
            ]
        }
        config = BuildEvalSetConfig(kis_sample_size=3, trake_sequence_count=0)
        checkpoints: list[int] = []

        await _generate_candidates(
            config,
            shots_by_video=shots_by_video,
            video_meta=video_meta,
            caption_generator=_FakeCaptionGenerator(),
            keyframe_source=_FakeKeyframeSource(keyframes_by_video),
            on_progress=lambda records, skipped, failed: checkpoints.append(len(records)),
        )

        # 1 video, 3 shots -> 3 checkpoints (one per shot), not 1 (per video).
        assert checkpoints == [1, 2, 3]

    async def test_unexpected_exception_type_is_recorded_not_left_uncaught(self, tmp_path):
        """The real incident: google-genai raises RateLimitError on 429,
        which is neither OSError/RuntimeError/ValueError nor any subclass of
        google.genai.errors.APIError (verified against the installed SDK,
        2026-08-09 -- two unrelated exception hierarchies). A narrow except
        clause would let it crash the whole batch; it must be caught and
        recorded as a failed shot instead."""
        shot = _shot("L21_V001", 1, 0.0, 1.0, "L21", "A")
        shots_by_video = {"L21_V001": [shot]}
        video_meta = {"L21_V001": VideoMeta(batch="L21", author="A", title="t1")}
        image_path = tmp_path / "frame.jpg"
        Image.new("RGB", (10, 10)).save(image_path)
        keyframes_by_video = {
            "L21_V001": [
                {
                    "shot_id": "L21_V001_shot1",
                    "frame_idx": 1,
                    "keyframe_id": "k1",
                    "image_path": str(image_path),
                }
            ]
        }
        config = BuildEvalSetConfig(kis_sample_size=1, trake_sequence_count=0)

        class _RateLimitLikeError(Exception):
            """Simulates google.genai's actual exception -- a plain
            Exception subclass unrelated to any built-in error type."""

        class _AlwaysRateLimited:
            def generate(self, image, title):
                raise _RateLimitLikeError("429 quota exceeded")

        result = await _generate_candidates(
            config,
            shots_by_video=shots_by_video,
            video_meta=video_meta,
            caption_generator=_AlwaysRateLimited(),
            keyframe_source=_FakeKeyframeSource(keyframes_by_video),
        )

        assert result["failed"] == ["L21_V001_shot1"]
        assert result["records"] == []


class TestFlagGenericCaptions:
    def test_caption_below_min_content_words_is_flagged_generic(self):
        config = BuildEvalSetConfig(
            min_content_words=4,
            near_duplicate_jaccard=0.6,
            max_near_duplicates=3,
            vietnamese_stopwords=_TEST_STOPWORDS,
        )
        records = [{"caption": "Một người đang nói"}]

        tags = flag_generic_captions(records, config)

        assert tags == ["generic_short"]

    def test_near_duplicate_captions_are_flagged_generic(self):
        config = BuildEvalSetConfig(
            min_content_words=2,
            near_duplicate_jaccard=0.5,
            max_near_duplicates=1,
            vietnamese_stopwords=_TEST_STOPWORDS,
        )
        records = [
            {"caption": "Phóng viên đứng trước toà nhà quốc hội buổi sáng"},
            {"caption": "Phóng viên đứng trước toà nhà quốc hội buổi chiều"},
            {"caption": "Phóng viên đứng trước toà nhà quốc hội buổi tối"},
            {"caption": "Cảnh sát giao thông kiểm tra nồng độ cồn ban đêm"},
        ]

        tags = flag_generic_captions(records, config)

        assert tags[0] == "generic_duplicate"
        assert tags[1] == "generic_duplicate"
        assert tags[2] == "generic_duplicate"

    def test_specific_caption_is_not_flagged(self):
        config = BuildEvalSetConfig(
            min_content_words=2,
            near_duplicate_jaccard=0.5,
            max_near_duplicates=1,
            vietnamese_stopwords=_TEST_STOPWORDS,
        )
        records = [
            {"caption": "Phóng viên đứng trước toà nhà quốc hội buổi sáng"},
            {"caption": "Phóng viên đứng trước toà nhà quốc hội buổi chiều"},
            {"caption": "Phóng viên đứng trước toà nhà quốc hội buổi tối"},
            {"caption": "Cảnh sát giao thông kiểm tra nồng độ cồn ban đêm"},
        ]

        tags = flag_generic_captions(records, config)

        assert tags[3] == "kept"


class TestExtractTrakeSequences:
    def test_trake_sequences_are_contiguous_and_time_ordered(self):
        segments = [
            GroundTruthSegment("L21_V001", 0.0, 1.0, 1, "a"),
            GroundTruthSegment("L21_V001", 1.0, 2.0, 2, "b"),
            GroundTruthSegment("L21_V001", 2.0, 3.0, 3, "c"),
            GroundTruthSegment("L21_V001", 10.0, 11.0, 8, "gap"),  # not contiguous with 1-3
        ]

        sequences = extract_trake_sequences(segments, n=3)

        assert len(sequences) == 1
        window = sequences[0]
        assert [s.order_index for s in window] == [1, 2, 3]
        assert [s.start_sec for s in window] == sorted(s.start_sec for s in window)

    def test_no_sequence_when_shots_are_not_contiguous(self):
        segments = [
            GroundTruthSegment("L21_V001", 0.0, 1.0, 1, "a"),
            GroundTruthSegment("L21_V001", 5.0, 6.0, 5, "b"),
            GroundTruthSegment("L21_V001", 6.0, 7.0, 6, "c"),
        ]

        assert extract_trake_sequences(segments, n=3) == []


class TestFreezeRecords:
    def test_eval_set_hash_changes_when_content_changes(self):
        record_a = [
            {
                "video_id": "L21_V001",
                "start_sec": 0.0,
                "end_sec": 1.0,
                "order_index": 1,
                "caption": "abc",
            }
        ]
        record_b = [
            {
                "video_id": "L21_V001",
                "start_sec": 0.0,
                "end_sec": 1.0,
                "order_index": 1,
                "caption": "xyz",
            }
        ]

        _, digest_a = freeze_records(record_a)
        _, digest_b = freeze_records(record_b)

        assert digest_a != digest_b

    def test_frozen_json_matches_ground_truth_segment_schema(self, tmp_path):
        approved = [
            {
                "video_id": "L21_V001",
                "start_sec": 1.0,
                "end_sec": 2.0,
                "order_index": 1,
                "caption": "abc",
                "keyframe_id": "k1",  # extra field -- must NOT leak into the frozen file
            }
        ]

        json_text, _ = freeze_records(approved)
        eval_path = tmp_path / "aic_eval_v1.json"
        eval_path.write_text(json_text, encoding="utf-8")

        adapter = AIC2026Adapter(dataset_root=tmp_path, eval_set_path=eval_path)
        ground_truth = adapter.load_ground_truth("all")

        assert len(ground_truth) == 1
        assert ground_truth[0].video_id == "L21_V001"
        assert ground_truth[0].caption == "abc"
        assert ground_truth[0].order_index == 1


class TestComputeQualityReport:
    def test_freeze_raises_on_unreviewed_record(self):
        candidates = [
            {
                "video_id": "L21_V001",
                "shot_id": 1,
                "caption": "abc",
                "approved": None,
                "auto_filter": "kept",
            },
        ]

        with pytest.raises(EvalSetBuildError, match="unreviewed"):
            compute_quality_report(candidates)

    def test_quality_report_marks_inter_reviewer_agreement_not_measured(self):
        candidates = [
            {
                "video_id": "L21_V001",
                "shot_id": 1,
                "caption": "abc",
                "approved": True,
                "auto_filter": "kept",
            },
            {
                "video_id": "L21_V001",
                "shot_id": 2,
                "caption": "def",
                "approved": False,
                "auto_filter": "kept",
            },
        ]

        report = compute_quality_report(candidates)

        assert report.inter_reviewer_agreement is None

    def test_rejection_rate_reflects_disapproved_fraction(self):
        candidates = [
            {
                "video_id": "v",
                "shot_id": 1,
                "caption": "a",
                "approved": True,
                "auto_filter": "kept",
            },
            {
                "video_id": "v",
                "shot_id": 2,
                "caption": "b",
                "approved": False,
                "auto_filter": "kept",
            },
            {
                "video_id": "v",
                "shot_id": 3,
                "caption": "c",
                "approved": False,
                "auto_filter": "kept",
            },
        ]

        report = compute_quality_report(candidates)

        assert report.approved_count == 1
        assert report.rejection_rate == pytest.approx(2 / 3)


class TestConfigLoading:
    def test_load_config_reads_yaml_into_dataclass(self, tmp_path):
        config_path = tmp_path / "build_eval_set.yaml"
        config_path.write_text(
            "seed: 123\nkis_sample_size: 50\nvietnamese_stopwords: [một, và]\n",
            encoding="utf-8",
        )

        config = _load_config(config_path)

        assert config.seed == 123
        assert config.kis_sample_size == 50
        assert config.vietnamese_stopwords == ("một", "và")

    def test_load_config_defaults_stopwords_to_empty_tuple_when_absent(self, tmp_path):
        config_path = tmp_path / "build_eval_set.yaml"
        config_path.write_text("seed: 1\n", encoding="utf-8")

        config = _load_config(config_path)

        assert config.vietnamese_stopwords == ()


class TestWriteCandidatesDoc:
    def test_write_candidates_doc_round_trips_through_resume(self, tmp_path):
        out_path = tmp_path / "aic_eval_v1.candidates.json"
        config = BuildEvalSetConfig(seed=1)
        records = [{"video_id": "L21_V001", "shot_id": 1, "caption": "abc", "approved": None}]

        _write_candidates_doc(out_path, config, records, skipped_no_keyframe=[], failed=["x"])

        doc = json.loads(out_path.read_text(encoding="utf-8"))
        assert doc["records"] == records
        assert doc["failed"] == ["x"]
        assert doc["config"]["seed"] == 1


class TestGeminiCaptionGeneratorRetry:
    def test_caption_generator_retries_on_transient_error(self, monkeypatch):
        config = BuildEvalSetConfig(max_retries=3, retry_base_delay_seconds=0.0)
        generator = object.__new__(GeminiCaptionGenerator)  # bypass __init__: no real Gemini client
        object.__setattr__(generator, "_config", config)

        calls = {"count": 0}

        def _flaky_call_once(image_b64, title):
            calls["count"] += 1
            if calls["count"] < 3:
                raise ConnectionError("simulated transient network error")
            return "caption tiếng Việt"

        monkeypatch.setattr(generator, "_call_once", _flaky_call_once)

        caption = generator.generate(Image.new("RGB", (10, 10)), "tiêu đề")

        assert caption == "caption tiếng Việt"
        assert calls["count"] == 3

    def test_caption_generator_raises_after_exhausting_retries(self, monkeypatch):
        config = BuildEvalSetConfig(max_retries=2, retry_base_delay_seconds=0.0)
        generator = object.__new__(GeminiCaptionGenerator)
        object.__setattr__(generator, "_config", config)

        def _always_fails(image_b64, title):
            raise OSError("simulated permanent network error")

        monkeypatch.setattr(generator, "_call_once", _always_fails)

        with pytest.raises(RuntimeError, match="failed after 2 attempts"):
            generator.generate(Image.new("RGB", (10, 10)), "tiêu đề")
