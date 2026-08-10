"""RED reproducer for pipelines/ingest.py (P0.7).

Uses fake ShotDetector/MultimodalEncoder/MongoStore/MilvusStore/OcrClient/
EsStore test doubles (no torch checkpoint, no Docker stack, no Gemini key
needed) plus the real 5s pilot fixture video for the parts that must
actually decode frames (extract_keyframe_images, ingest_video). Shots are
hand-picked to cover the fixture's known 150 frames (30fps, 5s —
tests/fixtures/aic2026_sample/Videos_L21_a/video/L21_V001.mp4).

The OCR branch itself (text filter, phash dedupe, Gemini call, budget cap)
is unit-tested against synthetic images in tests/test_ocr.py — the tests
here only cover the *wiring*: does ingest_video/run_ingest reach the right
ingest_status, does resume read keyframes back from Mongo instead of
redoing shot detection, does the token budget get shared (not reset) across
videos in one run_ingest call.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

import pipelines.ingest as ingest_module
from datasets.base import VideoRef
from pipelines.ingest import (
    STATUS_COMPLETE,
    STATUS_EMBEDDED,
    STATUS_FAILED,
    build_extracted_keyframes,
    extract_keyframe_images,
    ingest_video,
    run_ingest,
    save_keyframe_images,
)
from pipelines.ocr import OcrRunSummary
from pipelines.shot_detect import Shot

_FIXTURE_VIDEO = "tests/fixtures/aic2026_sample/Videos_L21_a/video/L21_V001.mp4"
_FIXTURE_FRAME_COUNT = 150  # 30fps * 5s
_TWO_SHOTS = [Shot(start_frame=0, end_frame=74), Shot(start_frame=75, end_frame=149)]
# a=0,b=74,span=74 -> 0,24,49,74 ; a=75,b=149,span=74 -> 75,99,124,149
_EXPECTED_FRAME_INDICES = [0, 24, 49, 74, 75, 99, 124, 149]
_AMPLE_TOKEN_BUDGET = 10_000_000  # far above anything one fixture video could spend


def _video_ref(video_id: str) -> VideoRef:
    return VideoRef(
        video_id=video_id,
        source_path=_FIXTURE_VIDEO,
        duration_sec=5.0,
        category="60 Giây Official",
        title="sample title",
        publish_date="01/08/2024",
        watch_url="https://youtube.com/watch?v=x",
        channel_id="UCxxx",
    )


class _FakeShotDetector:
    """Always returns the same shots, regardless of path (tests reuse one
    physical fixture file under several fake `video_id`s)."""

    def __init__(self, shots: list[Shot]):
        self._shots = shots
        self.calls: list[str] = []

    def detect_shots(self, video_path: str) -> list[Shot]:
        self.calls.append(video_path)
        return self._shots


class _RaisingShotDetector:
    def detect_shots(self, video_path: str) -> list[Shot]:
        raise ValueError("boom")


class _FailFirstCallShotDetector:
    """Fails on its first call, succeeds afterwards — simulates one bad video
    in a batch without needing path-based branching."""

    def __init__(self, shots: list[Shot]):
        self._shots = shots
        self.calls = 0

    def detect_shots(self, video_path: str) -> list[Shot]:
        self.calls += 1
        if self.calls == 1:
            raise ValueError("simulated shot-detection failure")
        return self._shots


class _FakeEncoder:
    embedding_dim = 4
    model_id = "fake"

    def encode_image(self, images):
        return np.zeros((len(images), 4), dtype=np.float32)

    def encode_text(self, texts):
        raise NotImplementedError


class _FakeMilvusStore:
    def __init__(self):
        self.upsert_calls: list[list] = []

    def upsert(self, vectors):
        self.upsert_calls.append(list(vectors))


class _FakeOcrClient:
    def __init__(self):
        self.calls = 0

    def extract_text(self, image):
        self.calls += 1
        return "recognized text"


class _FakeEsStore:
    def __init__(self):
        self.bulk_index_calls: list[list] = []

    async def bulk_index(self, docs):
        self.bulk_index_calls.append(list(docs))


class _FakeMongoStore:
    def __init__(self):
        self.videos: dict[str, dict] = {}
        self.keyframes: dict[str, dict] = {}

    async def upsert_video(self, doc):
        self.videos[doc.video_id] = dict(doc.__dict__)

    async def get_video(self, video_id):
        return self.videos.get(video_id)

    async def bulk_insert_keyframes(self, docs):
        count = 0
        for doc in docs:
            if doc.keyframe_id not in self.keyframes:
                self.keyframes[doc.keyframe_id] = dict(doc.__dict__)
                count += 1
        return count

    async def get_keyframes_for_video(self, video_id):
        docs = [doc for doc in self.keyframes.values() if doc["video_id"] == video_id]
        return sorted(docs, key=lambda d: d["global_index"])

    async def max_keyframe_global_index(self):
        if not self.keyframes:
            return None
        return max(k["global_index"] for k in self.keyframes.values())


def test_extract_keyframe_images_returns_real_frames_at_target_indices():
    images = extract_keyframe_images(_FIXTURE_VIDEO, _EXPECTED_FRAME_INDICES)

    assert set(images.keys()) == set(_EXPECTED_FRAME_INDICES)
    for image in images.values():
        assert isinstance(image, Image.Image)
        assert image.size[0] > 0 and image.size[1] > 0


def test_extract_keyframe_images_raises_when_frame_idx_out_of_range():
    with pytest.raises(ValueError, match="never produced frame_idx"):
        extract_keyframe_images(_FIXTURE_VIDEO, [_FIXTURE_FRAME_COUNT + 100])


def test_build_extracted_keyframes_assigns_contiguous_global_index_in_shot_order():
    video = _video_ref("L21_V001")
    placeholder_images = {idx: Image.new("RGB", (4, 4)) for idx in _EXPECTED_FRAME_INDICES}

    keyframes = build_extracted_keyframes(
        video, _TWO_SHOTS, placeholder_images, fps=30.0, global_index_start=10
    )

    assert [k.global_index for k in keyframes] == list(range(10, 18))
    assert [k.frame_idx for k in keyframes] == _EXPECTED_FRAME_INDICES
    assert keyframes[0].keyframe_id == "L21_V001_10"
    assert keyframes[0].shot_id == "L21_V001_shot0"
    assert keyframes[4].shot_id == "L21_V001_shot1"


def test_build_extracted_keyframes_derives_timestamp_from_frame_idx_and_fps():
    video = _video_ref("L21_V001")
    placeholder_images = {idx: Image.new("RGB", (4, 4)) for idx in _EXPECTED_FRAME_INDICES}

    keyframes = build_extracted_keyframes(
        video, _TWO_SHOTS, placeholder_images, fps=25.0, global_index_start=0
    )

    assert keyframes[0].timestamp == pytest.approx(0 / 25.0)
    assert keyframes[1].timestamp == pytest.approx(24 / 25.0)


def test_save_keyframe_images_writes_jpeg_and_thumbnail(tmp_path):
    video = _video_ref("L21_V001")
    placeholder_images = {0: Image.new("RGB", (640, 360))}
    keyframes = build_extracted_keyframes(
        video, [Shot(0, 0)], placeholder_images, fps=30.0, global_index_start=0
    )

    save_keyframe_images(keyframes, tmp_path)

    video_dir = tmp_path / "L21_V001"
    assert (video_dir / "L21_V001_0.jpg").exists()
    assert (video_dir / "L21_V001_0_thumb.jpg").exists()
    with Image.open(video_dir / "L21_V001_0_thumb.jpg") as thumb:
        assert max(thumb.size) <= 320


async def test_ingest_video_writes_mongo_and_milvus_then_completes_ocr_branch(tmp_path):
    video = _video_ref("L21_V001")
    detector = _FakeShotDetector(_TWO_SHOTS)
    encoder = _FakeEncoder()
    mongo = _FakeMongoStore()
    milvus = _FakeMilvusStore()
    es = _FakeEsStore()

    await ingest_video(
        video,
        shot_detector=detector,
        encoder=encoder,
        mongo=mongo,
        milvus=milvus,
        image_dir=tmp_path,
        ocr_client=_FakeOcrClient(),
        es=es,
        ocr_token_budget=_AMPLE_TOKEN_BUDGET,
    )

    video_doc = mongo.videos["L21_V001"]
    assert video_doc["ingest_status"] == STATUS_COMPLETE
    assert video_doc["s_v"] == 0
    assert video_doc["e_v"] == 7
    assert video_doc["num_keyframes"] == 8
    assert len(mongo.keyframes) == 8
    assert len(milvus.upsert_calls) == 1
    assert len(milvus.upsert_calls[0]) == 8
    assert len(es.bulk_index_calls) == 1  # OCR branch ran and indexed to ES


async def test_run_ingest_derives_each_video_batch_independently(tmp_path):
    """Regression: `batch` used to be one constant passed to run_ingest() for
    the whole call, so ingesting the real pilot split (L21-L24 together in
    one run) would have mislabeled every video outside the first batch."""
    videos = [_video_ref("L21_V001"), _video_ref("L24_V002")]
    mongo = _FakeMongoStore()

    await run_ingest(
        videos,
        shot_detector=_FakeShotDetector(_TWO_SHOTS),
        encoder=_FakeEncoder(),
        mongo=mongo,
        milvus=_FakeMilvusStore(),
        image_dir=tmp_path,
        ocr_client=_FakeOcrClient(),
        es=_FakeEsStore(),
        ocr_token_budget=_AMPLE_TOKEN_BUDGET,
    )

    assert mongo.videos["L21_V001"]["batch"] == "L21"
    assert mongo.videos["L24_V002"]["batch"] == "L24"


async def test_run_ingest_global_index_is_contiguous_across_videos(tmp_path):
    videos = [_video_ref("L21_V001a"), _video_ref("L21_V001b")]
    detector = _FakeShotDetector(_TWO_SHOTS)
    mongo = _FakeMongoStore()

    summary = await run_ingest(
        videos,
        shot_detector=detector,
        encoder=_FakeEncoder(),
        mongo=mongo,
        milvus=_FakeMilvusStore(),
        image_dir=tmp_path,
        ocr_client=_FakeOcrClient(),
        es=_FakeEsStore(),
        ocr_token_budget=_AMPLE_TOKEN_BUDGET,
    )

    assert summary.completed_video_ids == ["L21_V001a", "L21_V001b"]
    assert mongo.videos["L21_V001a"]["s_v"] == 0
    assert mongo.videos["L21_V001a"]["e_v"] == 7
    assert mongo.videos["L21_V001b"]["s_v"] == 8
    assert mongo.videos["L21_V001b"]["e_v"] == 15


async def test_run_ingest_skips_video_already_complete(tmp_path):
    mongo = _FakeMongoStore()
    mongo.videos["L21_V001"] = {"video_id": "L21_V001", "ingest_status": STATUS_COMPLETE, "e_v": 3}
    detector = _RaisingShotDetector()  # would raise if actually called

    summary = await run_ingest(
        [_video_ref("L21_V001")],
        shot_detector=detector,
        encoder=_FakeEncoder(),
        mongo=mongo,
        milvus=_FakeMilvusStore(),
        image_dir=tmp_path,
        ocr_client=_FakeOcrClient(),
        es=_FakeEsStore(),
        ocr_token_budget=_AMPLE_TOKEN_BUDGET,
    )

    assert summary.skipped_video_ids == ["L21_V001"]
    assert summary.completed_video_ids == []


async def test_run_ingest_records_failure_and_continues_to_next_video(tmp_path):
    videos = [_video_ref("broken"), _video_ref("L21_V001b")]
    mongo = _FakeMongoStore()

    summary = await run_ingest(
        videos,
        shot_detector=_FailFirstCallShotDetector(_TWO_SHOTS),
        encoder=_FakeEncoder(),
        mongo=mongo,
        milvus=_FakeMilvusStore(),
        image_dir=tmp_path,
        ocr_client=_FakeOcrClient(),
        es=_FakeEsStore(),
        ocr_token_budget=_AMPLE_TOKEN_BUDGET,
    )

    assert summary.failed_video_ids == ["broken"]
    assert summary.completed_video_ids == ["L21_V001b"]
    assert mongo.videos["broken"]["ingest_status"] == STATUS_FAILED


async def test_run_ingest_records_failure_for_exception_types_beyond_oserror_valueerror(tmp_path):
    """Regression for the code-review CRITICAL finding: BEiT3Encoder/MilvusStore
    don't normalize failures down to OSError/ValueError (e.g. a CUDA OOM is a
    RuntimeError), so a too-narrow except clause let one bad video crash the
    whole batch instead of being recorded and skipped."""

    class _RuntimeErrorShotDetector:
        def detect_shots(self, video_path: str) -> list[Shot]:
            raise RuntimeError("simulated CUDA OOM-like failure")

    mongo = _FakeMongoStore()

    summary = await run_ingest(
        [_video_ref("broken")],
        shot_detector=_RuntimeErrorShotDetector(),
        encoder=_FakeEncoder(),
        mongo=mongo,
        milvus=_FakeMilvusStore(),
        image_dir=tmp_path,
        ocr_client=_FakeOcrClient(),
        es=_FakeEsStore(),
        ocr_token_budget=_AMPLE_TOKEN_BUDGET,
    )

    assert summary.failed_video_ids == ["broken"]
    assert mongo.videos["broken"]["ingest_status"] == STATUS_FAILED


async def test_run_ingest_second_run_over_already_complete_video_skips_without_duplicating(
    tmp_path,
):
    """A video that reaches STATUS_COMPLETE in one run (VISION + OCR both
    done) must not be reprocessed by a second `run_ingest` over the same
    video list — no duplicate keyframes/global_index range, shot detector
    not called again."""
    video = _video_ref("L21_V001")
    detector = _FakeShotDetector(_TWO_SHOTS)
    mongo = _FakeMongoStore()
    milvus = _FakeMilvusStore()

    first = await run_ingest(
        [video],
        shot_detector=detector,
        encoder=_FakeEncoder(),
        mongo=mongo,
        milvus=milvus,
        image_dir=tmp_path,
        ocr_client=_FakeOcrClient(),
        es=_FakeEsStore(),
        ocr_token_budget=_AMPLE_TOKEN_BUDGET,
    )
    assert first.completed_video_ids == ["L21_V001"]
    assert mongo.videos["L21_V001"]["ingest_status"] == STATUS_COMPLETE

    second = await run_ingest(
        [video],
        shot_detector=detector,
        encoder=_FakeEncoder(),
        mongo=mongo,
        milvus=milvus,
        image_dir=tmp_path,
        ocr_client=_FakeOcrClient(),
        es=_FakeEsStore(),
        ocr_token_budget=_AMPLE_TOKEN_BUDGET,
    )

    assert second.skipped_video_ids == ["L21_V001"]
    assert second.completed_video_ids == []
    assert len(detector.calls) == 1  # not called again on the second run
    assert len(mongo.keyframes) == 8  # still 8, not 16
    assert len(milvus.upsert_calls) == 1  # no second upsert call
    assert mongo.videos["L21_V001"]["e_v"] == 7  # unchanged, not re-ranged to 8..15


async def test_run_ingest_resumes_ocr_for_video_stuck_at_embedded(tmp_path):
    """A video that reached STATUS_EMBEDDED in an earlier run (VISION done,
    OCR never attempted -- e.g. budget was exhausted or the process was
    killed before OCR wiring landed) must have OCR run for it on the next
    `run_ingest`, reading its keyframes back from Mongo instead of redoing
    shot detection/embedding."""
    video = _video_ref("L21_V001")
    detector = _FakeShotDetector(_TWO_SHOTS)
    mongo = _FakeMongoStore()
    milvus = _FakeMilvusStore()

    # First run with zero OCR budget: VISION completes, OCR is skipped, video
    # stays at STATUS_EMBEDDED.
    first = await run_ingest(
        [video],
        shot_detector=detector,
        encoder=_FakeEncoder(),
        mongo=mongo,
        milvus=milvus,
        image_dir=tmp_path,
        ocr_client=_FakeOcrClient(),
        es=_FakeEsStore(),
        ocr_token_budget=0,
    )
    assert first.completed_video_ids == ["L21_V001"]
    assert mongo.videos["L21_V001"]["ingest_status"] == STATUS_EMBEDDED
    assert len(detector.calls) == 1

    es = _FakeEsStore()
    second = await run_ingest(
        [video],
        shot_detector=detector,
        encoder=_FakeEncoder(),
        mongo=mongo,
        milvus=milvus,
        image_dir=tmp_path,
        ocr_client=_FakeOcrClient(),
        es=es,
        ocr_token_budget=_AMPLE_TOKEN_BUDGET,
    )

    assert second.completed_video_ids == ["L21_V001"]
    assert mongo.videos["L21_V001"]["ingest_status"] == STATUS_COMPLETE
    assert len(detector.calls) == 1  # resume must not redo shot detection
    assert len(milvus.upsert_calls) == 1  # no re-embedding either
    assert len(mongo.keyframes) == 8  # not duplicated
    assert len(es.bulk_index_calls) == 1  # OCR did run this time


async def test_ingest_video_ocr_failure_leaves_video_embedded_not_failed(tmp_path, monkeypatch):
    """Regression: a Gemini failure (quota exhausted, auth error, etc. --
    anything run_ocr() itself doesn't catch) used to propagate out of
    ingest_video(), get caught by run_ingest()'s per-video except clause, and
    overwrite the video's already-correct VISION VideoDocument with a zeroed
    STATUS_FAILED placeholder -- losing s_v/e_v/num_keyframes and forcing a
    full shot-detect/embed redo next run instead of a cheap OCR-only resume."""

    async def _raising_run_ocr(candidates, *, ocr_client, es, token_budget, config):
        raise RuntimeError("simulated Gemini quota exhausted (429)")

    monkeypatch.setattr(ingest_module, "run_ocr", _raising_run_ocr)

    video = _video_ref("L21_V001")
    detector = _FakeShotDetector(_TWO_SHOTS)
    mongo = _FakeMongoStore()
    milvus = _FakeMilvusStore()

    summary = await run_ingest(
        [video],
        shot_detector=detector,
        encoder=_FakeEncoder(),
        mongo=mongo,
        milvus=milvus,
        image_dir=tmp_path,
        ocr_client=_FakeOcrClient(),
        es=_FakeEsStore(),
        ocr_token_budget=_AMPLE_TOKEN_BUDGET,
    )

    video_doc = mongo.videos["L21_V001"]
    assert video_doc["ingest_status"] == STATUS_EMBEDDED  # not STATUS_FAILED
    assert video_doc["s_v"] == 0  # VISION's real values preserved, not zeroed
    assert video_doc["e_v"] == 7
    assert video_doc["num_keyframes"] == 8
    assert summary.completed_video_ids == ["L21_V001"]
    assert summary.failed_video_ids == []

    # Next run must resume OCR-only, not redo shot detection/embedding.
    monkeypatch.undo()
    second = await run_ingest(
        [video],
        shot_detector=detector,
        encoder=_FakeEncoder(),
        mongo=mongo,
        milvus=milvus,
        image_dir=tmp_path,
        ocr_client=_FakeOcrClient(),
        es=_FakeEsStore(),
        ocr_token_budget=_AMPLE_TOKEN_BUDGET,
    )
    assert second.completed_video_ids == ["L21_V001"]
    assert mongo.videos["L21_V001"]["ingest_status"] == STATUS_COMPLETE
    assert len(detector.calls) == 1  # still not called again
    assert len(mongo.keyframes) == 8  # still not duplicated


async def test_run_ingest_shares_ocr_token_budget_across_videos(tmp_path, monkeypatch):
    """The token budget passed to `run_ingest` is a pool for the *whole* run,
    not reset per video (references.md §5.1: the ~6.7M token cap is for the
    entire pilot, not per video). Monkeypatches `run_ocr` itself so this test
    stays independent of the real text-filter/dedupe heuristics (covered in
    tests/test_ocr.py) and only exercises the budget bookkeeping."""
    calls: list[int] = []

    async def _fake_run_ocr(candidates, *, ocr_client, es, token_budget, config):
        tokens_used = min(token_budget, 100)
        calls.append(token_budget)
        return OcrRunSummary(
            total_candidates=len(candidates),
            after_text_filter=len(candidates),
            after_dedupe=len(candidates),
            gemini_calls=1 if tokens_used else 0,
            tokens_used=tokens_used,
            stopped_on_budget=tokens_used < 100,
            failed_keyframe_ids=[],
        )

    monkeypatch.setattr(ingest_module, "run_ocr", _fake_run_ocr)

    videos = [_video_ref("L21_V001"), _video_ref("L21_V002"), _video_ref("L21_V003")]
    mongo = _FakeMongoStore()

    summary = await run_ingest(
        videos,
        shot_detector=_FakeShotDetector(_TWO_SHOTS),
        encoder=_FakeEncoder(),
        mongo=mongo,
        milvus=_FakeMilvusStore(),
        image_dir=tmp_path,
        ocr_client=_FakeOcrClient(),
        es=_FakeEsStore(),
        ocr_token_budget=150,
    )

    # video1 gets the full 150, spends 100 -> 50 left; video2 gets 50, spends
    # all 50 -> 0 left; video3 gets 0 and must never call run_ocr at all.
    assert calls == [150, 50]
    assert mongo.videos["L21_V001"]["ingest_status"] == STATUS_COMPLETE
    assert mongo.videos["L21_V002"]["ingest_status"] == STATUS_COMPLETE
    assert mongo.videos["L21_V003"]["ingest_status"] == STATUS_EMBEDDED
    assert summary.completed_video_ids == ["L21_V001", "L21_V002", "L21_V003"]
    assert summary.ocr_tokens_used == 150
