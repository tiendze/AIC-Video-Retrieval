"""RED reproducer for pipelines/asr.py (P0.11a TEXT branch — spoken dialogue).

Uses a fake AsrEsStore (Protocol test double) -- no real faster-whisper
model or Elasticsearch needed. `is_hallucination`/`to_frame_window` are pure
and tested directly against synthetic segments, following the same
convention as tests/test_ocr.py.
"""

from __future__ import annotations

import pytest

from pipelines.asr import (
    STATUS_DONE,
    AsrConfig,
    AsrSegment,
    VideoTiming,
    format_duration,
    is_hallucination,
    run_asr,
    should_transcribe,
    summarize_timings,
    to_frame_window,
)


def _segment(
    start: float = 1.0,
    end: float = 2.0,
    text: str = "xin chao cac ban",
    no_speech_prob: float = 0.1,
    avg_logprob: float = -0.3,
) -> AsrSegment:
    return AsrSegment(
        start=start, end=end, text=text, no_speech_prob=no_speech_prob, avg_logprob=avg_logprob
    )


class _FakeAsrEsStore:
    def __init__(self):
        self.bulk_index_calls: list[list] = []

    async def bulk_index(self, docs):
        self.bulk_index_calls.append(list(docs))


def test_hallucination_filter_drops_segment_above_no_speech_threshold():
    config = AsrConfig(no_speech_prob_threshold=0.6)
    segment = _segment(no_speech_prob=0.9)

    assert is_hallucination(segment, config) is True


def test_hallucination_filter_drops_segment_below_avg_logprob_threshold():
    config = AsrConfig(avg_logprob_threshold=-1.0)
    segment = _segment(avg_logprob=-2.5)

    assert is_hallucination(segment, config) is True


def test_hallucination_filter_keeps_confident_segment():
    config = AsrConfig(no_speech_prob_threshold=0.6, avg_logprob_threshold=-1.0, min_text_length=2)
    segment = _segment(no_speech_prob=0.1, avg_logprob=-0.2, text="tin tuc chinh")

    assert is_hallucination(segment, config) is False


def test_frame_window_uses_floor_and_ceil_so_no_frame_is_dropped():
    # fps=30: start=1.19s -> 35.7 -> floor=35 (round() would give 36);
    # end=2.01s -> 60.3 -> ceil=61 (round() would give 60). Proves this is
    # floor/ceil, not round() -- CLAUDE.md T2: the window only ever widens.
    segment = _segment(start=1.19, end=2.01)

    window = to_frame_window(segment, fps=30.0, frame_count=1000)

    assert window.start_frame_idx == 35
    assert window.end_frame_idx == 61


def test_frame_window_clamped_to_video_frame_count():
    segment = _segment(start=0.0, end=1000.0)

    window = to_frame_window(segment, fps=30.0, frame_count=100)

    assert window.start_frame_idx == 0
    assert window.end_frame_idx == 99  # frame_count - 1, never out of range


async def test_asr_indexes_only_kept_segments_to_elasticsearch():
    config = AsrConfig(no_speech_prob_threshold=0.6, avg_logprob_threshold=-1.0, min_text_length=2)
    segments = [
        _segment(text="doan noi that", no_speech_prob=0.1, avg_logprob=-0.2),
        _segment(text="", no_speech_prob=0.95, avg_logprob=-0.1),  # hallucinated: silence
    ]
    es = _FakeAsrEsStore()

    summary = await run_asr("video1", segments, es=es, fps=30.0, frame_count=1000, config=config)

    assert summary.kept_segments == 1
    assert len(es.bulk_index_calls[0]) == 1
    assert es.bulk_index_calls[0][0].text == "doan noi that"


def test_asr_thresholds_come_from_config_not_hardcoded():
    # An impossibly strict threshold makes even a confident segment fail --
    # proves the value comes from config, not a hardcoded constant.
    strict_config = AsrConfig(no_speech_prob_threshold=0.0)
    segment = _segment(no_speech_prob=0.05)

    assert is_hallucination(segment, strict_config) is True


async def test_asr_summary_counts_each_drop_reason_separately():
    config = AsrConfig(no_speech_prob_threshold=0.6, avg_logprob_threshold=-1.0, min_text_length=5)
    segments = [
        _segment(text="ok du dai", no_speech_prob=0.9, avg_logprob=-0.1),  # dropped: no_speech
        _segment(text="ok du dai", no_speech_prob=0.1, avg_logprob=-2.0),  # dropped: low_confidence
        _segment(text="a", no_speech_prob=0.1, avg_logprob=-0.1),  # dropped: short_text
        _segment(text="cau nay du dai va ro rang", no_speech_prob=0.1, avg_logprob=-0.1),  # kept
    ]
    es = _FakeAsrEsStore()

    summary = await run_asr("video1", segments, es=es, fps=30.0, frame_count=1000, config=config)

    assert summary.total_segments == 4
    assert summary.kept_segments == 1
    assert summary.dropped_no_speech == 1
    assert summary.dropped_low_confidence == 1
    assert summary.dropped_short_text == 1


def test_timing_summary_reports_total_and_mean_wall_clock_per_video():
    timings = [
        VideoTiming(video_id="L21_V001", wall_sec=100.0, audio_sec=1200.0),
        VideoTiming(video_id="L21_V002", wall_sec=60.0, audio_sec=800.0),
    ]

    summary = summarize_timings(timings)

    assert summary.num_videos == 2
    assert summary.total_wall_sec == 160.0
    assert summary.mean_wall_sec == 80.0
    assert summary.total_audio_sec == 2000.0


def test_timing_summary_speed_ratio_is_audio_seconds_per_wall_second():
    # 2000s of audio in 160s of wall clock = 12.5x realtime. This is the number
    # that projects a full-corpus run without re-measuring: remaining audio
    # divided by this ratio.
    timings = [
        VideoTiming(video_id="L21_V001", wall_sec=100.0, audio_sec=1200.0),
        VideoTiming(video_id="L21_V002", wall_sec=60.0, audio_sec=800.0),
    ]

    assert summarize_timings(timings).speed_ratio == 12.5


def test_timing_summary_of_a_run_that_transcribed_nothing_is_all_zeros():
    # Every video already `done` is a normal outcome, not an error -- must not
    # divide by zero.
    summary = summarize_timings([])

    assert summary.num_videos == 0
    assert summary.total_wall_sec == 0.0
    assert summary.mean_wall_sec == 0.0
    assert summary.speed_ratio == 0.0


def test_format_duration_scales_from_seconds_to_hours():
    assert format_duration(45.4) == "45s"
    assert format_duration(125.0) == "2m 05s"
    assert format_duration(3725.0) == "1h 02m 05s"


def test_asr_config_field_is_model_id_not_model_size():
    # research-asr-phowhisper-batched.plan.md T2: the field renamed from
    # model_size -> model_id (a PhoWhisper checkpoint is a path, not a size
    # label like "large-v3"). Default value is unchanged by the rename.
    config = AsrConfig()

    assert config.model_id == "large-v3"
    assert not hasattr(config, "model_size")


def test_asr_config_raises_when_batched_without_vad_filter():
    # faster_whisper/transcribe.py:413-419 (F3): batched mode with no VAD and
    # no clip_timestamps raises RuntimeError on any audio > 30s -- every video
    # in this corpus qualifies, so this must fail at config construction, not
    # mid-run after the model is already loaded.
    with pytest.raises(ValueError, match="vad_filter"):
        AsrConfig(batched=True, vad_filter=False)


def test_asr_config_allows_batched_with_vad_filter_enabled():
    config = AsrConfig(batched=True, vad_filter=True)

    assert config.batched is True


def test_asr_config_raises_when_batch_size_below_one():
    with pytest.raises(ValueError, match="batch_size"):
        AsrConfig(batch_size=0)


class _FakeWhisperModel:
    """Stands in for faster_whisper.WhisperModel -- records the kwargs
    transcribe() was called with, no GPU/checkpoint download needed."""

    def __init__(self, model_id, device, compute_type):
        self.model_id = model_id
        self.device = device
        self.compute_type = compute_type
        self.transcribe_calls: list[tuple[str, dict]] = []

    def transcribe(self, video_path, **kwargs):
        self.transcribe_calls.append((video_path, kwargs))
        return [], None


class _FakeBatchedInferencePipeline:
    """Stands in for faster_whisper.BatchedInferencePipeline -- wraps a
    (fake) WhisperModel the same way the real class does."""

    def __init__(self, model):
        self.model = model
        self.transcribe_calls: list[tuple[str, dict]] = []

    def transcribe(self, video_path, **kwargs):
        self.transcribe_calls.append((video_path, kwargs))
        return [], None


class TestFasterWhisperClientBatchedBranch:
    """faster_whisper.WhisperModel/BatchedInferencePipeline are monkeypatched
    at module level -- pipelines/asr.py imports them lazily inside __init__,
    so patching the faster_whisper module attribute before construction is
    enough (research-asr-phowhisper-batched.plan.md T4)."""

    def test_sequential_path_never_passes_batch_size(self, monkeypatch):
        monkeypatch.setattr("faster_whisper.WhisperModel", _FakeWhisperModel)
        from pipelines.asr import FasterWhisperClient

        config = AsrConfig(batched=False)
        client = FasterWhisperClient(config=config)
        client.transcribe("video.mp4")

        video_path, kwargs = client._model.transcribe_calls[0]
        assert video_path == "video.mp4"
        assert "batch_size" not in kwargs

    def test_batched_path_wraps_model_in_batched_inference_pipeline(self, monkeypatch):
        monkeypatch.setattr("faster_whisper.WhisperModel", _FakeWhisperModel)
        monkeypatch.setattr(
            "faster_whisper.BatchedInferencePipeline", _FakeBatchedInferencePipeline
        )
        from pipelines.asr import FasterWhisperClient

        config = AsrConfig(batched=True, vad_filter=True)
        client = FasterWhisperClient(config=config)

        assert isinstance(client._model, _FakeBatchedInferencePipeline)
        assert isinstance(client._model.model, _FakeWhisperModel)

    def test_batched_path_passes_configured_batch_size(self, monkeypatch):
        monkeypatch.setattr("faster_whisper.WhisperModel", _FakeWhisperModel)
        monkeypatch.setattr(
            "faster_whisper.BatchedInferencePipeline", _FakeBatchedInferencePipeline
        )
        from pipelines.asr import FasterWhisperClient

        config = AsrConfig(batched=True, vad_filter=True, batch_size=4)
        client = FasterWhisperClient(config=config)
        client.transcribe("video.mp4")

        _video_path, kwargs = client._model.transcribe_calls[0]
        assert kwargs["batch_size"] == 4


class TestShouldTranscribe:
    """research-asr-phowhisper-batched.plan.md F1: resume must key off BOTH
    asr_status and model_id. Without the model_id check, switching from
    large-v3 to PhoWhisper and re-running would skip every video already
    `done` under the old model, yet _dump_json_artifacts would still stamp
    the exported JSON's meta.model_id with the NEW model -- a file that lies
    about which checkpoint produced it, with no crash, no log line."""

    def test_transcribes_when_no_prior_run_exists(self):
        assert should_transcribe(None, model_id="large-v3", force=False) is True

    def test_skips_when_same_model_already_done(self):
        existing_run = {"asr_status": STATUS_DONE, "model_id": "large-v3"}

        assert should_transcribe(existing_run, model_id="large-v3", force=False) is False

    def test_transcribes_again_when_model_id_changed_even_if_done(self):
        existing_run = {"asr_status": STATUS_DONE, "model_id": "large-v3"}

        assert (
            should_transcribe(existing_run, model_id="models/PhoWhisper-large-ct2", force=False)
            is True
        )

    def test_transcribes_when_prior_run_failed_regardless_of_model_id(self):
        existing_run = {"asr_status": "failed", "model_id": "large-v3"}

        assert should_transcribe(existing_run, model_id="large-v3", force=False) is True

    def test_force_always_transcribes_even_when_done_and_same_model(self):
        existing_run = {"asr_status": STATUS_DONE, "model_id": "large-v3"}

        assert should_transcribe(existing_run, model_id="large-v3", force=True) is True
