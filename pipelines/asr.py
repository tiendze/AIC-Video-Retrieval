"""ASR branch (P0.11a TEXT pipeline) — a fourth retrieval channel from spoken
dialogue, architecture.md §3.2 node A (straight from the `.mp4`), not node G
(never routed through a keyframe):

    .mp4 -> faster-whisper transcribe -> [1] hallucination filter -> [2] index ES (BM25)

Segments are indexed at SEGMENT granularity, never forced onto keyframe
boundaries: a single spoken sentence often spans several shots, and a
static studio shot can contain several sentences (plan P0.11). `frame_idx`
window per segment is a *retrieval window* only, derived via floor/ceil —
never the BTC submission `frame_id`, which always comes from a keyframe's
own `frame_idx` in Mongo/Milvus (CLAUDE.md T2).

Every tunable threshold lives in `AsrConfig`, not as a module-level magic
number (CLAUDE.md §6) — the field defaults here match
`configs/ingest/asr.yaml`, which is the source of truth a caller should
load and pass in.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from app.stores.es_client import AsrEsStore, AsrSegmentDocument
from pipelines.text_normalize import normalize_spoken_numbers

logger = logging.getLogger(__name__)

STATUS_DONE = "done"
STATUS_FAILED = "failed"

_SECONDS_PER_MINUTE = 60
_SECONDS_PER_HOUR = 3600

_REASON_NO_SPEECH = "no_speech"
_REASON_LOW_CONFIDENCE = "low_confidence"
_REASON_SHORT_TEXT = "short_text"


@dataclass(frozen=True)
class AsrConfig:
    """Mirrors `configs/ingest/asr.yaml` — see that file for the rationale
    behind each default."""

    model_id: str = "large-v3"
    compute_type: str = "float16"
    device: str = "cuda"
    beam_size: int = 5
    language: str = "vi"
    vad_filter: bool = True
    condition_on_previous_text: bool = False
    no_speech_prob_threshold: float = 0.6
    avg_logprob_threshold: float = -1.0
    min_text_length: int = 2
    batched: bool = False
    batch_size: int = 8
    # Extracts per-word timestamps (faster_whisper==1.2.1, verified against
    # transcribe.py:32-36/58) -- the only way to re-cut a batched-mode
    # segment locally without re-running Whisper. Batched mode hard-codes
    # each segment's boundary to the VAD chunk (up to ~30s of continuous
    # speech), not Whisper's own per-sentence token timestamps
    # (transcribe.py:398-409: max_speech_duration_s is forced to
    # chunk_length and any caller-supplied value is popped and discarded).
    # With word_timestamps on, resegment_by_pause() below recovers
    # sentence-length pieces from the pause between words -- a policy that
    # can be retuned on CPU at import time, no GPU needed.
    word_timestamps: bool = False

    def __post_init__(self) -> None:
        # faster_whisper/transcribe.py:413-419 (research-asr-phowhisper-batched
        # F3): batched mode with vad_filter=False and no clip_timestamps raises
        # RuntimeError on any audio > 30s — every video here qualifies. Fail at
        # construction, not mid-run after the model is already loaded.
        if self.batched and not self.vad_filter:
            raise ValueError(
                "AsrConfig(batched=True) requires vad_filter=True — faster-whisper's "
                "batched pipeline raises RuntimeError on audio > 30s otherwise "
                "(research-asr-phowhisper-batched.plan.md F3)"
            )
        if self.batch_size < 1:
            raise ValueError(f"AsrConfig.batch_size must be >= 1, got {self.batch_size}")


_DEFAULT_ASR_CONFIG = AsrConfig()


@dataclass(frozen=True)
class Word:
    """One decoded word with its own timestamp (faster_whisper's `Word`,
    references.md §5.3) — only populated when `AsrConfig.word_timestamps` is
    True. `word` carries its own leading space when it isn't sentence-initial
    (faster_whisper/transcribe.py:1917 relies on this), so reassembling text
    from a list of `Word` is `"".join(w.word for w in words)`, never
    `" ".join(...)`."""

    start: float
    end: float
    word: str
    probability: float


@dataclass(frozen=True)
class AsrSegment:
    """One segment as returned by faster-whisper's `Segment` (references.md
    §5.3) — only the fields the hallucination filter and frame-window math
    need. `words` is `None` unless `AsrConfig.word_timestamps=True`."""

    start: float
    end: float
    text: str
    no_speech_prob: float
    avg_logprob: float
    words: Sequence[Word] | None = None


@dataclass(frozen=True)
class FrameWindow:
    start_frame_idx: int
    end_frame_idx: int


class AsrClient(Protocol):
    """Firewall around the faster-whisper SDK — swappable for a fake in
    tests, or a different ASR backend (PhoWhisper, WhisperX) later."""

    def transcribe(self, video_path: str) -> Sequence[AsrSegment]: ...


@dataclass(frozen=True)
class VideoTiming:
    """Wall-clock cost of one video, next to how much audio it bought.
    `audio_sec` comes from the decoder (`frame_count / fps`), never from the
    organiser's `media-info.length`, which is rounded to whole seconds."""

    video_id: str
    wall_sec: float
    audio_sec: float


@dataclass(frozen=True)
class AsrTimingSummary:
    """`speed_ratio` is the projection tool: audio seconds transcribed per
    wall-clock second. Remaining corpus audio divided by it estimates the
    remaining run time without measuring again."""

    num_videos: int
    total_wall_sec: float
    mean_wall_sec: float
    total_audio_sec: float
    speed_ratio: float


def summarize_timings(timings: Sequence[VideoTiming]) -> AsrTimingSummary:
    """A run where every video was skipped as already `done` is a normal
    outcome, so an empty input returns zeros rather than dividing by zero."""
    if not timings:
        return AsrTimingSummary(
            num_videos=0,
            total_wall_sec=0.0,
            mean_wall_sec=0.0,
            total_audio_sec=0.0,
            speed_ratio=0.0,
        )

    total_wall_sec = sum(timing.wall_sec for timing in timings)
    total_audio_sec = sum(timing.audio_sec for timing in timings)
    return AsrTimingSummary(
        num_videos=len(timings),
        total_wall_sec=total_wall_sec,
        mean_wall_sec=total_wall_sec / len(timings),
        total_audio_sec=total_audio_sec,
        speed_ratio=total_audio_sec / total_wall_sec if total_wall_sec else 0.0,
    )


def format_duration(seconds: float) -> str:
    """`3725.0` -> `1h 02m 05s`. Multi-hour runs are the norm here, and a raw
    second count is unreadable at that scale."""
    total = int(seconds)
    hours, remainder = divmod(total, _SECONDS_PER_HOUR)
    minutes, secs = divmod(remainder, _SECONDS_PER_MINUTE)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


@dataclass(frozen=True)
class AsrRunSummary:
    total_segments: int
    kept_segments: int
    dropped_no_speech: int
    dropped_low_confidence: int
    dropped_short_text: int


def _hallucination_reason(
    segment: AsrSegment, config: AsrConfig = _DEFAULT_ASR_CONFIG
) -> str | None:
    """`None` when `segment` looks like real speech; otherwise the specific
    reason it was dropped, so callers can report drop counts per reason
    (plan P0.11a: 'báo cáo tỉ lệ cắt của từng bước riêng biệt, đừng gộp')."""
    if segment.no_speech_prob > config.no_speech_prob_threshold:
        return _REASON_NO_SPEECH
    if segment.avg_logprob < config.avg_logprob_threshold:
        return _REASON_LOW_CONFIDENCE
    if len(segment.text.strip()) < config.min_text_length:
        return _REASON_SHORT_TEXT
    return None


def is_hallucination(segment: AsrSegment, config: AsrConfig = _DEFAULT_ASR_CONFIG) -> bool:
    """True when `segment` looks like Whisper hallucinating over silence or
    noise rather than real speech — news jingles and silent B-roll are
    common triggers for this dataset (references.md §5.3). Checked via OR
    across three signals (`no_speech_prob`, `avg_logprob`, text length);
    every threshold comes from `AsrConfig`, never hardcoded here."""
    return _hallucination_reason(segment, config) is not None


def should_transcribe(existing_run: dict | None, model_id: str, force: bool) -> bool:
    """`True` when a video needs a transcribe call this run.

    Resume must key off BOTH `asr_status` and `model_id` (research-asr-
    phowhisper-batched.plan.md F1): checking `asr_status` alone means
    switching checkpoints and re-running silently skips every video already
    `done` under the OLD model, while the run's JSON export still gets
    stamped with the NEW `model_id` — a file that lies about which
    checkpoint produced it, with no crash and no log line."""
    if force:
        return True
    if existing_run is None:
        return True
    if existing_run.get("asr_status") != STATUS_DONE:
        return True
    return existing_run.get("model_id") != model_id


def to_frame_window(segment: AsrSegment, fps: float, frame_count: int) -> FrameWindow:
    """floor on the left edge, ceil on the right edge — the window only ever
    *widens*, never drops a frame a TRAKE answer could need. `round()` is
    forbidden here (CLAUDE.md T2)."""
    start_frame_idx = max(0, math.floor(segment.start * fps))
    end_frame_idx = min(frame_count - 1, math.ceil(segment.end * fps))
    return FrameWindow(start_frame_idx=start_frame_idx, end_frame_idx=end_frame_idx)


class FasterWhisperClient:
    """Wraps `faster_whisper.WhisperModel.transcribe()` (references.md §5.3,
    verified against faster-whisper==1.2.1). Audio is decoded straight from
    the `.mp4` via PyAV (bundled with the package) — no separate extraction
    step."""

    def __init__(self, config: AsrConfig = _DEFAULT_ASR_CONFIG) -> None:
        # lazy: only needed when this class is used
        from faster_whisper import BatchedInferencePipeline, WhisperModel

        model = WhisperModel(
            config.model_id, device=config.device, compute_type=config.compute_type
        )
        # Batched decodes VAD chunks of the SAME video in one pass instead of
        # one chunk at a time; sequential is kept as an escape hatch
        # (faster-whisper#1179 reports dropped segments on some inputs).
        self._model = BatchedInferencePipeline(model=model) if config.batched else model
        self._config = config

    def transcribe(self, video_path: str) -> list[AsrSegment]:
        # batch_size is rejected by WhisperModel.transcribe() -- only the
        # batched path accepts it, so it is passed conditionally.
        extra = {"batch_size": self._config.batch_size} if self._config.batched else {}
        segments, _info = self._model.transcribe(
            video_path,
            beam_size=self._config.beam_size,
            language=self._config.language,
            vad_filter=self._config.vad_filter,
            condition_on_previous_text=self._config.condition_on_previous_text,
            word_timestamps=self._config.word_timestamps,
            **extra,
        )
        return [
            AsrSegment(
                start=segment.start,
                end=segment.end,
                text=segment.text,
                no_speech_prob=segment.no_speech_prob,
                avg_logprob=segment.avg_logprob,
                words=(
                    tuple(
                        Word(start=w.start, end=w.end, word=w.word, probability=w.probability)
                        for w in segment.words
                    )
                    if segment.words is not None
                    else None
                ),
            )
            for segment in segments
        ]


def resegment_by_pause(
    words: Sequence[Word],
    *,
    min_pause_sec: float,
    no_speech_prob: float,
    avg_logprob: float,
) -> list[AsrSegment]:
    """Re-cuts one long batched-mode segment into sentence-length pieces
    using the gap between consecutive words as the cut signal — the only
    signal available, since batched mode's segment boundary comes from the
    VAD chunk, not from Whisper's own token timestamps (`AsrConfig.
    word_timestamps` docstring; faster_whisper==1.2.1 transcribe.py:398-409
    hard-codes `max_speech_duration_s=chunk_length`, discarding any
    caller-supplied value).

    `no_speech_prob`/`avg_logprob` are the ORIGINAL segment's values, carried
    onto every sub-segment unchanged — batched mode already coarsens both to
    one pair per VAD chunk (plan finding F2), so cutting the text finer
    cannot manufacture a per-sub-segment confidence that was never measured.
    A caller who wants finer confidence must set `batched=False`.
    """
    if not words:
        return []

    groups: list[list[Word]] = [[words[0]]]
    # words/words[1:] intentionally differ by one element -- this pairs
    # consecutive words to inspect the gap between them, so zip's shorter-
    # wins truncation is correct here, not a bug (strict=False documents that).
    for previous, current in zip(words, words[1:], strict=False):
        if current.start - previous.end >= min_pause_sec:
            groups.append([])
        groups[-1].append(current)

    return [
        AsrSegment(
            start=group[0].start,
            end=group[-1].end,
            text="".join(w.word for w in group),
            no_speech_prob=no_speech_prob,
            avg_logprob=avg_logprob,
        )
        for group in groups
    ]


async def run_asr(
    video_id: str,
    segments: Sequence[AsrSegment],
    *,
    es: AsrEsStore,
    fps: float,
    frame_count: int,
    config: AsrConfig = _DEFAULT_ASR_CONFIG,
) -> AsrRunSummary:
    """Filters hallucinated segments out of one video's transcript, then
    bulk-indexes the rest to Elasticsearch at segment granularity.
    `segment_id` is derived from the segment's position in the *original*
    transcript (not the filtered list), so it stays stable regardless of how
    many segments get dropped."""
    documents: list[AsrSegmentDocument] = []
    dropped_no_speech = 0
    dropped_low_confidence = 0
    dropped_short_text = 0

    for index, segment in enumerate(segments):
        reason = _hallucination_reason(segment, config)
        if reason == _REASON_NO_SPEECH:
            dropped_no_speech += 1
            continue
        if reason == _REASON_LOW_CONFIDENCE:
            dropped_low_confidence += 1
            continue
        if reason == _REASON_SHORT_TEXT:
            dropped_short_text += 1
            continue

        window = to_frame_window(segment, fps, frame_count)
        # Normalized AFTER the hallucination filter above, which must judge
        # Whisper's own raw output -- normalizing first would let a shrunk
        # "sáu mươi giây" -> "60 giây" cross min_text_length differently
        # than the un-normalized text the filter actually saw.
        documents.append(
            AsrSegmentDocument(
                segment_id=f"{video_id}_seg{index}",
                video_id=video_id,
                start_time=segment.start,
                end_time=segment.end,
                text=normalize_spoken_numbers(segment.text),
                start_frame_idx=window.start_frame_idx,
                end_frame_idx=window.end_frame_idx,
            )
        )

    await es.bulk_index(documents)

    return AsrRunSummary(
        total_segments=len(segments),
        kept_segments=len(documents),
        dropped_no_speech=dropped_no_speech,
        dropped_low_confidence=dropped_low_confidence,
        dropped_short_text=dropped_short_text,
    )
