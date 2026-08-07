"""Compares per-video `num_segments` between two `asr_transcripts.json`-shaped
exports (`pipelines/export_asr.py`'s `build_asr_export()` output) —
research-asr-phowhisper-batched.plan.md T6b.

A free, pure-function check against faster-whisper#1179 (batched inference
silently dropping segments on some inputs): both exports already carry
`num_segments` per video, so this needs no GPU, no re-transcription, and no
manual labels — it is meant to run right after switching `batched: true`,
comparing against the large-v3-sequential baseline JSON saved before that
switch.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SegmentCountDiff:
    """One video's segment count, before vs after. `before`/`after` are 0
    when the video is absent from that side's export — absence is itself
    meaningful (a video that vanished, or one transcribed for the first
    time), not something to skip."""

    video_id: str
    before: int
    after: int

    @property
    def delta(self) -> int:
        return self.after - self.before

    @property
    def dropped(self) -> bool:
        """True when this video lost segments — the faster-whisper#1179
        signature. Equal or higher counts are never flagged: this check is
        for regressions, not for confirming growth."""
        return self.delta < 0


def compare_segment_counts(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> list[SegmentCountDiff]:
    """Both arguments are `build_asr_export()`-shaped dicts (or JSON loaded
    straight from an `asr_transcripts*.json` file). Returns one
    `SegmentCountDiff` per `video_id` seen in either export, sorted by
    `video_id` so output order is stable across runs."""
    before_counts = {video["video_id"]: video["num_segments"] for video in before["videos"]}
    after_counts = {video["video_id"]: video["num_segments"] for video in after["videos"]}
    video_ids = sorted(set(before_counts) | set(after_counts))
    return [
        SegmentCountDiff(
            video_id=video_id,
            before=before_counts.get(video_id, 0),
            after=after_counts.get(video_id, 0),
        )
        for video_id in video_ids
    ]
