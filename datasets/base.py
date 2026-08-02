from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class VideoRef:
    video_id: str
    source_path: str
    duration_sec: float
    category: str | None = None
    # Optional so any DatasetAdapter without this metadata (e.g. a future
    # dataset, or AIC2026Adapter's own no-metadata fallback) can simply leave
    # these None — MongoDB's `videos` collection (architecture.md §4.3) wants
    # them, but no pipeline code may depend on them being present.
    title: str | None = None
    publish_date: str | None = None
    watch_url: str | None = None
    channel_id: str | None = None


@dataclass(frozen=True)
class GroundTruthSegment:
    video_id: str
    start_sec: float
    end_sec: float
    order_index: int  # position within the event sequence — TRAKE ground truth
    caption: str  # source query for Textual KIS


class DatasetAdapter(Protocol):
    def iter_videos(self, split: str) -> Iterator[VideoRef]: ...
    def load_ground_truth(self, split: str) -> Sequence[GroundTruthSegment]: ...
