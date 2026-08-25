"""Reader for the local AIC 2026 public dataset (`references.md` §2).

The prelim-round data is AIC 2026 batch 1 (identical content to AIC 2025
batch 1); batch 2 is announced but not released yet, so `iter_videos` must
stay batch-agnostic and simply report whatever is on disk.

Only the videos are mandatory. Keyframes / objects / CLIP features are
re-derived by this project's own models, and per the official data notice
some videos ship no metadata file at all — so metadata is read when present
and substituted from `map-keyframes` when not, rather than being required.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from datasets.base import GroundTruthSegment, VideoRef, batch_of

PILOT_BATCHES = frozenset({"L21", "L22", "L23", "L24"})
_VALID_SPLITS = frozenset({"pilot", "all"})

# The download ships metadata under `metadata/` while the older Kaggle mirror
# used `features/media-info/`; both are accepted so a re-download of either
# layout works without touching the pipeline.
_METADATA_DIR_NAMES = ("metadata", "media-info")
_MAP_KEYFRAMES_DIR_NAME = "map-keyframes"

_PUBLISH_DATE_FORMAT = "%d/%m/%Y"


@dataclass(frozen=True)
class ReferenceKeyframe:
    """One row of the organiser's `map-keyframes/<video_id>.csv`.

    These are the *organiser's* keyframes, used only as an independent
    control when measuring this project's own shot cuts (PRD §6.2, R10) —
    never as an ingest source.
    """

    n: int
    pts_time: float
    fps: float
    frame_idx: int


class AIC2026Adapter:
    """Reads the local AIC 2026 public dataset.

    `dataset_root` is the directory holding the per-shard video folders
    (e.g. `data/`, containing `Videos_L21_a/video/*.mp4`). Sibling
    `metadata/` and `map-keyframes/` folders are discovered by globbing, at
    any depth, because the release nests them inconsistently.

    `eval_set_path` points at the self-generated ground truth produced by
    `eval/build_eval_set.py` (PRD §6.1) — the dataset itself ships no labels.
    """

    def __init__(self, dataset_root: Path, eval_set_path: Path) -> None:
        self._dataset_root = dataset_root
        self._eval_set_path = eval_set_path
        self._video_paths: dict[str, Path] | None = None
        self._metadata_paths: dict[str, Path] | None = None
        self._map_keyframe_paths: dict[str, Path] | None = None

    def iter_videos(self, split: str) -> Iterator[VideoRef]:
        _validate_split(split)
        for video_id, video_path in sorted(self._discover_videos().items()):
            if split == "pilot" and batch_of(video_id) not in PILOT_BATCHES:
                continue
            metadata = self._load_metadata(video_id)
            yield VideoRef(
                video_id=video_id,
                source_path=str(video_path),
                duration_sec=self._duration_sec(video_id, metadata),
                category=metadata.get("author") if metadata else None,
                title=metadata.get("title") if metadata else None,
                publish_date=metadata.get("publish_date") if metadata else None,
                watch_url=metadata.get("watch_url") if metadata else None,
                channel_id=metadata.get("channel_id") if metadata else None,
            )

    def load_ground_truth(self, split: str) -> Sequence[GroundTruthSegment]:
        _validate_split(split)
        if not self._eval_set_path.exists():
            raise FileNotFoundError(
                f"Eval set not found at {self._eval_set_path}. AIC 2026 ships no "
                "ground truth — run eval/build_eval_set.py first (PRD §6.1)."
            )
        with self._eval_set_path.open(encoding="utf-8") as f:
            records = json.load(f)
        return [
            GroundTruthSegment(
                video_id=record["video_id"],
                start_sec=float(record["start_sec"]),
                end_sec=float(record["end_sec"]),
                order_index=int(record["order_index"]),
                caption=record["caption"],
            )
            for record in records
            if split == "all" or batch_of(record["video_id"]) in PILOT_BATCHES
        ]

    def publish_date(self, video_id: str) -> date:
        """Parses `publish_date` — format is `DD/MM/YYYY`, never ISO-8601, so
        it must be parsed explicitly rather than guessed.

        Raises `FileNotFoundError` when the video ships no metadata file: an
        absent publish date is missing data, not an empty result.
        """
        metadata = self._load_metadata(video_id)
        if metadata is None:
            raise FileNotFoundError(
                f"No metadata file for {video_id}; publish_date is unavailable."
        )
        return datetime.strptime(metadata["publish_date"], _PUBLISH_DATE_FORMAT).date()

    def load_metadata(self, video_id: str) -> dict | None:
        """Returns the organiser metadata JSON for `video_id`, when present.

        This is separate from `iter_videos()` because some import paths have
        keyframe/vector artifacts and metadata locally, but not the raw `.mp4`
        files that `iter_videos()` uses for video discovery.
        """
        return self._load_metadata(video_id)

    def load_reference_keyframes(self, video_id: str) -> Sequence[ReferenceKeyframe]:
        """Reads `map-keyframes/<video_id>.csv` — every value is a string on
        disk and must be cast explicitly (`references.md` §2)."""
        path = self._discover_map_keyframes().get(video_id)
        if path is None:
            raise FileNotFoundError(
                f"No map-keyframes CSV for {video_id} under {self._dataset_root}."
            )
        with path.open(encoding="utf-8", newline="") as f:
            return [
                ReferenceKeyframe(
                    n=int(row["n"]),
                    pts_time=float(row["pts_time"]),
                    fps=float(row["fps"]),
                    frame_idx=int(row["frame_idx"]),
                )
                for row in csv.DictReader(f)
            ]

    def _duration_sec(self, video_id: str, metadata: dict | None) -> float:
        """`media-info.length` when available (seconds, int), else derived
        from the last `map-keyframes` row.

        The fallback is a *lower bound* — it only reaches the last organiser
        keyframe, not the true end of the video — so it is used for planning
        and reporting, never as a submission timestamp.
        """
        if metadata is not None:
            return float(metadata["length"])
        rows = self.load_reference_keyframes(video_id)
        if not rows:
            raise ValueError(f"map-keyframes CSV for {video_id} has no rows.")
        last = rows[-1]
        return last.frame_idx / last.fps

    def _discover_videos(self) -> dict[str, Path]:
        if self._video_paths is None:
            paths: dict[str, Path] = {}
            # Shard directory names are inconsistent (`Videos_L21_a` vs the
            # singular `Video_L30_a`) and the release has been published both
            # flat and nested, so this must glob rather than format a path.
            # `setdefault` keeps the first path per id: the release has
            # shipped duplicate `.mp4` files under the same `video_id`, and
            # counting files instead of ids would ingest them twice and
            # inflate the denominator of every metric.
            for mp4_path in sorted(self._dataset_root.rglob("Video*/video/*.mp4")):
                paths.setdefault(mp4_path.stem, mp4_path)
            self._video_paths = paths
        return self._video_paths

    def _discover_map_keyframes(self) -> dict[str, Path]:
        if self._map_keyframe_paths is None:
            self._map_keyframe_paths = self._index_by_video_id((_MAP_KEYFRAMES_DIR_NAME,), "*.csv")
        return self._map_keyframe_paths

    def _load_metadata(self, video_id: str) -> dict | None:
        if self._metadata_paths is None:
            self._metadata_paths = self._index_by_video_id(_METADATA_DIR_NAMES, "*.json")
        path = self._metadata_paths.get(video_id)
        if path is None:
            return None
        with path.open(encoding="utf-8") as f:
            return json.load(f)

    def _index_by_video_id(self, dir_names: tuple[str, ...], pattern: str) -> dict[str, Path]:
        paths: dict[str, Path] = {}
        for dir_name in dir_names:
            for path in sorted(self._dataset_root.rglob(f"{dir_name}/{pattern}")):
                paths.setdefault(path.stem, path)
        return paths


def _validate_split(split: str) -> None:
    if split not in _VALID_SPLITS:
        raise ValueError(f"split must be one of {sorted(_VALID_SPLITS)}, got {split!r}")
