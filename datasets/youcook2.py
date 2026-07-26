import json
from collections.abc import Iterator, Sequence
from pathlib import Path

from datasets.base import GroundTruthSegment, VideoRef


class YouCook2Adapter:
    """Reads `youcookii_annotations_trainval.json` (PRD §6.1 schema)."""

    def __init__(self, annotations_path: Path, video_dir: Path) -> None:
        self._annotations_path = annotations_path
        self._video_dir = video_dir
        self._database: dict | None = None

    def iter_videos(self, split: str) -> Iterator[VideoRef]:
        for video_id, entry in self._load_database().items():
            if entry["subset"] != split:
                continue
            yield VideoRef(
                video_id=video_id,
                source_path=str(self._video_dir / f"{video_id}.mp4"),
                duration_sec=float(entry["duration"]),
                category=entry.get("recipe_type"),
            )

    def load_ground_truth(self, split: str) -> Sequence[GroundTruthSegment]:
        segments: list[GroundTruthSegment] = []
        for video_id, entry in self._load_database().items():
            if entry["subset"] != split:
                continue
            for order_index, annotation in enumerate(entry["annotations"]):
                start_sec, end_sec = annotation["segment"]
                segments.append(
                    GroundTruthSegment(
                        video_id=video_id,
                        start_sec=float(start_sec),
                        end_sec=float(end_sec),
                        order_index=order_index,
                        caption=annotation["sentence"],
                    )
                )
        return segments

    def video_url(self, video_id: str) -> str:
        """YouTube source link — used by scripts/download_videos.py, not part
        of the DatasetAdapter Protocol (AIC's adapter may not have a URL)."""
        return self._load_database()[video_id]["video_url"]

    def _load_database(self) -> dict:
        if self._database is None:
            with self._annotations_path.open(encoding="utf-8") as f:
                self._database = json.load(f)["database"]
        return self._database
