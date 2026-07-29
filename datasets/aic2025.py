import json
from collections.abc import Iterator, Sequence
from datetime import date, datetime
from pathlib import Path

from datasets.base import GroundTruthSegment, VideoRef

PILOT_BATCHES = frozenset({"L21", "L22", "L23", "L24"})
_VALID_SPLITS = frozenset({"pilot", "all"})


class AIC2025Adapter:
    """Reads the local AIC 2025 public dataset (`references.md` §2).

    `dataset_root` is the version directory that contains `Videos/Videos/`
    and `features/` (e.g.
    `data/datasets/aresusayhi/ai-challenge-2025/versions/2`).
    `eval_set_path` points at the self-generated ground truth produced by
    `eval/build_eval_set.py` (PRD §6.1) — the dataset itself ships no labels.
    """

    def __init__(self, dataset_root: Path, eval_set_path: Path) -> None:
        self._dataset_root = dataset_root
        self._eval_set_path = eval_set_path
        self._video_paths: dict[str, Path] | None = None

    def iter_videos(self, split: str) -> Iterator[VideoRef]:
        _validate_split(split)
        for video_id, video_path in sorted(self._discover_videos().items()):
            if split == "pilot" and _batch_of(video_id) not in PILOT_BATCHES:
                continue
            media_info = self._load_media_info(video_id)
            yield VideoRef(
                video_id=video_id,
                source_path=str(video_path),
                duration_sec=float(media_info["length"]),
                category=media_info.get("author"),
            )

    def load_ground_truth(self, split: str) -> Sequence[GroundTruthSegment]:
        _validate_split(split)
        if not self._eval_set_path.exists():
            raise FileNotFoundError(
                f"Eval set not found at {self._eval_set_path}. AIC 2025 ships no "
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
            if split == "all" or _batch_of(record["video_id"]) in PILOT_BATCHES
        ]

    def publish_date(self, video_id: str) -> date:
        """Parses `media-info.publish_date` — format is `DD/MM/YYYY`, never
        ISO-8601, so it must be parsed explicitly rather than guessed."""
        raw = self._load_media_info(video_id)["publish_date"]
        return datetime.strptime(raw, "%d/%m/%Y").date()

    def _discover_videos(self) -> dict[str, Path]:
        if self._video_paths is None:
            paths: dict[str, Path] = {}
            # Directory names are inconsistent (`Videos_L21_a` vs the
            # singular `Video_L30_a`) so this must glob, not format a path.
            for mp4_path in sorted(self._dataset_root.glob("Videos/Videos/Video*/video/*.mp4")):
                paths.setdefault(mp4_path.stem, mp4_path)
            self._video_paths = paths
        return self._video_paths

    def _load_media_info(self, video_id: str) -> dict:
        info_path = self._dataset_root / "features" / "media-info" / f"{video_id}.json"
        with info_path.open(encoding="utf-8") as f:
            return json.load(f)


def _batch_of(video_id: str) -> str:
    return video_id.split("_V", 1)[0]


def _validate_split(split: str) -> None:
    if split not in _VALID_SPLITS:
        raise ValueError(f"split must be one of {sorted(_VALID_SPLITS)}, got {split!r}")
