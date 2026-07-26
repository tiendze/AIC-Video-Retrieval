"""Download the YouCook2 validation-split videos via yt-dlp.

Resumable and idempotent (PRD §7.3 / T1): videos already present on disk are
skipped, not re-downloaded. Videos that fail to download (removed link,
region lock, ...) are logged to `--failed-path` and the run continues —
never abort the whole split over one broken video.

Usage:
    python -m scripts.download_videos \\
        --annotations data/youcookii_annotations_trainval.json \\
        --video-dir data/videos \\
        --freeze-path configs/eval/val_available_v1.txt
"""

import argparse
import hashlib
import json
import logging
import subprocess
from collections.abc import Callable, Iterable
from pathlib import Path

from datasets.base import VideoRef
from datasets.youcook2 import YouCook2Adapter

logger = logging.getLogger(__name__)

DEFAULT_SPLIT = "validation"
DEFAULT_ANNOTATIONS_PATH = Path("data/youcookii_annotations_trainval.json")
DEFAULT_VIDEO_DIR = Path("data/videos")
DEFAULT_FAILED_PATH = Path("data/failed_downloads.jsonl")
DEFAULT_FREEZE_PATH = Path("configs/eval/val_available_v1.txt")


class DownloadError(Exception):
    """Raised by a `Downloader` when a video cannot be fetched."""


Downloader = Callable[[VideoRef], None]


def is_already_downloaded(video: VideoRef) -> bool:
    return Path(video.source_path).exists()


def append_failed_download(failed_path: Path, video_id: str, reason: str) -> None:
    failed_path.parent.mkdir(parents=True, exist_ok=True)
    with failed_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"video_id": video_id, "reason": reason}) + "\n")


def freeze_available_videos(video_ids: Iterable[str], output_path: Path) -> str:
    """Write the sorted, newline-joined list of available video_ids plus a
    sha256 digest file next to it. This list is the official denominator
    (PRD §6.0 R4) for every metric report downstream."""
    sorted_ids = sorted(video_ids)
    content = "".join(f"{video_id}\n" for video_id in sorted_ids)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")

    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    hash_path = output_path.with_suffix(output_path.suffix + ".sha256")
    hash_path.write_text(f"{digest}\n", encoding="utf-8")
    return digest


def download_split(
    adapter: YouCook2Adapter,
    split: str,
    failed_path: Path,
    downloader: Downloader,
) -> list[str]:
    """Download every video in `split`. Returns the video_ids that ended up
    available on disk (already-present + newly downloaded), in iteration
    order. Videos that fail are logged via `append_failed_download` and
    excluded from the result — the run always continues to the next video."""
    available: list[str] = []
    for video in adapter.iter_videos(split):
        if is_already_downloaded(video):
            available.append(video.video_id)
            continue
        try:
            downloader(video)
        except DownloadError as exc:
            logger.warning("download failed for %s: %s", video.video_id, exc)
            append_failed_download(failed_path, video.video_id, str(exc))
            continue
        available.append(video.video_id)
    return available


def make_yt_dlp_downloader(adapter: YouCook2Adapter) -> Downloader:
    """Real downloader: shells out to yt-dlp. Not exercised by unit tests
    (network I/O) — covered by tests/integration/ once available."""

    def _download(video: VideoRef) -> None:
        target = Path(video.source_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        url = adapter.video_url(video.video_id)
        result = subprocess.run(  # noqa: S603
            ["yt-dlp", "-f", "mp4", "-o", str(target), url],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise DownloadError(result.stderr.strip() or "yt-dlp exited non-zero")

    return _download


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS_PATH)
    parser.add_argument("--video-dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--failed-path", type=Path, default=DEFAULT_FAILED_PATH)
    parser.add_argument("--freeze-path", type=Path, default=DEFAULT_FREEZE_PATH)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()

    adapter = YouCook2Adapter(annotations_path=args.annotations, video_dir=args.video_dir)
    available = download_split(
        adapter=adapter,
        split=args.split,
        failed_path=args.failed_path,
        downloader=make_yt_dlp_downloader(adapter),
    )
    digest = freeze_available_videos(available, args.freeze_path)
    logger.info(
        "%d/%d videos available, frozen to %s (sha256=%s)",
        len(available),
        len(list(adapter.iter_videos(args.split))),
        args.freeze_path,
        digest,
    )


if __name__ == "__main__":
    main()
