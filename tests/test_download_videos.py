import json
from pathlib import Path

import pytest

from datasets.base import VideoRef
from scripts.download_videos import (
    DownloadError,
    append_failed_download,
    download_split,
    freeze_available_videos,
    is_already_downloaded,
)


class FakeAdapter:
    def __init__(self, videos: list[VideoRef]) -> None:
        self._videos = videos

    def iter_videos(self, split: str) -> list[VideoRef]:
        return list(self._videos)


def make_video(video_id: str, video_dir: Path) -> VideoRef:
    return VideoRef(
        video_id=video_id,
        source_path=str(video_dir / f"{video_id}.mp4"),
        duration_sec=60.0,
    )


def test_is_already_downloaded_checks_the_video_source_path(tmp_path: Path) -> None:
    video = make_video("vid001", tmp_path)

    assert is_already_downloaded(video) is False

    Path(video.source_path).write_bytes(b"fake video bytes")

    assert is_already_downloaded(video) is True


def test_download_split_skips_videos_already_on_disk(tmp_path: Path) -> None:
    already_downloaded = make_video("vid001", tmp_path)
    Path(already_downloaded.source_path).write_bytes(b"already here")
    to_download = make_video("vid002", tmp_path)
    adapter = FakeAdapter([already_downloaded, to_download])
    calls: list[str] = []

    def fake_downloader(video: VideoRef) -> None:
        calls.append(video.video_id)
        Path(video.source_path).write_bytes(b"downloaded")

    available = download_split(
        adapter=adapter,
        split="validation",
        failed_path=tmp_path / "failed_downloads.jsonl",
        downloader=fake_downloader,
    )

    assert calls == ["vid002"]  # never re-downloaded vid001
    assert set(available) == {"vid001", "vid002"}


def test_download_split_logs_failure_and_continues(tmp_path: Path) -> None:
    broken = make_video("vid_broken", tmp_path)
    healthy = make_video("vid_healthy", tmp_path)
    adapter = FakeAdapter([broken, healthy])
    failed_path = tmp_path / "failed_downloads.jsonl"

    def flaky_downloader(video: VideoRef) -> None:
        if video.video_id == "vid_broken":
            raise DownloadError("video unavailable (region locked)")
        Path(video.source_path).write_bytes(b"downloaded")

    available = download_split(
        adapter=adapter, split="validation", failed_path=failed_path, downloader=flaky_downloader
    )

    assert available == ["vid_healthy"]
    failed_lines = failed_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(failed_lines) == 1
    failed_entry = json.loads(failed_lines[0])
    assert failed_entry == {
        "video_id": "vid_broken",
        "reason": "video unavailable (region locked)",
    }


def test_append_failed_download_is_additive_across_calls(tmp_path: Path) -> None:
    failed_path = tmp_path / "failed_downloads.jsonl"

    append_failed_download(failed_path, "vid_a", "reason A")
    append_failed_download(failed_path, "vid_b", "reason B")

    lines = failed_path.read_text(encoding="utf-8").strip().splitlines()
    assert [json.loads(line)["video_id"] for line in lines] == ["vid_a", "vid_b"]


def test_freeze_available_videos_writes_sorted_ids_and_a_matching_hash(tmp_path: Path) -> None:
    output_path = tmp_path / "val_available_v1.txt"

    digest = freeze_available_videos(["vid002", "vid001"], output_path)

    assert output_path.read_text(encoding="utf-8") == "vid001\nvid002\n"
    hash_path = output_path.with_suffix(output_path.suffix + ".sha256")
    assert hash_path.read_text(encoding="utf-8").strip() == digest
    assert len(digest) == 64  # sha256 hex digest length


@pytest.mark.parametrize("videos", [[], ["only_one"]])
def test_freeze_available_videos_handles_edge_cases(tmp_path: Path, videos: list[str]) -> None:
    output_path = tmp_path / "val_available_v1.txt"

    freeze_available_videos(videos, output_path)

    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert lines == videos
