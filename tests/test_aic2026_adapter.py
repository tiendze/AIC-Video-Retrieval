from datetime import date
from pathlib import Path

import pytest

from datasets.aic2026 import AIC2026Adapter
from datasets.base import VideoRef

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "aic2026_sample"
BARE_FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "aic2026_no_metadata"
MISSING_EVAL_SET_PATH = FIXTURE_ROOT / "does_not_exist.json"
EVAL_SET_PATH = FIXTURE_ROOT / "eval_set_v1_sample.json"


def make_adapter(
    eval_set_path: Path = MISSING_EVAL_SET_PATH, dataset_root: Path = FIXTURE_ROOT
) -> AIC2026Adapter:
    return AIC2026Adapter(dataset_root=dataset_root, eval_set_path=eval_set_path)


def _tail(video: VideoRef) -> tuple[str, ...]:
    """Last three path components, so assertions do not depend on the OS
    path separator."""
    return Path(video.source_path).parts[-3:]


def test_iter_videos_deduplicates_repeated_video_id() -> None:
    adapter = make_adapter()

    videos = list(adapter.iter_videos(split="all"))

    # fixture has 4 .mp4 files on disk (L21_V001 appears in two shard dirs)
    # but only 3 unique video_id
    assert len(videos) == 3
    assert len({v.video_id for v in videos}) == 3


def test_iter_videos_handles_singular_video_directory_name() -> None:
    adapter = make_adapter()

    videos = {v.video_id: v for v in adapter.iter_videos(split="all")}

    assert "L30_V001" in videos
    assert _tail(videos["L30_V001"]) == ("Video_L30_a", "video", "L30_V001.mp4")


def test_iter_videos_finds_shards_directly_under_dataset_root() -> None:
    # The 2026 download is flat (`data/Videos_L21_a/video/`), unlike the
    # earlier `Videos/Videos/Videos_L21_a/video/` nesting.
    adapter = make_adapter()

    videos = {v.video_id: v for v in adapter.iter_videos(split="all")}

    assert _tail(videos["L21_V001"]) == ("Videos_L21_a", "video", "L21_V001.mp4")


def test_pilot_split_contains_only_l21_to_l24() -> None:
    adapter = make_adapter()

    pilot_ids = {v.video_id for v in adapter.iter_videos(split="pilot")}

    assert pilot_ids == {"L21_V001", "L24_V002"}


def test_duration_sec_is_read_from_metadata_length_field() -> None:
    adapter = make_adapter()

    videos = {v.video_id: v for v in adapter.iter_videos(split="all")}

    assert videos["L21_V001"].duration_sec == 120.0
    assert videos["L21_V001"].category == "Channel A"


def test_duration_sec_falls_back_to_map_keyframes_when_metadata_missing() -> None:
    # Some videos ship no metadata file at all (official AIC 2026 data notice).
    adapter = make_adapter()

    videos = {v.video_id: v for v in adapter.iter_videos(split="all")}

    # last map-keyframes row of L30_V001: frame_idx 2250 at 25 fps
    assert videos["L30_V001"].duration_sec == 90.0
    assert videos["L30_V001"].category is None


def test_iter_videos_raises_when_no_metadata_and_no_map_keyframes() -> None:
    # Missing data must fail loudly rather than yield an invented duration.
    adapter = make_adapter(dataset_root=BARE_FIXTURE_ROOT)

    with pytest.raises(FileNotFoundError):
        list(adapter.iter_videos(split="all"))


def test_publish_date_parsed_as_day_first() -> None:
    adapter = make_adapter()

    # "05/03/2024" is DD/MM/YYYY -> 5 March 2024, not 3 May
    assert adapter.publish_date("L21_V001") == date(2024, 3, 5)


def test_publish_date_raises_when_metadata_missing() -> None:
    adapter = make_adapter()

    with pytest.raises(FileNotFoundError):
        adapter.publish_date("L30_V001")


def test_load_reference_keyframes_casts_every_column() -> None:
    adapter = make_adapter()

    rows = adapter.load_reference_keyframes("L21_V001")

    assert [r.n for r in rows] == [1, 2, 3]
    assert rows[2].pts_time == 8.7
    assert rows[2].fps == 30.0
    assert rows[2].frame_idx == 261


def test_load_ground_truth_raises_when_eval_set_missing() -> None:
    adapter = make_adapter(eval_set_path=MISSING_EVAL_SET_PATH)

    with pytest.raises(FileNotFoundError):
        adapter.load_ground_truth(split="pilot")


def test_load_ground_truth_returns_segments_from_eval_set_file() -> None:
    adapter = make_adapter(eval_set_path=EVAL_SET_PATH)

    segments = adapter.load_ground_truth(split="pilot")

    # L30_V001 record must be excluded — it is outside the pilot L21-L24 batches
    assert [s.video_id for s in segments] == ["L21_V001", "L21_V001"]
    assert [s.order_index for s in segments] == [0, 1]
