from datetime import date
from pathlib import Path

import pytest

from datasets.aic2025 import AIC2025Adapter

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "aic2025_sample"
MISSING_EVAL_SET_PATH = FIXTURE_ROOT / "does_not_exist.json"
EVAL_SET_PATH = FIXTURE_ROOT / "eval_set_v1_sample.json"


def make_adapter(eval_set_path: Path = MISSING_EVAL_SET_PATH) -> AIC2025Adapter:
    return AIC2025Adapter(dataset_root=FIXTURE_ROOT, eval_set_path=eval_set_path)


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
    assert videos["L30_V001"].source_path.endswith("Video_L30_a/video/L30_V001.mp4")


def test_pilot_split_contains_only_l21_to_l24() -> None:
    adapter = make_adapter()

    pilot_ids = {v.video_id for v in adapter.iter_videos(split="pilot")}

    assert pilot_ids == {"L21_V001", "L24_V002"}


def test_duration_sec_is_read_from_media_info_length_field() -> None:
    adapter = make_adapter()

    videos = {v.video_id: v for v in adapter.iter_videos(split="all")}

    assert videos["L21_V001"].duration_sec == 120.0
    assert videos["L21_V001"].category == "Channel A"


def test_publish_date_parsed_as_day_first() -> None:
    adapter = make_adapter()

    # "05/03/2024" is DD/MM/YYYY -> 5 March 2024, not 3 May
    assert adapter.publish_date("L21_V001") == date(2024, 3, 5)


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
