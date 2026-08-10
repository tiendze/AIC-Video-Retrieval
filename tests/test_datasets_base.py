import dataclasses

import pytest

from datasets.base import VideoRef, batch_of


def test_video_ref_is_immutable() -> None:
    ref = VideoRef(video_id="vid001", source_path="data/videos/vid001.mp4", duration_sec=120.5)

    with pytest.raises(dataclasses.FrozenInstanceError):
        ref.video_id = "changed"  # type: ignore[misc]


def test_batch_of_derives_release_batch_from_video_id() -> None:
    assert batch_of("L21_V001") == "L21"
    assert batch_of("L24_V002") == "L24"
