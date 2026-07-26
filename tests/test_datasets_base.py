import dataclasses

import pytest

from datasets.base import VideoRef


def test_video_ref_is_immutable() -> None:
    ref = VideoRef(video_id="vid001", source_path="data/videos/vid001.mp4", duration_sec=120.5)

    with pytest.raises(dataclasses.FrozenInstanceError):
        ref.video_id = "changed"  # type: ignore[misc]
