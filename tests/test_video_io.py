"""RED reproducer for pipelines/video_io.py (P0.7).

Builds a synthetic video with a fractional fps (29.97, like real broadcast
footage) so `test_iter_frames_rgb_frame_idx_survives_fractional_fps` can prove
`frame_idx` comes from the decode counter, not `round(timestamp * fps)` —
CLAUDE.md's sharpest "silent failure" invariant for this phase.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from pipelines.video_io import (
    frame_idx_to_timestamp,
    iter_frames_rgb,
    read_video_info,
    resize_for_transnetv2,
)

_FRACTIONAL_FPS = 29.97
_FRAME_COUNT = 20
_FRAME_SIZE = (64, 48)  # (width, height)


@pytest.fixture()
def fractional_fps_video(tmp_path) -> str:
    path = str(tmp_path / "sample.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), _FRACTIONAL_FPS, _FRAME_SIZE)
    for i in range(_FRAME_COUNT):
        # Each frame's blue-channel value encodes its index (widely spaced so
        # it survives mp4v's lossy quantization) so a test can read back
        # "which frame index produced this pixel" independently of any
        # fps-based arithmetic.
        frame = np.zeros((_FRAME_SIZE[1], _FRAME_SIZE[0], 3), dtype=np.uint8)
        frame[:, :, 0] = i * 12
        writer.write(frame)
    writer.release()
    return path


def test_iter_frames_rgb_frame_idx_survives_fractional_fps(fractional_fps_video):
    """`round(timestamp * fps)` drifts off the true index on a 29.97fps video
    long before frame 20; the decode counter must not."""
    for frame_idx, frame in iter_frames_rgb(fractional_fps_video):
        encoded_value = int(frame[0, 0, 2])  # BGR blue channel -> RGB red channel
        # Tolerant of mp4v's lossy quantization noise; still fails hard if the
        # decoder skips/reorders frames instead of counting sequentially.
        assert abs(encoded_value - frame_idx * 12) <= 15


def test_iter_frames_rgb_yields_sequential_indices_from_zero(fractional_fps_video):
    indices = [frame_idx for frame_idx, _ in iter_frames_rgb(fractional_fps_video)]
    assert indices == list(range(len(indices)))
    assert len(indices) == _FRAME_COUNT


def test_iter_frames_rgb_channel_order_is_rgb_not_bgr(tmp_path):
    path = str(tmp_path / "red.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 25.0, _FRAME_SIZE)
    frame_bgr = np.zeros((_FRAME_SIZE[1], _FRAME_SIZE[0], 3), dtype=np.uint8)
    frame_bgr[:, :, 2] = 255  # pure red, stored BGR-style (OpenCV convention)
    writer.write(frame_bgr)
    writer.release()

    _, decoded = next(iter_frames_rgb(path))
    assert decoded[0, 0, 0] > 200  # red channel dominant after BGR->RGB
    assert decoded[0, 0, 2] < 50


def test_iter_frames_rgb_raises_on_missing_file():
    with pytest.raises(OSError):
        next(iter_frames_rgb("does_not_exist.mp4"))


def test_read_video_info_returns_fps_and_frame_count(fractional_fps_video):
    info = read_video_info(fractional_fps_video)
    assert info.fps == pytest.approx(_FRACTIONAL_FPS, abs=0.1)
    assert info.frame_count == _FRAME_COUNT


def test_resize_for_transnetv2_output_shape_and_dtype():
    frame = np.zeros((_FRAME_SIZE[1], _FRAME_SIZE[0], 3), dtype=np.uint8)
    resized = resize_for_transnetv2(frame)
    assert resized.shape == (27, 48, 3)
    assert resized.dtype == np.uint8


def test_frame_idx_to_timestamp_is_frame_idx_over_fps():
    assert frame_idx_to_timestamp(frame_idx=30, fps=25.0) == pytest.approx(1.2)
