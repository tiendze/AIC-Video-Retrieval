"""Sequential video frame decoding — the single source of truth for `frame_idx`.

CLAUDE.md invariant: `frame_idx` must come from the decoder itself, never from
`timestamp * fps` (rounding a fractional fps like 29.97 into the wrong frame
silently trips the TRAKE submission window). This module is the one place
video frames are read, so every other pipeline module (shot_detect, keyframe
extraction, embed) sources frame indices from here instead of recomputing
them. `frame_idx` is a counter incremented on each successful decode, never
`CAP_PROP_POS_FRAMES` (which can drift on variable-frame-rate video).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import cv2
import numpy as np

_TRANSNETV2_INPUT_SIZE = (48, 27)  # (width, height) per references.md §3.1


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    frame_count: int


def read_video_info(video_path: str) -> VideoInfo:
    """Reads `fps` — the one conversion constant `frame_idx -> timestamp` must
    use (CLAUDE.md §3: `timestamp` is derived, never the other way around)."""
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise OSError(f"Could not open video file: {video_path}")
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if fps <= 0:
        raise ValueError(f"Video reports non-positive fps ({fps}): {video_path}")
    return VideoInfo(fps=fps, frame_count=frame_count)


def iter_frames_rgb(video_path: str) -> Iterator[tuple[int, np.ndarray]]:
    """Sequentially decodes every frame, yielding `(frame_idx, frame_rgb)`."""
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise OSError(f"Could not open video file: {video_path}")
    try:
        frame_idx = 0
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                return
            yield frame_idx, cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_idx += 1
    finally:
        capture.release()


def resize_for_transnetv2(frame_rgb: np.ndarray) -> np.ndarray:
    """Resizes to the model's fixed 48(width) x 27(height) input, dtype uint8
    (references.md §3.1 — TransNetV2 asserts on any other shape/dtype)."""
    return cv2.resize(frame_rgb, _TRANSNETV2_INPUT_SIZE, interpolation=cv2.INTER_AREA)


def frame_idx_to_timestamp(frame_idx: int, fps: float) -> float:
    """The one allowed direction of conversion — `timestamp` is for display
    and eval hit-testing only, never for submission (CLAUDE.md §3)."""
    return frame_idx / fps
