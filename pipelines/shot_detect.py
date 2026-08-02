"""ShotDetector — TransNetV2 wrapper (P0.7, architecture.md §3.2).

Detects shot-transition boundaries so pipelines/keyframe.py can sample 4
keyframes per shot per the paper formula (PRD §F1). Ports the upstream
sliding-window algorithm (`inference/transnetv2.py`'s `predict_frames` /
`predictions_to_scenes`) verbatim in logic, but frames come from
pipelines/video_io.py's own sequential OpenCV decode instead of upstream's
ffmpeg pipe: this project uses exactly one decoder for both shot detection and
later keyframe extraction, so `frame_idx` never depends on two independent
decoders agreeing on frame count (CLAUDE.md's `frame_idx` invariant).

See scripts/setup_transnetv2.sh for how to obtain the checkpoint.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import torch

from pipelines.vendor.transnetv2_upstream.transnetv2_pytorch import TransNetV2
from pipelines.video_io import iter_frames_rgb, resize_for_transnetv2

_WINDOW_SIZE = 100
_STRIDE = 50
_CONTEXT = 25
_SCENE_THRESHOLD = 0.5


@dataclass(frozen=True)
class Shot:
    """Inclusive frame range `[start_frame, end_frame]` of one detected shot."""

    start_frame: int
    end_frame: int


def pad_frames(frames: np.ndarray) -> np.ndarray:
    """Pads with copies of the first/last frame so every real frame falls in
    the unpadded middle `_STRIDE` frames of some window (upstream's own
    padding scheme, `inference/transnetv2.py::predict_frames`)."""
    pad_end = _CONTEXT + _STRIDE - (len(frames) % _STRIDE or _STRIDE)
    return np.concatenate(
        [
            np.repeat(frames[:1], _CONTEXT, axis=0),
            frames,
            np.repeat(frames[-1:], pad_end, axis=0),
        ]
    )


def sliding_windows(padded_frames: np.ndarray) -> Iterator[np.ndarray]:
    """Yields overlapping `_WINDOW_SIZE`-frame windows, `_STRIDE` frames apart."""
    pointer = 0
    while pointer + _WINDOW_SIZE <= len(padded_frames):
        yield padded_frames[pointer : pointer + _WINDOW_SIZE]
        pointer += _STRIDE


def predictions_to_shots(predictions: np.ndarray) -> list[Shot]:
    """Direct port of upstream `predictions_to_scenes` (soCzech/TransNetV2,
    `inference/transnetv2.py`): binarizes at `_SCENE_THRESHOLD`, then a shot
    is a run of below-threshold frames closed by the first above-threshold
    frame that follows it. Kept as upstream defines it (not "fixed") so shot
    boundaries match the reference implementation exactly."""
    is_cut = (predictions > _SCENE_THRESHOLD).astype(np.uint8)

    shots: list[list[int]] = []
    t_prev, start, i, t = 0, 0, 0, 0
    for i, t in enumerate(is_cut):
        if t_prev == 1 and t == 0:
            start = i
        if t_prev == 0 and t == 1 and i != 0:
            shots.append([start, i])
        t_prev = t
    if t == 0:
        shots.append([start, i])

    if not shots:
        return [Shot(start_frame=0, end_frame=len(predictions) - 1)]
    return [Shot(start_frame=s, end_frame=e) for s, e in shots]


class ShotDetector:
    """Runs TransNetV2 over a whole video and returns its shot boundaries."""

    def __init__(self, checkpoint_path: str, device: str = "cuda") -> None:
        self._device = torch.device(device)
        self._model = TransNetV2()
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self._model.load_state_dict(state_dict)
        self._model.to(self._device).eval()

    def detect_shots(self, video_path: str) -> list[Shot]:
        frames = np.stack(
            [resize_for_transnetv2(frame) for _, frame in iter_frames_rgb(video_path)]
        )
        if len(frames) == 0:
            raise ValueError(f"Video has no readable frames: {video_path}")
        predictions = self._predict_frames(frames)
        return predictions_to_shots(predictions)

    def _predict_frames(self, frames: np.ndarray) -> np.ndarray:
        predictions = []
        with torch.no_grad():
            for window in sliding_windows(pad_frames(frames)):
                window_tensor = torch.from_numpy(window).unsqueeze(0).to(self._device)
                logits, _ = self._model(window_tensor)
                probs = torch.sigmoid(logits)[0, _CONTEXT : _CONTEXT + _STRIDE, 0]
                predictions.append(probs.cpu().numpy())
        return np.concatenate(predictions)[: len(frames)]
