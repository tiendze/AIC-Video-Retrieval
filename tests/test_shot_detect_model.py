"""Model-gated integration test for pipelines/shot_detect.py (P0.7).

Split from tests/test_shot_detect.py because a module-level `pytest.skip(...,
allow_module_level=True)` skips the *whole* module, which would also hide the
pure-logic tests that need no checkpoint. Requires a real checkpoint
(scripts/setup_transnetv2.sh) — same pattern as tests/test_beit3_encoder.py.

    TRANSNETV2_CHECKPOINT_PATH=models/transnetv2/transnetv2-pytorch-weights.pth \
    pytest -m model tests/test_shot_detect_model.py -v
"""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.model

_CHECKPOINT_PATH = os.environ.get("TRANSNETV2_CHECKPOINT_PATH")

if not _CHECKPOINT_PATH:
    pytest.skip(
        "TRANSNETV2_CHECKPOINT_PATH not set — run scripts/setup_transnetv2.sh first",
        allow_module_level=True,
    )


@pytest.fixture(scope="module")
def detector():
    from pipelines.shot_detect import ShotDetector

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return ShotDetector(checkpoint_path=_CHECKPOINT_PATH, device=device)


def test_detect_shots_on_real_pilot_video_covers_whole_video(detector):
    from pipelines.video_io import read_video_info

    video_path = "tests/fixtures/aic2026_sample/Videos_L21_a/video/L21_V001.mp4"
    info = read_video_info(video_path)

    shots = detector.detect_shots(video_path)

    assert len(shots) >= 1
    assert shots[0].start_frame == 0
    assert shots[-1].end_frame == info.frame_count - 1
    for prev_shot, next_shot in zip(shots, shots[1:], strict=False):
        assert next_shot.start_frame == prev_shot.end_frame + 1
