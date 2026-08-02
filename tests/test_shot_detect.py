"""RED reproducer for pipelines/shot_detect.py (P0.7).

`pad_frames` / `sliding_windows` / `predictions_to_shots` are pure numpy logic
ported from upstream TransNetV2 — tested here with no torch/checkpoint needed.
`ShotDetector.detect_shots` needs a real checkpoint (scripts/setup_transnetv2.sh)
and is gated behind `pytest.mark.model`, same pattern as tests/test_beit3_encoder.py.
"""

from __future__ import annotations

import numpy as np

from pipelines.shot_detect import (
    Shot,
    pad_frames,
    predictions_to_shots,
    sliding_windows,
)

_FRAME_SHAPE = (27, 48, 3)


def _fake_frames(n: int) -> np.ndarray:
    return np.stack([np.full(_FRAME_SHAPE, fill_value=i % 256, dtype=np.uint8) for i in range(n)])


def test_pad_frames_start_padding_is_25_copies_of_first_frame():
    frames = _fake_frames(10)
    padded = pad_frames(frames)
    np.testing.assert_array_equal(padded[:25], np.repeat(frames[:1], 25, axis=0))


def test_pad_frames_end_padding_length_is_between_25_and_74():
    for n in [1, 10, 50, 51, 99, 100, 101]:
        padded = pad_frames(_fake_frames(n))
        end_padding_len = len(padded) - 25 - n
        assert 25 <= end_padding_len <= 74, f"n={n} gave end padding {end_padding_len}"


def test_pad_frames_end_padding_is_copies_of_last_frame():
    frames = _fake_frames(10)
    padded = pad_frames(frames)
    end_padding = padded[25 + 10 :]
    np.testing.assert_array_equal(end_padding, np.repeat(frames[-1:], len(end_padding), axis=0))


def test_sliding_windows_have_correct_size_and_stride():
    padded = pad_frames(_fake_frames(60))
    windows = list(sliding_windows(padded))
    assert all(w.shape[0] == 100 for w in windows)
    assert len(windows) >= 2


def test_predictions_to_shots_matches_hand_computed_example():
    # cut markers (>0.5) at frames 4 and 9 -> two shots: [0,4] and [5,9], then
    # a final shot [10,14] since the tail never re-enters a cut.
    predictions = np.array(
        [0.1, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1]
    )
    shots = predictions_to_shots(predictions)
    assert shots == [
        Shot(start_frame=0, end_frame=4),
        Shot(start_frame=5, end_frame=9),
        Shot(start_frame=10, end_frame=14),
    ]


def test_predictions_to_shots_returns_full_range_when_no_transition_detected():
    predictions = np.full(20, 0.1)
    shots = predictions_to_shots(predictions)
    assert shots == [Shot(start_frame=0, end_frame=19)]


def test_predictions_to_shots_covers_every_frame_with_no_gaps_for_isolated_cuts():
    """Realistic case: cuts are isolated single frames (as TransNetV2's
    predictions typically look for hard cuts). Contiguity only holds for this
    case — see the two quirk tests below for where it does not."""
    predictions = np.full(200, 0.1)
    cut_positions = [10, 30, 50, 70, 90, 110, 130, 150, 170, 190]  # spaced apart, no two adjacent
    predictions[cut_positions] = 0.9

    shots = predictions_to_shots(predictions)

    assert min(s.start_frame for s in shots) == 0
    assert max(s.end_frame for s in shots) == 199
    for prev_shot, next_shot in zip(shots, shots[1:], strict=False):
        assert next_shot.start_frame == prev_shot.end_frame + 1


def test_predictions_to_shots_excludes_frame_zero_when_it_is_itself_a_cut():
    """Known upstream quirk (soCzech/TransNetV2 `predictions_to_scenes`): its
    `i != 0` guard means a cut classified at frame 0 is not attributed to any
    shot. Ported as-is, not silently patched, since this module reproduces
    the reference algorithm's exact behavior (CLAUDE.md distinguishes 📄
    paper/upstream behavior from 🔧 this project's own choices)."""
    predictions = np.array([0.9, 0.1, 0.1, 0.9, 0.1, 0.1])
    shots = predictions_to_shots(predictions)
    assert shots[0].start_frame == 1


def test_predictions_to_shots_leaves_gap_for_multi_frame_transition_runs():
    """Another upstream quirk: only the *first* frame of a run of consecutive
    above-threshold frames (e.g. a multi-frame fade/dissolve) is attributed to
    the preceding shot's end; the remaining frames of that run belong to no
    shot at all. Documented rather than patched, same rationale as above."""
    predictions = np.array([0.1, 0.1, 0.9, 0.9, 0.9, 0.1, 0.1])
    shots = predictions_to_shots(predictions)
    assert shots == [Shot(start_frame=0, end_frame=2), Shot(start_frame=5, end_frame=6)]
