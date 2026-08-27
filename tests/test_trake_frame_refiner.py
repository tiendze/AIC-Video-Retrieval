"""TrakeFrameRefiner (B6) tests.

The refiner is the only request-time code that touches raw video, which makes
it the easiest place to reintroduce the `timestamp * fps` bug CLAUDE.md §3
forbids -- so the tests below pin where `frame_idx` comes from, that ordering
survives refinement, and that every failure degrades to the keyframe-level
answer instead of failing the request.

`WindowDecoder` is faked, so none of this needs OpenCV or a real .mp4.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.services.trake_chain import CandidateFrame, ChainCandidate
from app.services.trake_frame_refiner import (
    DecodedFrame,
    FrameRefineConfig,
    TrakeFrameRefiner,
    VideoPathResolver,
)

_DIM = 4
_FPS = 25.0


def _unit(hot: int) -> np.ndarray:
    vector = np.zeros(_DIM, dtype=np.float32)
    vector[hot] = 1.0
    return vector


class _FakeEncoder:
    """Encodes an image by reading a marker pixel: an image tagged `hot` maps
    onto basis vector `hot`, so the query vector `_unit(h)` scores 1.0 on
    exactly the frames tagged `h`."""

    embedding_dim = _DIM

    def __init__(self) -> None:
        self.batches: list[int] = []

    @property
    def model_id(self) -> str:
        return "fake-encoder"

    def encode_text(self, texts):  # pragma: no cover - not used here
        raise AssertionError("encode_text must not be called by the refiner")

    def encode_image(self, images):
        self.batches.append(len(images))
        return np.stack([_unit(int(np.asarray(image)[0, 0, 0])) for image in images])


def _image(hot: int) -> Image.Image:
    pixels = np.zeros((2, 2, 3), dtype=np.uint8)
    pixels[0, 0, 0] = hot
    return Image.fromarray(pixels)


class _FakeDecoder:
    """Returns a scripted window per requested center frame."""

    def __init__(self, windows: dict[int, list[DecodedFrame]], error: Exception | None = None):
        self._windows = windows
        self._error = error
        self.calls: list[tuple[int, int, int]] = []

    def read_window(self, video_path, center_frame_idx, half_window_frames, stride):
        self.calls.append((center_frame_idx, half_window_frames, stride))
        if self._error is not None:
            raise self._error
        return self._windows.get(center_frame_idx, [])


class _FakeResolver:
    def __init__(self, path: Path | None) -> None:
        self._path = path

    def resolve(self, video_id: str) -> Path | None:
        return self._path


def _decoded(frame_idx: int, hot: int) -> DecodedFrame:
    return DecodedFrame(frame_idx=frame_idx, timestamp=frame_idx / _FPS, image=_image(hot))


def _config(**overrides) -> FrameRefineConfig:
    values = dict(
        enabled=True,
        video_root=Path("/videos"),
        window_seconds=2.0,
        stride_frames=1,
        max_frames_per_event=60,
        max_sequences=20,
        batch_size=16,
    )
    values.update(overrides)
    return FrameRefineConfig(**values)


def _frame(frame_idx: int) -> CandidateFrame:
    return CandidateFrame(
        corpus_index=frame_idx,
        keyframe_id=f"L21_V001_{frame_idx}",
        video_id="L21_V001",
        frame_idx=frame_idx,
        timestamp=frame_idx / _FPS,
        score=1.0,
        semantic_z=1.0,
        kis_rank=2,
    )


def _candidate(*frame_indices: int) -> ChainCandidate:
    return ChainCandidate(
        video_id="L21_V001",
        frames=tuple(_frame(index) for index in frame_indices),
        min_score=1.0,
        mean_score=1.0,
        coverage=1.0,
        score=1.0,
    )


def _refiner(decoder, config=None, path=Path("/videos/L21_V001.mp4"), encoder=None):
    return TrakeFrameRefiner(
        encoder=encoder or _FakeEncoder(),
        config=config or _config(),
        path_resolver=_FakeResolver(path),
        decoder=decoder,
    )


class TestRefinement:
    def test_moves_the_answer_off_the_keyframe_onto_the_true_frame(self) -> None:
        # The keyframe is frame 500 but the moment the query describes is at
        # 512 -- 12 frames away, outside the graded window even though the shot
        # is right. This is the whole reason B6 exists.
        windows = {
            500: [_decoded(index, hot=0 if index == 512 else 3) for index in range(480, 521)],
            900: [_decoded(index, hot=1 if index == 903 else 3) for index in range(880, 921)],
        }
        refined = _refiner(_FakeDecoder(windows)).refine(
            [_candidate(500, 900)],
            np.stack([_unit(0), _unit(1)]),
            {"L21_V001": _FPS},
        )

        assert refined[0].frame_indices == (512, 903)

    def test_frame_idx_comes_from_the_decoder_not_from_timestamp_times_fps(self) -> None:
        # The decoded frame carries frame_idx 512 with a timestamp that would
        # round to 511 at this fps. The submitted value must be the decoder's.
        drifting = DecodedFrame(frame_idx=512, timestamp=511.4 / _FPS, image=_image(0))
        windows = {500: [drifting]}
        refined = _refiner(_FakeDecoder(windows)).refine(
            [_candidate(500)], np.stack([_unit(0)]), {"L21_V001": _FPS}
        )

        assert refined[0].frames[0].frame_idx == 512

    def test_ordering_survives_refinement(self) -> None:
        # Event 2's best-scoring frame in its window sits BEFORE event 1's
        # refined answer. Taking it would produce a time-reversed sequence.
        windows = {
            500: [_decoded(index, hot=0 if index == 519 else 3) for index in range(480, 521)],
            520: [_decoded(index, hot=1 if index == 505 else 3) for index in range(500, 541)],
        }
        refined = _refiner(_FakeDecoder(windows)).refine(
            [_candidate(500, 520)],
            np.stack([_unit(0), _unit(1)]),
            {"L21_V001": _FPS},
        )

        indices = refined[0].frame_indices
        assert indices[0] == 519
        assert indices[1] > indices[0]

    def test_an_event_without_a_text_query_keeps_its_keyframe(self) -> None:
        # A zero query vector has no argmax worth trusting, so that event is
        # left where chaining put it.
        windows = {500: [_decoded(index, hot=0) for index in range(480, 521)]}
        decoder = _FakeDecoder(windows)
        refined = _refiner(decoder).refine(
            [_candidate(500)], np.zeros((1, _DIM), dtype=np.float32), {"L21_V001": _FPS}
        )

        assert refined[0].frame_indices == (500,)
        assert decoder.calls == []

    def test_window_is_sized_from_the_video_fps(self) -> None:
        decoder = _FakeDecoder({})
        _refiner(decoder, config=_config(window_seconds=2.0, stride_frames=3)).refine(
            [_candidate(500)], np.stack([_unit(0)]), {"L21_V001": _FPS}
        )

        assert decoder.calls == [(500, 50, 3)]

    def test_only_the_top_sequences_are_refined(self) -> None:
        windows = {500: [_decoded(512, hot=0)]}
        decoder = _FakeDecoder(windows)
        candidates = [_candidate(500) for _ in range(5)]
        refined = _refiner(decoder, config=_config(max_sequences=2)).refine(
            candidates, np.stack([_unit(0)]), {"L21_V001": _FPS}
        )

        assert len(decoder.calls) == 2
        assert [item.frame_indices for item in refined[:2]] == [(512,), (512,)]
        assert [item.frame_indices for item in refined[2:]] == [(500,), (500,), (500,)]

    def test_sequence_scores_are_left_untouched(self) -> None:
        # Refinement sharpens WHICH frame is submitted; it is not a reranking
        # step and must not silently reorder the submission.
        windows = {500: [_decoded(512, hot=0)]}
        original = _candidate(500)
        refined = _refiner(_FakeDecoder(windows)).refine(
            [original], np.stack([_unit(0)]), {"L21_V001": _FPS}
        )

        assert refined[0].score == original.score
        assert refined[0].min_score == original.min_score


class TestDegradedPaths:
    def test_disabled_refiner_is_a_pass_through(self) -> None:
        decoder = _FakeDecoder({500: [_decoded(512, hot=0)]})
        refiner = _refiner(decoder, config=_config(enabled=False))

        assert refiner.enabled is False
        refined = refiner.refine([_candidate(500)], np.stack([_unit(0)]), {"L21_V001": _FPS})
        assert refined[0].frame_indices == (500,)
        assert decoder.calls == []

    def test_a_missing_video_file_keeps_the_keyframe_level_answer(self) -> None:
        decoder = _FakeDecoder({})
        refined = _refiner(decoder, path=None).refine(
            [_candidate(500)], np.stack([_unit(0)]), {"L21_V001": _FPS}
        )

        assert refined[0].frame_indices == (500,)
        assert decoder.calls == []

    def test_a_missing_fps_keeps_the_keyframe_level_answer(self) -> None:
        decoder = _FakeDecoder({500: [_decoded(512, hot=0)]})
        refined = _refiner(decoder).refine([_candidate(500)], np.stack([_unit(0)]), {})

        assert refined[0].frame_indices == (500,)
        assert decoder.calls == []

    def test_an_unreadable_video_does_not_fail_the_request(self) -> None:
        decoder = _FakeDecoder({}, error=OSError("Could not open video file"))
        refined = _refiner(decoder).refine(
            [_candidate(500)], np.stack([_unit(0)]), {"L21_V001": _FPS}
        )

        assert refined[0].frame_indices == (500,)

    def test_an_empty_window_keeps_the_keyframe(self) -> None:
        refined = _refiner(_FakeDecoder({500: []})).refine(
            [_candidate(500)], np.stack([_unit(0)]), {"L21_V001": _FPS}
        )

        assert refined[0].frame_indices == (500,)


class TestVideoPathResolver:
    def test_globs_instead_of_joining_because_folder_names_vary(self, tmp_path) -> None:
        nested = tmp_path / "Videos_L21_a" / "video"
        nested.mkdir(parents=True)
        (nested / "L21_V001.mp4").write_bytes(b"")

        resolver = VideoPathResolver(tmp_path)
        assert resolver.resolve("L21_V001") == nested / "L21_V001.mp4"

    def test_returns_none_for_a_video_that_is_not_on_disk(self, tmp_path) -> None:
        assert VideoPathResolver(tmp_path).resolve("L99_V999") is None

    def test_caches_a_miss_so_a_full_glob_runs_once_per_video(self, tmp_path) -> None:
        resolver = VideoPathResolver(tmp_path)
        assert resolver.resolve("L99_V999") is None
        (tmp_path / "L99_V999.mp4").write_bytes(b"")
        assert resolver.resolve("L99_V999") is None


class TestConfig:
    def test_video_root_is_resolved_against_the_repo_root(self) -> None:
        config = FrameRefineConfig.from_mapping(
            {
                "enabled": True,
                "video_root": "data/video",
                "window_seconds": 2.0,
                "stride_frames": 3,
                "max_frames_per_event": 60,
                "max_sequences": 20,
                "batch_size": 16,
            },
            Path("/repo"),
        )

        assert config.video_root == Path("/repo/data/video")
        assert config.enabled is True

    def test_batching_respects_the_configured_batch_size(self) -> None:
        encoder = _FakeEncoder()
        windows = {
            500: [_decoded(index, hot=0 if index == 512 else 3) for index in range(500, 521)]
        }
        _refiner(_FakeDecoder(windows), config=_config(batch_size=8), encoder=encoder).refine(
            [_candidate(500)], np.stack([_unit(0)]), {"L21_V001": _FPS}
        )

        assert encoder.batches == [8, 8, 5]


@pytest.mark.parametrize("stride", [1, 3, 5])
def test_stride_is_passed_through_untouched(stride: int) -> None:
    decoder = _FakeDecoder({})
    _refiner(decoder, config=_config(stride_frames=stride)).refine(
        [_candidate(500)], np.stack([_unit(0)]), {"L21_V001": _FPS}
    )

    assert decoder.calls[0][2] == stride
