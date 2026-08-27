"""B6 -- dense local frame refinement for TRAKE.

The gap this closes. Retrieval runs on keyframes, which are sparse (one per
shot boundary), but TRAKE is graded on a window of under 10 ABSOLUTE frames --
about 0.3s at 30fps (CLAUDE.md §1.2). So a sequence can name the right video
and the right shot and still score zero because the nearest keyframe sits 1.5s
away from the moment being asked about. Nothing else in the pipeline can fix
that: chaining, reranking and video ranking all operate on the keyframe grid,
so they all inherit its resolution.

What this does. For each chosen keyframe, decode a small window around it at
the video's native frame rate, encode those frames with the SAME encoder used
at ingest (CLAUDE.md §3 -- a second encoder or a second preprocessing path is a
silent failure, not a crash), and re-pick the argmax against that event's query
vector. Strict temporal ordering is re-imposed afterwards, so refinement can
never produce an out-of-order sequence.

`frame_idx` comes from the decoder's own counter, never from `timestamp * fps`
(`pipelines/video_io.py` module docstring) -- this step is the easiest place in
the codebase to reintroduce that bug, because it is the only request-time code
that touches raw video.

Disabled by default (`configs/search/trake.yaml: frame_refine.enabled`): it
needs the source .mp4 files on disk, and BTC folder names are inconsistent so
the path has to be globbed rather than joined (references.md §2). Every failure
here is logged and skipped -- a missing video file downgrades the sequence to
its keyframe-level answer, it never fails the request (CLAUDE.md §7).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from app.encoders.base import MultimodalEncoder
from app.services.trake_chain import CandidateFrame, ChainCandidate

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FrameRefineConfig:
    enabled: bool
    video_root: Path
    window_seconds: float
    stride_frames: int
    max_frames_per_event: int
    max_sequences: int
    batch_size: int

    @classmethod
    def from_mapping(cls, mapping: dict, repo_root: Path) -> FrameRefineConfig:
        return cls(
            enabled=bool(mapping["enabled"]),
            video_root=repo_root / mapping["video_root"],
            window_seconds=float(mapping["window_seconds"]),
            stride_frames=int(mapping["stride_frames"]),
            max_frames_per_event=int(mapping["max_frames_per_event"]),
            max_sequences=int(mapping["max_sequences"]),
            batch_size=int(mapping["batch_size"]),
        )


@dataclass(frozen=True)
class DecodedFrame:
    frame_idx: int
    timestamp: float
    image: Image.Image


class VideoPathResolver:
    """Finds a video's file by globbing, because the dataset's directory names
    are not consistent (references.md §2) -- `video_root / f"{video_id}.mp4"`
    resolves for some batches and not others."""

    def __init__(self, video_root: Path) -> None:
        self._video_root = video_root
        self._cache: dict[str, Path | None] = {}

    def resolve(self, video_id: str) -> Path | None:
        if video_id in self._cache:
            return self._cache[video_id]
        found: Path | None = None
        if self._video_root.is_dir():
            for match in self._video_root.glob(f"**/{video_id}.mp4"):
                found = match
                break
        self._cache[video_id] = found
        return found


class WindowDecoder:
    """Reads a frame window around a target frame index.

    Seeking is unavoidable here -- decoding a 40-minute video from frame 0 to
    reach frame 50,000 would blow the request budget. `CAP_PROP_POS_FRAMES` can
    land off-target on variable-frame-rate video, so the position is read BACK
    after the seek and used as the counter's base instead of assuming the seek
    was exact. That keeps `frame_idx` sourced from the decoder even on the seek
    path.
    """

    def read_window(
        self, video_path: Path, center_frame_idx: int, half_window_frames: int, stride: int
    ) -> list[DecodedFrame]:
        import cv2  # local import: OpenCV is heavy and only this path needs it

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise OSError(f"Could not open video file: {video_path}")
        try:
            fps = capture.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                raise ValueError(f"Video reports non-positive fps ({fps}): {video_path}")
            start = max(center_frame_idx - half_window_frames, 0)
            capture.set(cv2.CAP_PROP_POS_FRAMES, start)
            actual_start = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
            stop = center_frame_idx + half_window_frames
            frames: list[DecodedFrame] = []
            frame_idx = actual_start
            step = max(stride, 1)
            while frame_idx <= stop:
                ok, frame_bgr = capture.read()
                if not ok:
                    break
                if (frame_idx - actual_start) % step == 0:
                    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    frames.append(
                        DecodedFrame(
                            frame_idx=frame_idx,
                            timestamp=frame_idx / fps,
                            image=Image.fromarray(rgb),
                        )
                    )
                frame_idx += 1
            return frames
        finally:
            capture.release()


class TrakeFrameRefiner:
    """Turns "right shot" into "right frame" for the top sequences only.

    Cost is bounded by `max_sequences * n_events * max_frames_per_event`
    images. With the shipped defaults (20 sequences, 5 events, 60 frames) that
    is 6,000 images -- a few seconds of fp16 inference on an 8GB card, run
    sequentially (CLAUDE.md §1.4: local GPU work is safer serial than
    parallel).
    """

    def __init__(
        self,
        encoder: MultimodalEncoder,
        config: FrameRefineConfig,
        path_resolver: VideoPathResolver | None = None,
        decoder: WindowDecoder | None = None,
    ) -> None:
        self._encoder = encoder
        self._config = config
        self._path_resolver = path_resolver or VideoPathResolver(config.video_root)
        self._decoder = decoder or WindowDecoder()

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    def refine(
        self,
        candidates: Sequence[ChainCandidate],
        event_vectors: np.ndarray,
        fps_by_video: dict[str, float],
    ) -> list[ChainCandidate]:
        """Returns the same sequences with sharper `frame_idx` values.

        `event_vectors[i]` is event `i`'s L2-normalized query vector. Events
        without a text query (OCR-only or ASR-only rows) must be passed as a
        zero row: their argmax would be meaningless, so they are left on their
        original keyframe.
        """
        if not self._config.enabled:
            return list(candidates)
        refined: list[ChainCandidate] = []
        for index, candidate in enumerate(candidates):
            if index >= self._config.max_sequences:
                refined.append(candidate)
                continue
            refined.append(self._refine_one(candidate, event_vectors, fps_by_video))
        return refined

    def _refine_one(
        self,
        candidate: ChainCandidate,
        event_vectors: np.ndarray,
        fps_by_video: dict[str, float],
    ) -> ChainCandidate:
        video_path = self._path_resolver.resolve(candidate.video_id)
        if video_path is None:
            logger.info(
                "trake refine: no video file for video_id=%s under %s -- keeping "
                "keyframe-level answer",
                candidate.video_id,
                self._config.video_root,
            )
            return candidate

        fps = fps_by_video.get(candidate.video_id)
        if fps is None or fps <= 0:
            logger.warning(
                "trake refine: video_id=%s has no usable fps -- keeping keyframe-level answer",
                candidate.video_id,
            )
            return candidate

        half_window = int(round(self._config.window_seconds * fps))
        new_frames: list[CandidateFrame] = []
        previous_frame_idx = -1
        for event_index, frame in enumerate(candidate.frames):
            vector = event_vectors[event_index]
            if not np.any(vector):
                new_frames.append(frame)
                previous_frame_idx = frame.frame_idx
                continue
            picked = self._best_frame_in_window(
                video_path=video_path,
                frame=frame,
                vector=vector,
                half_window=half_window,
                lower_bound=previous_frame_idx,
            )
            new_frames.append(picked)
            previous_frame_idx = picked.frame_idx

        return ChainCandidate(
            video_id=candidate.video_id,
            frames=tuple(new_frames),
            min_score=candidate.min_score,
            mean_score=candidate.mean_score,
            coverage=candidate.coverage,
            score=candidate.score,
        )

    def _best_frame_in_window(
        self,
        *,
        video_path: Path,
        frame: CandidateFrame,
        vector: np.ndarray,
        half_window: int,
        lower_bound: int,
    ) -> CandidateFrame:
        try:
            window = self._decoder.read_window(
                video_path, frame.frame_idx, half_window, self._config.stride_frames
            )
        except (OSError, ValueError) as exc:
            logger.warning(
                "trake refine: could not read window around frame_idx=%d of video_id=%s: %s",
                frame.frame_idx,
                frame.video_id,
                exc,
            )
            return frame

        # Strict ordering survives refinement: a candidate that would land at or
        # before the previous event's refined frame is not eligible.
        eligible = [decoded for decoded in window if decoded.frame_idx > lower_bound]
        eligible = eligible[: self._config.max_frames_per_event]
        if not eligible:
            return frame

        similarities = self._score_images(eligible, vector)
        best = int(np.argmax(similarities))
        chosen = eligible[best]
        return CandidateFrame(
            corpus_index=frame.corpus_index,
            keyframe_id=frame.keyframe_id,
            video_id=frame.video_id,
            # From the decoder's own counter -- never timestamp * fps.
            frame_idx=chosen.frame_idx,
            timestamp=chosen.timestamp,
            score=frame.score,
            semantic_z=frame.semantic_z,
            kis_evidence=frame.kis_evidence,
            kis_rank=frame.kis_rank,
        )

    def _score_images(self, frames: Sequence[DecodedFrame], vector: np.ndarray) -> np.ndarray:
        similarities: list[np.ndarray] = []
        batch_size = max(self._config.batch_size, 1)
        for start in range(0, len(frames), batch_size):
            batch = [decoded.image for decoded in frames[start : start + batch_size]]
            # Encoder contract: (N, dim) float32, already L2-normalized, so the
            # dot product IS the cosine (app/encoders/base.py).
            embeddings = self._encoder.encode_image(batch)
            similarities.append(embeddings @ vector)
        return np.concatenate(similarities)
