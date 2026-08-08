"""Imports a Kaggle-produced asr_out/ artifact into local ES/MongoDB
(.claude/plans/kaggle-asr-transcribe.plan.md, Nhip A T1-T3).

Kaggle only decodes Whisper and exports RAW (unfiltered) segments -- no
hallucination filter, no frame-index window, no ES/Mongo write (plan Q2:
"xuat RAW, loc o local"). This module is the local side: validate the
artifact against the running machine's own config/checkpoint (F3 -- must not
let the Nhip 1 model_id bug back in through the Kaggle door), cross-check
Kaggle's fps/frame_count against a fresh read of the local .mp4 (F1 -- the
two machines' copies of a video are not provably identical), optionally
re-cut a batched-mode segment into sentence-length pieces using word
timestamps (pipelines/asr.py's resegment_by_pause), then hand the resulting
AsrSegment list to the EXISTING run_asr() -- no second filtering code path.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from pipelines.asr import AsrConfig, AsrSegment, Word, resegment_by_pause
from pipelines.verify_embed_parity import load_manifest, sha256_file
from pipelines.video_io import VideoInfo

__all__ = [
    "AsrArtifactError",
    "load_manifest",
    "verify_manifest",
    "discover_video_ids",
    "load_run_index",
    "verify_video_info_matches",
    "load_raw_segment_rows",
    "build_segments_for_video",
]


class AsrArtifactError(Exception):
    """A manifest/contract/cross-check failed -- import must not proceed."""


def verify_manifest(manifest: dict, local_config: AsrConfig, checkpoint_dir: Path) -> None:
    """Raises AsrArtifactError if the artifact was produced by a different
    model_id or checkpoint than the one this machine's asr.yaml points at.

    This is F3's guard: silently importing a mismatched artifact would skip
    every video already `done` under the OLD model on the next `run_asr.py`
    call while stamping the export with the new model_id -- provenance that
    lies about which checkpoint produced it, with no crash (research-asr-
    phowhisper-batched.plan.md F1, reopened via the Kaggle path).
    """
    if manifest["model_id"] != local_config.model_id:
        raise AsrArtifactError(
            f"model_id mismatch: manifest={manifest['model_id']!r} "
            f"local asr.yaml={local_config.model_id!r}"
        )

    actual_checkpoint = sha256_file(checkpoint_dir / "model.bin")
    if actual_checkpoint != manifest["checkpoint_sha256"]:
        raise AsrArtifactError(
            f"checkpoint_sha256 mismatch: manifest={manifest['checkpoint_sha256']} "
            f"local({checkpoint_dir / 'model.bin'})={actual_checkpoint}"
        )

    actual_vocabulary = sha256_file(checkpoint_dir / "vocabulary.json")
    if actual_vocabulary != manifest["vocabulary_sha256"]:
        raise AsrArtifactError(
            f"vocabulary_sha256 mismatch: manifest={manifest['vocabulary_sha256']} "
            f"local({checkpoint_dir / 'vocabulary.json'})={actual_vocabulary}"
        )


def discover_video_ids(segments_dir: Path) -> list[str]:
    """`video_id`s the Kaggle run actually produced a JSONL for, sorted.
    Reads filenames only -- `load_run_index` is the source of truth for
    which of these are actually `status: "ok"`."""
    return sorted(path.stem for path in segments_dir.glob("*.jsonl"))


def load_run_index(runs_path: Path) -> dict[str, dict]:
    """One row per `video_id`, keyed by the LAST line seen for it -- mirrors
    the Kaggle-side resume contract (a video retried after a `failed` row
    appends a second, newer line for the same id; the newer one wins)."""
    if not runs_path.exists():
        raise AsrArtifactError(f"runs.jsonl not found: {runs_path}")
    index: dict[str, dict] = {}
    with runs_path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            index[row["video_id"]] = row
    return index


def verify_video_info_matches(
    video_id: str,
    local: VideoInfo,
    kaggle_fps: float,
    kaggle_frame_count: int,
    *,
    fps_tolerance: float,
    frame_count_tolerance: int,
) -> None:
    """F1: the frame-index window is only trustworthy if both machines agree
    on `fps`/`frame_count` for the SAME video. Raises with both numbers on
    mismatch -- never silently trusts either side (CLAUDE.md sec 3: a
    rounding mismatch here is a silent TRAKE-window miss, not a crash)."""
    if abs(local.fps - kaggle_fps) > fps_tolerance:
        raise AsrArtifactError(
            f"{video_id}: fps mismatch beyond tolerance ({fps_tolerance}) -- "
            f"local={local.fps} kaggle={kaggle_fps}"
        )
    if abs(local.frame_count - kaggle_frame_count) > frame_count_tolerance:
        raise AsrArtifactError(
            f"{video_id}: frame_count mismatch beyond tolerance "
            f"({frame_count_tolerance}) -- local={local.frame_count} kaggle={kaggle_frame_count}"
        )


def load_raw_segment_rows(segments_path: Path) -> list[dict]:
    """Reads one video's RAW JSONL, enforcing the artifact contract (plan
    "Hop dong artifact" bullet 1): `row["index"]` MUST equal its line
    number -- this is what `run_asr()` derives `segment_id` from
    (pipelines/asr.py `f"{video_id}_seg{index}"`), so a mismatch here means
    `segment_id` would silently disagree between runs."""
    rows: list[dict] = []
    with segments_path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            row = json.loads(line)
            if row["index"] != line_no:
                raise AsrArtifactError(
                    f"{segments_path.name}: row index {row['index']} != line number "
                    f"{line_no} -- artifact contract violated, segment_id would be wrong"
                )
            rows.append(row)
    return rows


def _row_to_words(raw_words: list[dict] | None) -> tuple[Word, ...] | None:
    if raw_words is None:
        return None
    return tuple(
        Word(start=w["start"], end=w["end"], word=w["word"], probability=w["probability"])
        for w in raw_words
    )


def build_segments_for_video(
    raw_rows: Sequence[dict], resegment_min_pause_sec: float | None
) -> list[AsrSegment]:
    """Turns one video's RAW rows into the `AsrSegment` list `run_asr()`
    expects, in file order (order is `segment_id`'s stability contract --
    CLAUDE.md sec 3).

    When `resegment_min_pause_sec` is set AND a row carries `words`, the row
    is re-cut into sentence-length pieces via `resegment_by_pause()` instead
    of being kept as one long batched-mode segment (pipelines/asr.py's
    `word_timestamps` docstring). A row without `words` -- e.g. an artifact
    produced before `word_timestamps` existed -- degrades gracefully to the
    original single segment rather than blocking the whole video's import.
    """
    segments: list[AsrSegment] = []
    for row in raw_rows:
        words = _row_to_words(row.get("words"))
        if resegment_min_pause_sec is not None and words:
            segments.extend(
                resegment_by_pause(
                    words,
                    min_pause_sec=resegment_min_pause_sec,
                    no_speech_prob=row["no_speech_prob"],
                    avg_logprob=row["avg_logprob"],
                )
            )
        else:
            segments.append(
                AsrSegment(
                    start=row["start"],
                    end=row["end"],
                    text=row["text"],
                    no_speech_prob=row["no_speech_prob"],
                    avg_logprob=row["avg_logprob"],
                )
            )
    return segments
