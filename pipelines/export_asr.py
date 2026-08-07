"""Hand-off artifact for the ASR branch (P0.11a) — turns the `asr_segments`
Elasticsearch index back into one shareable JSON file:

    Elasticsearch (asr_segments) -> scroll -> group by video_id -> .json

Elasticsearch stays the source of truth; this file is a *derived* copy for
teammates who don't run the stack locally. It is rebuilt from the index, never
by re-running Whisper — transcription is the expensive, irreversible step
(architecture.md §3.2 note 7).

Container shape mirrors `data/shots.json` (`{"videos": [...]}`, one entry per
video keyed by `video_name`) so both files parse the same way. Frame fields
deliberately do NOT: shots.json's `start_frame`/`end_frame` are real shot
boundaries, while `start_frame_idx`/`end_frame_idx` here are floor/ceil
*retrieval windows* derived from segment timings, and never a submittable
`frame_id` (CLAUDE.md T2). Different meaning ⇒ different name, so the two files
cannot be joined on frame columns by accident.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

_VIDEO_FILE_SUFFIX = ".mp4"
_OVERLAP_DECIMALS = 3

_FRAME_WINDOW_NOTE = (
    "start_frame_idx/end_frame_idx are floor/ceil retrieval windows derived from "
    "start_sec/end_sec, NOT shot boundaries and NOT a submittable frame_id. "
    "A submission frame_id always comes from a keyframe's own frame_idx."
)

_ALIGNMENT_NOTE = (
    "transcript is null when nobody speaks during the shot. A sentence spanning "
    "several shots is repeated under each one, so segment_id is NOT unique in this "
    "file and num_transcript_rows exceeds the segment count; overlap_sec says how "
    "many seconds of that sentence fall inside the shot. Shot boundaries are copied "
    "verbatim from shots.json."
)


def _to_export_segment(document: Mapping[str, Any]) -> dict[str, Any]:
    """Renames the ES field names to the export's own vocabulary. `start_sec`/
    `end_sec` match shots.json; the frame fields keep their `_idx` suffix on
    purpose (see module docstring). Whisper prefixes most segments with a
    leading space — stripped here so the JSON reads cleanly."""
    return {
        "segment_id": document["segment_id"],
        "start_sec": document["start_time"],
        "end_sec": document["end_time"],
        "start_frame_idx": document["start_frame_idx"],
        "end_frame_idx": document["end_frame_idx"],
        "text": document["text"].strip(),
    }


def _segment_sort_key(segment: Mapping[str, Any]) -> tuple[float, float, str]:
    """Chronological, not lexicographic: `L21_V001_seg10` sorts before
    `..._seg2` as a string, which would scramble a transcript."""
    return (segment["start_sec"], segment["end_sec"], segment["segment_id"])


def build_asr_export(
    documents: Iterable[Mapping[str, Any]], *, model_id: str, generated_at: str
) -> dict[str, Any]:
    """Groups flat ES segment documents into one JSON-serializable document.

    Args:
        documents: `_source` dicts from the `asr_segments` index.
        model_id: Whisper checkpoint that produced them, recorded for
            provenance (CLAUDE.md §8 — every reported artifact names its model).
        generated_at: ISO-8601 UTC timestamp of this export.

    Raises:
        ValueError: when `documents` is empty. An empty index means ASR has not
            run yet, which must fail loudly rather than produce a valid-looking
            file with zero videos in it (CLAUDE.md §7).
    """
    segments_by_video: dict[str, list[dict[str, Any]]] = {}
    for document in documents:
        video_id = document["video_id"]
        segments_by_video.setdefault(video_id, []).append(_to_export_segment(document))

    if not segments_by_video:
        raise ValueError(
            "no ASR segments to export — run scripts/run_asr.py first, or check "
            "that the Elasticsearch index name matches"
        )

    videos = [
        {
            "video_name": f"{video_id}{_VIDEO_FILE_SUFFIX}",
            "video_id": video_id,
            "num_segments": len(segments),
            "segments": sorted(segments, key=_segment_sort_key),
        }
        for video_id, segments in sorted(segments_by_video.items())
    ]

    return {
        "meta": {
            "model_id": model_id,
            "generated_at": generated_at,
            "num_videos": len(videos),
            "num_segments": sum(video["num_segments"] for video in videos),
            "frame_window_note": _FRAME_WINDOW_NOTE,
        },
        "videos": videos,
    }


def _overlap_sec(shot: Mapping[str, Any], segment: Mapping[str, Any]) -> float:
    """Seconds of speech audible inside this shot. Overlap is measured in
    SECONDS, the native unit of both files — going through frame indices would
    re-introduce an fps rounding step the project has no reason to take."""
    overlap = min(shot["end_sec"], segment["end_sec"]) - max(
        shot["start_sec"], segment["start_sec"]
    )
    return round(max(0.0, overlap), _OVERLAP_DECIMALS)


def _transcript_for_shots(
    shots: Sequence[Mapping[str, Any]], segments: Sequence[Mapping[str, Any]]
) -> list[list[dict[str, Any]] | None]:
    """One entry per shot, in shot order: every segment overlapping that shot,
    or `None` when nobody speaks during it.

    A sentence spanning several shots is repeated under each of them — the
    file answers "what is being said while these frames are on screen", and a
    sentence is genuinely audible across every shot it spans. Consequence: the
    total transcript row count exceeds the segment count, and `segment_id`
    is NOT unique in this file (it stays unique in `asr_transcripts.json`).

    Both sequences must be sorted by `start_sec`; the cursor only moves
    forward, which assumes Whisper's segments do not overlap each other.
    """
    transcripts: list[list[dict[str, Any]] | None] = []
    cursor = 0
    for shot in shots:
        while cursor < len(segments) and segments[cursor]["end_sec"] <= shot["start_sec"]:
            cursor += 1

        matched: list[dict[str, Any]] = []
        index = cursor
        while index < len(segments) and segments[index]["start_sec"] < shot["end_sec"]:
            segment = segments[index]
            if segment["end_sec"] > shot["start_sec"]:
                matched.append(
                    {
                        "segment_id": segment["segment_id"],
                        "start_sec": segment["start_sec"],
                        "end_sec": segment["end_sec"],
                        "overlap_sec": _overlap_sec(shot, segment),
                        "text": segment["text"],
                    }
                )
            index += 1
        transcripts.append(matched or None)
    return transcripts


def build_shot_aligned_export(
    asr_export: Mapping[str, Any], shots_document: Mapping[str, Any], *, generated_at: str
) -> dict[str, Any]:
    """Overlays `build_asr_export`'s output onto `data/shots.json`'s shot
    boundaries, so each shot carries the speech heard while it is on screen.

    Every field of the original shot is preserved verbatim (`shot_id`,
    `start_frame`, `end_frame`, `start_sec`, `end_sec`) with one field added:
    `transcript`, either a list of overlapping segments or `None`.

    Videos present in the ASR export but absent from `shots_document` are
    skipped and listed in `meta.videos_missing_shots` rather than silently
    dropped — they stay available in `asr_transcripts.json`.

    Raises:
        ValueError: when no video can be aligned at all, which means the two
            files describe disjoint video sets (CLAUDE.md §7 — fail loudly
            instead of writing a file with zero videos).
    """
    shots_by_video: dict[str, list[dict[str, Any]]] = {
        entry["video_name"].removesuffix(_VIDEO_FILE_SUFFIX): sorted(
            entry["shots"], key=lambda shot: shot["start_sec"]
        )
        for entry in shots_document["videos"]
    }

    videos: list[dict[str, Any]] = []
    missing_shots: list[str] = []
    for video in asr_export["videos"]:
        shots = shots_by_video.get(video["video_id"])
        if shots is None:
            missing_shots.append(video["video_id"])
            continue

        transcripts = _transcript_for_shots(shots, video["segments"])
        videos.append(
            {
                "video_name": video["video_name"],
                "video_id": video["video_id"],
                "num_shots": len(shots),
                "shots": [
                    {**shot, "transcript": transcript}
                    for shot, transcript in zip(shots, transcripts, strict=True)
                ],
            }
        )

    if not videos:
        raise ValueError(
            "no video could be aligned — the ASR export and shots.json share no "
            f"video_id (missing from shots.json: {missing_shots})"
        )

    all_shots = [shot for video in videos for shot in video["shots"]]
    return {
        "meta": {
            "model_id": asr_export["meta"]["model_id"],
            "generated_at": generated_at,
            "num_videos": len(videos),
            "num_shots": len(all_shots),
            "num_shots_with_speech": sum(1 for shot in all_shots if shot["transcript"]),
            "num_transcript_rows": sum(len(shot["transcript"] or ()) for shot in all_shots),
            "videos_missing_shots": missing_shots,
            "alignment_note": _ALIGNMENT_NOTE,
        },
        "videos": videos,
    }
