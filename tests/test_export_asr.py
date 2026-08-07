"""RED reproducer for pipelines/export_asr.py (P0.11a hand-off artifact).

Pure function over plain Elasticsearch `_source` dicts -- no Elasticsearch
needed. Mirrors tests/test_embed_transcripts.py's fake-double convention.
"""

from __future__ import annotations

import pytest

from pipelines.export_asr import build_asr_export, build_shot_aligned_export

_GENERATED_AT = "2026-08-04T00:00:00Z"


def _doc(
    video_id: str = "L21_V001",
    index: int = 0,
    start_time: float = 0.0,
    end_time: float = 1.0,
    text: str = "xin chao cac ban",
    start_frame_idx: int = 0,
    end_frame_idx: int = 25,
) -> dict:
    return {
        "segment_id": f"{video_id}_seg{index}",
        "video_id": video_id,
        "start_time": start_time,
        "end_time": end_time,
        "text": text,
        "start_frame_idx": start_frame_idx,
        "end_frame_idx": end_frame_idx,
    }


def test_export_groups_segments_under_one_entry_per_video():
    docs = [
        _doc(video_id="L21_V002", index=0),
        _doc(video_id="L21_V001", index=0),
        _doc(video_id="L21_V001", index=1, start_time=2.0),
    ]

    export = build_asr_export(docs, model_id="large-v3", generated_at=_GENERATED_AT)

    assert [v["video_id"] for v in export["videos"]] == ["L21_V001", "L21_V002"]
    assert [v["num_segments"] for v in export["videos"]] == [2, 1]


def test_export_names_video_with_mp4_suffix_like_shots_json():
    export = build_asr_export(
        [_doc(video_id="L21_V001")], model_id="large-v3", generated_at=_GENERATED_AT
    )

    assert export["videos"][0]["video_name"] == "L21_V001.mp4"


def test_export_orders_segments_chronologically_not_lexicographically_by_id():
    # "seg10" sorts before "seg2" as a string -- ordering must come from
    # start_time so a transcript reads top-to-bottom in spoken order.
    docs = [
        _doc(index=10, start_time=30.0, end_time=31.0),
        _doc(index=2, start_time=6.0, end_time=7.0),
    ]

    export = build_asr_export(docs, model_id="large-v3", generated_at=_GENERATED_AT)

    segments = export["videos"][0]["segments"]
    assert [s["segment_id"] for s in segments] == ["L21_V001_seg2", "L21_V001_seg10"]


def test_export_keeps_frame_window_fields_distinct_from_shot_boundary_names():
    # CLAUDE.md T2: these are floor/ceil retrieval windows -- NOT shots.json's
    # shot boundaries and NOT a submittable frame_id. Reusing shots.json's
    # `start_frame`/`end_frame` names here would invite a wrong join.
    export = build_asr_export(
        [_doc(start_frame_idx=7, end_frame_idx=99)],
        model_id="large-v3",
        generated_at=_GENERATED_AT,
    )

    segment = export["videos"][0]["segments"][0]
    assert segment["start_frame_idx"] == 7
    assert segment["end_frame_idx"] == 99
    assert "start_frame" not in segment
    assert "end_frame" not in segment


def test_export_strips_whisper_leading_space_from_text():
    export = build_asr_export(
        [_doc(text=" Chao mung quy vi ")], model_id="large-v3", generated_at=_GENERATED_AT
    )

    assert export["videos"][0]["segments"][0]["text"] == "Chao mung quy vi"


def test_export_records_provenance_in_meta():
    docs = [_doc(index=0), _doc(index=1, start_time=2.0), _doc(video_id="L21_V002")]

    export = build_asr_export(docs, model_id="large-v3", generated_at=_GENERATED_AT)

    meta = export["meta"]
    assert meta["model_id"] == "large-v3"
    assert meta["generated_at"] == _GENERATED_AT
    assert meta["num_videos"] == 2
    assert meta["num_segments"] == 3


def test_export_raises_on_empty_input_instead_of_producing_an_empty_file():
    # CLAUDE.md section 7: missing data must fail loudly, never look like a
    # successful run that happened to produce nothing.
    with pytest.raises(ValueError, match="no ASR segments"):
        build_asr_export([], model_id="large-v3", generated_at=_GENERATED_AT)


def _shot(shot_id: int, start_sec: float, end_sec: float, fps: float = 30.0) -> dict:
    return {
        "shot_id": shot_id,
        "start_frame": int(start_sec * fps),
        "end_frame": int(end_sec * fps),
        "start_sec": start_sec,
        "end_sec": end_sec,
    }


def _shots_document(*shots: dict, video_name: str = "L21_V001.mp4") -> dict:
    return {"videos": [{"video_name": video_name, "shots": list(shots)}]}


def test_shot_alignment_repeats_a_sentence_under_every_shot_it_spans():
    # The chosen semantics: a sentence is genuinely audible across every shot
    # it spans, so each shot answers "what is being said right now" on its own.
    asr_export = build_asr_export(
        [_doc(index=2, start_time=10.29, end_time=18.94, text="dong bang song Cuu Long")],
        model_id="large-v3",
        generated_at=_GENERATED_AT,
    )
    shots = _shots_document(
        _shot(12, 11.33, 13.67), _shot(13, 13.70, 15.23), _shot(14, 15.26, 20.0)
    )

    aligned = build_shot_aligned_export(asr_export, shots, generated_at=_GENERATED_AT)

    rows = [shot["transcript"] for shot in aligned["videos"][0]["shots"]]
    assert [len(row) for row in rows] == [1, 1, 1]
    assert {row[0]["segment_id"] for row in rows} == {"L21_V001_seg2"}
    assert aligned["meta"]["num_transcript_rows"] == 3
    assert aligned["meta"]["num_shots_with_speech"] == 3


def test_shot_alignment_writes_null_for_a_silent_shot():
    asr_export = build_asr_export(
        [_doc(index=0, start_time=10.0, end_time=12.0)],
        model_id="large-v3",
        generated_at=_GENERATED_AT,
    )
    shots = _shots_document(_shot(1, 0.0, 5.0), _shot(2, 10.5, 11.0))

    aligned = build_shot_aligned_export(asr_export, shots, generated_at=_GENERATED_AT)

    shot_entries = aligned["videos"][0]["shots"]
    assert shot_entries[0]["transcript"] is None
    assert shot_entries[1]["transcript"] is not None
    assert aligned["meta"]["num_shots"] == 2
    assert aligned["meta"]["num_shots_with_speech"] == 1


def test_shot_alignment_does_not_match_a_shot_that_merely_touches_the_segment_edge():
    # A shot ending exactly when speech starts contains no speech at all.
    asr_export = build_asr_export(
        [_doc(index=0, start_time=5.0, end_time=8.0)],
        model_id="large-v3",
        generated_at=_GENERATED_AT,
    )
    shots = _shots_document(_shot(1, 2.0, 5.0), _shot(2, 8.0, 9.0))

    aligned = build_shot_aligned_export(asr_export, shots, generated_at=_GENERATED_AT)

    assert [shot["transcript"] for shot in aligned["videos"][0]["shots"]] == [None, None]


def test_shot_alignment_reports_seconds_of_speech_inside_each_shot():
    asr_export = build_asr_export(
        [_doc(index=0, start_time=10.0, end_time=20.0)],
        model_id="large-v3",
        generated_at=_GENERATED_AT,
    )
    shots = _shots_document(_shot(1, 8.0, 12.0), _shot(2, 12.0, 25.0))

    aligned = build_shot_aligned_export(asr_export, shots, generated_at=_GENERATED_AT)

    shot_entries = aligned["videos"][0]["shots"]
    assert shot_entries[0]["transcript"][0]["overlap_sec"] == 2.0  # 10.0 -> 12.0
    assert shot_entries[1]["transcript"][0]["overlap_sec"] == 8.0  # 12.0 -> 20.0


def test_shot_alignment_preserves_original_shot_fields_verbatim():
    asr_export = build_asr_export([_doc()], model_id="large-v3", generated_at=_GENERATED_AT)
    shots = _shots_document(_shot(7, 0.0, 1.0))

    aligned = build_shot_aligned_export(asr_export, shots, generated_at=_GENERATED_AT)

    shot_entry = aligned["videos"][0]["shots"][0]
    assert shot_entry["shot_id"] == 7
    assert shot_entry["start_frame"] == 0
    assert shot_entry["end_frame"] == 30
    assert shot_entry["start_sec"] == 0.0
    assert shot_entry["end_sec"] == 1.0


def test_shot_alignment_lists_videos_missing_from_shots_json_instead_of_dropping_them():
    asr_export = build_asr_export(
        [_doc(video_id="L21_V001"), _doc(video_id="L99_V999")],
        model_id="large-v3",
        generated_at=_GENERATED_AT,
    )
    shots = _shots_document(_shot(1, 0.0, 5.0))

    aligned = build_shot_aligned_export(asr_export, shots, generated_at=_GENERATED_AT)

    assert [v["video_id"] for v in aligned["videos"]] == ["L21_V001"]
    assert aligned["meta"]["videos_missing_shots"] == ["L99_V999"]


def test_shot_alignment_raises_when_no_video_matches_at_all():
    asr_export = build_asr_export(
        [_doc(video_id="L99_V999")], model_id="large-v3", generated_at=_GENERATED_AT
    )
    shots = _shots_document(_shot(1, 0.0, 5.0))

    with pytest.raises(ValueError, match="no video could be aligned"):
        build_shot_aligned_export(asr_export, shots, generated_at=_GENERATED_AT)
