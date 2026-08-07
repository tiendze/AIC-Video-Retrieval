"""RED reproducer for pipelines/compare_asr_segment_counts.py
(research-asr-phowhisper-batched.plan.md T6b) — a free check against
faster-whisper#1179 (batched inference silently dropping segments),
using only the num_segments field two asr_transcripts.json-shaped
exports already carry (pipelines/export_asr.py's build_asr_export
output). Pure function, no GPU, no manual labels needed.
"""

from __future__ import annotations

from pipelines.compare_asr_segment_counts import compare_segment_counts


def _export(*video_counts: tuple[str, int]) -> dict:
    return {
        "meta": {"model_id": "unused"},
        "videos": [
            {"video_id": video_id, "num_segments": num_segments}
            for video_id, num_segments in video_counts
        ],
    }


def test_reports_before_and_after_count_for_each_video():
    before = _export(("L21_V001", 40), ("L21_V002", 30))
    after = _export(("L21_V001", 40), ("L21_V002", 30))

    diffs = compare_segment_counts(before, after)

    assert [d.video_id for d in diffs] == ["L21_V001", "L21_V002"]
    assert diffs[0].before == 40
    assert diffs[0].after == 40


def test_flags_dropped_when_after_count_is_lower():
    # faster-whisper#1179's signature: batched silently drops segments.
    before = _export(("L21_V001", 40))
    after = _export(("L21_V001", 25))

    diffs = compare_segment_counts(before, after)

    assert diffs[0].delta == -15
    assert diffs[0].dropped is True


def test_does_not_flag_when_after_count_is_equal_or_higher():
    before = _export(("L21_V001", 40))
    after = _export(("L21_V001", 40))

    assert compare_segment_counts(before, after)[0].dropped is False

    after_more = _export(("L21_V001", 45))
    assert compare_segment_counts(before, after_more)[0].dropped is False


def test_video_missing_from_after_export_counts_as_a_full_drop_to_zero():
    # A video vanishing entirely between runs is itself a red flag -- must
    # not be silently skipped just because it's absent from one side.
    before = _export(("L21_V001", 40))
    after = _export()

    diffs = compare_segment_counts(before, after)

    assert diffs[0].video_id == "L21_V001"
    assert diffs[0].before == 40
    assert diffs[0].after == 0
    assert diffs[0].dropped is True


def test_video_only_in_after_export_is_not_flagged_as_dropped():
    # A video transcribed for the first time (0 -> N) is growth, not a drop.
    before = _export()
    after = _export(("L21_V999", 10))

    diffs = compare_segment_counts(before, after)

    assert diffs[0].before == 0
    assert diffs[0].after == 10
    assert diffs[0].dropped is False


def test_results_are_sorted_by_video_id_regardless_of_input_order():
    before = _export(("L21_V002", 1), ("L21_V001", 1))
    after = _export(("L21_V001", 1), ("L21_V002", 1))

    diffs = compare_segment_counts(before, after)

    assert [d.video_id for d in diffs] == ["L21_V001", "L21_V002"]
