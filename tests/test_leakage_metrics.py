"""RED reproducer for eval/metrics/leakage.py (plan §P0.9) -- the
caption<->ocr_text label-leakage test. High vocabulary overlap between a
ground-truth caption and the OCR text of its own answer keyframe means
Gemini transcribed on-screen text into the caption (CLAUDE.md §3: model
sinh nhãn phải khác model bị đánh giá) -- if that happened, the OCR
channel's score is just the system finding text Gemini already copied out,
not real retrieval.
"""

from __future__ import annotations

import pytest

from eval.metrics.leakage import LeakageRecord, caption_ocr_jaccard, compute_leakage_report

_STOPWORDS: frozenset[str] = frozenset({"la", "va", "mot"})


def test_caption_ocr_jaccard_is_1_when_texts_share_every_content_word():
    overlap = caption_ocr_jaccard("chay rung nui", ["chay rung nui"], _STOPWORDS)

    assert overlap == pytest.approx(1.0)


def test_caption_ocr_jaccard_is_0_when_no_words_shared():
    overlap = caption_ocr_jaccard("chay rung nui", ["hop bao chinh phu"], _STOPWORDS)

    assert overlap == pytest.approx(0.0)


def test_caption_ocr_jaccard_takes_the_max_across_multiple_keyframe_ocr_texts():
    # The shot's OCR texts come from several keyframes; only one needs to
    # leak for this to be a real leakage signal, so the max (not mean) is
    # the correct aggregation.
    overlap = caption_ocr_jaccard(
        "chay rung nui", ["hop bao chinh phu", "chay rung nui"], _STOPWORDS
    )

    assert overlap == pytest.approx(1.0)


def test_caption_ocr_jaccard_returns_0_when_ocr_texts_is_empty():
    # No OCR coverage for this shot is expected data (OCR is still partial),
    # not an error -- 0.0 overlap, not a raise.
    overlap = caption_ocr_jaccard("chay rung nui", [], _STOPWORDS)

    assert overlap == pytest.approx(0.0)


def test_caption_ocr_jaccard_returns_0_when_caption_has_no_content_words():
    overlap = caption_ocr_jaccard("la va mot", ["chay rung nui"], _STOPWORDS)

    assert overlap == pytest.approx(0.0)


def test_compute_leakage_report_aggregates_mean_median_and_high_overlap_rate():
    records = [
        LeakageRecord(
            video_id="L21_V001", order_index=1, caption="chay rung nui", ocr_texts=["chay rung nui"]
        ),
        LeakageRecord(
            video_id="L21_V002", order_index=2, caption="chay rung nui", ocr_texts=["hop bao"]
        ),
    ]

    report = compute_leakage_report(records, stopwords=_STOPWORDS, threshold=0.5)

    assert report.total == 2
    assert report.mean_overlap == pytest.approx(0.5)
    assert report.high_overlap_count == 1
    assert report.high_overlap_rate == pytest.approx(0.5)
    assert report.threshold == pytest.approx(0.5)


def test_compute_leakage_report_raises_on_empty_records():
    with pytest.raises(ValueError):
        compute_leakage_report([], stopwords=_STOPWORDS, threshold=0.5)
