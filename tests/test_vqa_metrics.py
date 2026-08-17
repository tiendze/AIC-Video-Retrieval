"""RED reproducer for eval/metrics/vqa.py (plan §P3.4, PRD §6.7) -- pure
scoring functions over VQA query results. Mirrors tests/test_trake_metrics.py
and tests/test_retrieval_metrics.py's structure: dataclasses reduced to only
what scoring needs, pure aggregate functions, ValueError on empty input
instead of a silent 0.0 (CLAUDE.md §7).

Two distinct "grounded" concepts are exercised on purpose (see
app/services/vqa_service.py's GroundedAnswer docstring for the gate-level
one, which this module does not re-check):

- `retrieval_hit`: does ANY keyframe shown to the model (the context/grid)
  fall inside the ground-truth segment -- "did retrieval even give the model
  a chance".
- `evidence_grounded`: does ANY CITED evidence keyframe fall inside the
  ground-truth segment -- "did the model point at the right one".

`grounded_accuracy` requires both a text-correct answer AND
`evidence_grounded`; `hallucination_rate` is defined by the plan table as
confident despite the context never containing a correct keyframe at all
(confident AND NOT retrieval_hit) -- deliberately independent of whether the
answer text happens to be right, since a confident answer with no correct
keyframe in context could not have been produced from real evidence.
"""

from __future__ import annotations

import pytest

from eval.metrics.vqa import (
    VqaContextKeyframe,
    VqaGroundTruthQuery,
    VqaPrediction,
    VqaQueryResult,
    abstention_rate,
    exact_match,
    exact_match_rate,
    grounded_accuracy,
    hallucination_rate,
    mean_llm_judge_score,
    mean_token_f1,
    normalize_answer,
    retrieval_conditioned_accuracy,
    score_query,
    token_f1,
)

CONFIDENCE_THRESHOLD = 0.5


def _gt(video_id: str = "L21_V001", start: float = 10.0, end: float = 12.0, ref: str = "Đen") -> VqaGroundTruthQuery:
    return VqaGroundTruthQuery(video_id=video_id, start_sec=start, end_sec=end, reference_answer=ref)


def _kf(video_id: str, timestamp: float) -> VqaContextKeyframe:
    return VqaContextKeyframe(video_id=video_id, timestamp=timestamp)


def _prediction(
    status: str = "answered",
    answer: str | None = "Đen",
    confidence: float | None = 0.9,
    evidence: tuple[VqaContextKeyframe, ...] = (),
    context: tuple[VqaContextKeyframe, ...] = (),
) -> VqaPrediction:
    return VqaPrediction(
        status=status, answer=answer, confidence=confidence,
        evidence_keyframes=evidence, context_keyframes=context,
    )


# --------------------------------------------------------------------------
# normalize_answer / exact_match
# --------------------------------------------------------------------------


def test_normalize_answer_lowercases_and_strips_punctuation():
    assert normalize_answer("Đen kịt!") == normalize_answer("đen kịt")


def test_normalize_answer_removes_english_articles():
    assert normalize_answer("the red cup") == normalize_answer("red cup")


def test_normalize_answer_preserves_vietnamese_diacritics():
    assert normalize_answer("đen") != normalize_answer("den")


def test_exact_match_true_after_normalization():
    assert exact_match("Đen, kịt.", "đen kịt") is True


def test_exact_match_false_for_different_words():
    assert exact_match("Trắng", "Đen") is False


# --------------------------------------------------------------------------
# token_f1
# --------------------------------------------------------------------------


def test_token_f1_identical_answers_scores_1():
    assert token_f1("phủ tro bão", "phủ tro bão") == pytest.approx(1.0)


def test_token_f1_disjoint_answers_scores_0():
    assert token_f1("trắng xóa", "đen kịt") == pytest.approx(0.0)


def test_token_f1_partial_overlap_scores_between_0_and_1():
    score = token_f1("phủ tro", "phủ tro bão")
    assert 0.0 < score < 1.0


def test_token_f1_empty_prediction_scores_0():
    assert token_f1("", "đen kịt") == pytest.approx(0.0)


# --------------------------------------------------------------------------
# score_query
# --------------------------------------------------------------------------


def test_score_query_correct_answer_with_grounded_evidence():
    gt = _gt()
    prediction = _prediction(
        answer="Đen",
        evidence=(_kf("L21_V001", 11.0),),
        context=(_kf("L21_V001", 11.0),),
    )

    result = score_query(prediction, gt, CONFIDENCE_THRESHOLD)

    assert result.exact_match is True
    assert result.evidence_grounded is True
    assert result.retrieval_hit is True
    assert result.grounded_correct is True


def test_score_query_correct_text_but_evidence_outside_segment_is_not_grounded():
    gt = _gt()
    prediction = _prediction(
        answer="Đen",
        evidence=(_kf("L21_V001", 999.0),),
        context=(_kf("L21_V001", 11.0), _kf("L21_V001", 999.0)),
    )

    result = score_query(prediction, gt, CONFIDENCE_THRESHOLD)

    assert result.exact_match is True
    assert result.evidence_grounded is False
    assert result.retrieval_hit is True  # context DID contain a correct keyframe
    assert result.grounded_correct is False


def test_score_query_wrong_answer_is_never_grounded_correct_even_with_right_evidence():
    gt = _gt()
    prediction = _prediction(
        answer="Trắng",
        evidence=(_kf("L21_V001", 11.0),),
        context=(_kf("L21_V001", 11.0),),
    )

    result = score_query(prediction, gt, CONFIDENCE_THRESHOLD)

    assert result.exact_match is False
    assert result.grounded_correct is False


def test_score_query_retrieval_hit_false_when_context_never_touches_segment():
    gt = _gt()
    prediction = _prediction(
        answer="Đen",
        evidence=(),
        context=(_kf("L21_V001", 500.0),),
    )

    result = score_query(prediction, gt, CONFIDENCE_THRESHOLD)

    assert result.retrieval_hit is False


def test_score_query_confident_requires_answered_status_and_threshold():
    gt = _gt()
    answered_confident = score_query(_prediction(status="answered", confidence=0.9), gt, CONFIDENCE_THRESHOLD)
    answered_low_confidence = score_query(_prediction(status="answered", confidence=0.1), gt, CONFIDENCE_THRESHOLD)
    abstained_high_confidence = score_query(
        _prediction(status="insufficient_evidence", answer="Không đủ căn cứ", confidence=0.9), gt, CONFIDENCE_THRESHOLD
    )
    unavailable = score_query(_prediction(status="unavailable", answer=None, confidence=None), gt, CONFIDENCE_THRESHOLD)

    assert answered_confident.confident is True
    assert answered_low_confidence.confident is False
    assert abstained_high_confidence.confident is False
    assert unavailable.confident is False


def test_score_query_unavailable_status_scores_as_wrong_not_a_crash():
    gt = _gt()
    prediction = _prediction(status="unavailable", answer=None, confidence=None)

    result = score_query(prediction, gt, CONFIDENCE_THRESHOLD)

    assert result.exact_match is False
    assert result.grounded_correct is False


# --------------------------------------------------------------------------
# aggregate functions
# --------------------------------------------------------------------------


def _result(
    status: str = "answered",
    exact: bool = True,
    f1: float = 1.0,
    retrieval_hit: bool = True,
    evidence_grounded: bool = True,
    confident: bool = True,
) -> VqaQueryResult:
    return VqaQueryResult(
        status=status,
        exact_match=exact,
        token_f1=f1,
        retrieval_hit=retrieval_hit,
        evidence_grounded=evidence_grounded,
        grounded_correct=exact and evidence_grounded,
        confident=confident,
    )


def test_exact_match_rate_averages_over_results():
    results = [_result(exact=True), _result(exact=False)]
    assert exact_match_rate(results) == pytest.approx(0.5)


def test_exact_match_rate_raises_on_empty_results():
    with pytest.raises(ValueError):
        exact_match_rate([])


def test_mean_token_f1_averages_over_results():
    results = [_result(f1=1.0), _result(f1=0.0)]
    assert mean_token_f1(results) == pytest.approx(0.5)


def test_grounded_accuracy_requires_both_exact_match_and_evidence_grounded():
    results = [
        _result(exact=True, evidence_grounded=True),
        _result(exact=True, evidence_grounded=False),
        _result(exact=False, evidence_grounded=True),
    ]
    assert grounded_accuracy(results) == pytest.approx(1 / 3)


def test_retrieval_conditioned_accuracy_only_counts_retrieval_hits():
    results = [
        _result(retrieval_hit=True, exact=True, evidence_grounded=True),
        _result(retrieval_hit=True, exact=False, evidence_grounded=False),
        _result(retrieval_hit=False, exact=True, evidence_grounded=True),  # excluded
    ]
    assert retrieval_conditioned_accuracy(results) == pytest.approx(0.5)


def test_retrieval_conditioned_accuracy_raises_when_no_query_had_a_retrieval_hit():
    results = [_result(retrieval_hit=False)]
    with pytest.raises(ValueError):
        retrieval_conditioned_accuracy(results)


def test_hallucination_rate_is_confident_and_not_retrieval_hit():
    results = [
        _result(confident=True, retrieval_hit=False),  # hallucination
        _result(confident=True, retrieval_hit=True),  # grounded confidence, fine
        _result(confident=False, retrieval_hit=False),  # not confident, fine
    ]
    assert hallucination_rate(results) == pytest.approx(1 / 3)


def test_abstention_rate_counts_insufficient_evidence_status():
    results = [
        _result(status="insufficient_evidence"),
        _result(status="answered"),
        _result(status="answered"),
    ]
    assert abstention_rate(results) == pytest.approx(1 / 3)


def test_abstention_rate_raises_on_empty_results():
    with pytest.raises(ValueError):
        abstention_rate([])


# --------------------------------------------------------------------------
# mean_llm_judge_score -- reference-only rubric aggregate (plan §P3.4:
# "Chỉ tham khảo -- có thiên lệch tự đánh giá"). The actual Gemini-calling
# judge client is I/O and out of scope for this pure metrics module (mirrors
# eval/metrics/retrieval.py never calling Milvus).
# --------------------------------------------------------------------------


def test_mean_llm_judge_score_averages_scores():
    assert mean_llm_judge_score([3, 5]) == pytest.approx(4.0)


def test_mean_llm_judge_score_raises_on_empty():
    with pytest.raises(ValueError):
        mean_llm_judge_score([])


def test_mean_llm_judge_score_raises_on_out_of_range_score():
    with pytest.raises(ValueError):
        mean_llm_judge_score([0])
    with pytest.raises(ValueError):
        mean_llm_judge_score([6])
