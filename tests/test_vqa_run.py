"""RED reproducer for eval/vqa_run.py (plan §P3.4 wiring step) -- VQA
evaluation orchestration kept as its own module, mirroring eval/trake_run.py's
split from eval/run.py (CLAUDE.md §6's 800-line ceiling; chủ dự án
2026-08-14 confirmed splitting rather than growing eval/run.py further).

A Fake stands in for QaService -- no Gemini/Milvus/Mongo here, mirroring
tests/test_trake_run.py's own pattern; the real
`python -m eval.run --config configs/eval/aic_qa_v1.yaml --score-vqa` run
against the live QaService is a manual step (CLAUDE.md §10), not exercised by
this test.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from eval.vqa_run import (
    VqaEvalRecord,
    VqaGroupMetrics,
    VqaReportMetrics,
    aggregate_vqa_metrics,
    load_qa_eval_set,
    render_vqa_report_md,
    run_vqa_eval,
    run_vqa_queries,
)
from eval.metrics.vqa import VqaQueryResult


@dataclass(frozen=True)
class _FakeSearchHit:
    keyframe_id: str
    video_id: str
    timestamp: float


@dataclass(frozen=True)
class _FakeQaResult:
    """Duck-types `QaResult` (app/services/vqa_service.py) -- only the
    fields run_vqa_queries actually reads."""

    status: str
    answer: str | None
    confidence: float | None
    evidence_keyframe_ids: tuple[str, ...]
    keyframes: tuple[_FakeSearchHit, ...]


class _FakeAnswerer:
    """Stands in for QaService -- same `answer_question(question, top_k)`
    shape."""

    def __init__(self, results_by_call: list[_FakeQaResult]) -> None:
        self._results_by_call = results_by_call
        self.calls: list[tuple[str, int]] = []

    async def answer_question(self, question: str, top_k: int) -> _FakeQaResult:
        self.calls.append((question, top_k))
        return self._results_by_call[len(self.calls) - 1]


def _record(
    video_id: str = "L21_V001",
    start: float = 10.0,
    end: float = 12.0,
    question: str = "Cây cối màu gì?",
    reference: str = "Đen kịt",
    answerable_by_ocr: bool = False,
) -> VqaEvalRecord:
    return VqaEvalRecord(
        video_id=video_id,
        start_sec=start,
        end_sec=end,
        question=question,
        reference_answer=reference,
        answerable_by_ocr=answerable_by_ocr,
    )


def _grounded_hit() -> _FakeQaResult:
    kf = _FakeSearchHit(keyframe_id="L21_V001_f100", video_id="L21_V001", timestamp=11.0)
    return _FakeQaResult(
        status="answered",
        answer="Đen kịt",
        confidence=0.9,
        evidence_keyframe_ids=("L21_V001_f100",),
        keyframes=(kf,),
    )


# --------------------------------------------------------------------------
# load_qa_eval_set
# --------------------------------------------------------------------------


def test_load_qa_eval_set_converts_raw_json_records():
    raw = [
        {
            "video_id": "L21_V001",
            "start_sec": 10.0,
            "end_sec": 12.0,
            "question": "Cây cối màu gì?",
            "reference_answer": "Đen kịt",
            "answerable_by_ocr": False,
            "keyframe_id": "L21_V001_f100",  # present in the real file, unused here
        }
    ]

    records = load_qa_eval_set(raw)

    assert records == [_record()]


# --------------------------------------------------------------------------
# run_vqa_queries
# --------------------------------------------------------------------------


async def test_run_vqa_queries_sends_question_and_top_k():
    answerer = _FakeAnswerer([_grounded_hit()])

    await run_vqa_queries(answerer, [_record()], top_k=10, confidence_threshold=0.5)

    assert answerer.calls == [("Cây cối màu gì?", 10)]


async def test_run_vqa_queries_resolves_evidence_ids_to_context_keyframes():
    answerer = _FakeAnswerer([_grounded_hit()])

    results = await run_vqa_queries(answerer, [_record()], top_k=10, confidence_threshold=0.5)

    assert results[0].evidence_grounded is True
    assert results[0].grounded_correct is True


async def test_run_vqa_queries_drops_evidence_ids_missing_from_the_grid_without_raising():
    kf = _FakeSearchHit(keyframe_id="L21_V001_f999", video_id="L21_V001", timestamp=999.0)
    result = _FakeQaResult(
        status="answered",
        answer="Đen kịt",
        confidence=0.9,
        evidence_keyframe_ids=("not_in_grid",),
        keyframes=(kf,),
    )
    answerer = _FakeAnswerer([result])

    results = await run_vqa_queries(answerer, [_record()], top_k=10, confidence_threshold=0.5)

    assert results[0].evidence_grounded is False


async def test_run_vqa_queries_one_result_per_record_in_order():
    answerer = _FakeAnswerer([_grounded_hit(), _grounded_hit()])
    records = [_record(video_id="L21_V001"), _record(video_id="L21_V002")]

    results = await run_vqa_queries(answerer, records, top_k=10, confidence_threshold=0.5)

    assert len(results) == 2
    assert all(isinstance(r, VqaQueryResult) for r in results)


# --------------------------------------------------------------------------
# aggregate_vqa_metrics
# --------------------------------------------------------------------------


def _result(
    exact: bool = True,
    retrieval_hit: bool = True,
    evidence_grounded: bool = True,
    confident: bool = True,
    status: str = "answered",
) -> VqaQueryResult:
    return VqaQueryResult(
        status=status,
        exact_match=exact,
        token_f1=1.0 if exact else 0.0,
        retrieval_hit=retrieval_hit,
        evidence_grounded=evidence_grounded,
        grounded_correct=exact and evidence_grounded,
        confident=confident,
    )


def test_aggregate_vqa_metrics_computes_overall_group():
    paired = [(False, _result(exact=True)), (False, _result(exact=False))]

    metrics = aggregate_vqa_metrics(paired, wall_clock_sec=1.5)

    assert metrics.overall.num_queries == 2
    assert metrics.overall.exact_match == pytest.approx(0.5)
    assert metrics.wall_clock_sec == pytest.approx(1.5)


def test_aggregate_vqa_metrics_splits_by_answerable_by_ocr_flag():
    paired = [
        (True, _result(exact=True)),
        (False, _result(exact=False)),
        (False, _result(exact=False)),
    ]

    metrics = aggregate_vqa_metrics(paired, wall_clock_sec=0.0)

    assert metrics.answerable_by_ocr is not None
    assert metrics.answerable_by_ocr.num_queries == 1
    assert metrics.answerable_by_ocr.exact_match == pytest.approx(1.0)
    assert metrics.not_answerable_by_ocr is not None
    assert metrics.not_answerable_by_ocr.num_queries == 2
    assert metrics.not_answerable_by_ocr.exact_match == pytest.approx(0.0)


def test_aggregate_vqa_metrics_ocr_group_is_none_when_no_such_record_exists():
    paired = [(False, _result())]

    metrics = aggregate_vqa_metrics(paired, wall_clock_sec=0.0)

    assert metrics.answerable_by_ocr is None
    assert metrics.not_answerable_by_ocr is not None


def test_aggregate_vqa_metrics_retrieval_conditioned_accuracy_is_none_without_a_retrieval_hit():
    paired = [(False, _result(retrieval_hit=False))]

    metrics = aggregate_vqa_metrics(paired, wall_clock_sec=0.0)

    assert metrics.overall.retrieval_conditioned_accuracy is None


# --------------------------------------------------------------------------
# run_vqa_eval
# --------------------------------------------------------------------------


async def test_run_vqa_eval_raises_on_empty_records():
    answerer = _FakeAnswerer([])
    with pytest.raises(ValueError):
        await run_vqa_eval(answerer, [], top_k=10, confidence_threshold=0.5)


async def test_run_vqa_eval_returns_report_metrics():
    answerer = _FakeAnswerer([_grounded_hit()])

    metrics = await run_vqa_eval(answerer, [_record()], top_k=10, confidence_threshold=0.5)

    assert isinstance(metrics, VqaReportMetrics)
    assert metrics.overall.num_queries == 1
    assert metrics.wall_clock_sec >= 0.0


# --------------------------------------------------------------------------
# render_vqa_report_md
# --------------------------------------------------------------------------


def test_render_vqa_report_md_includes_headline_numbers():
    group = VqaGroupMetrics(
        num_queries=1,
        exact_match=1.0,
        token_f1=1.0,
        grounded_accuracy=1.0,
        retrieval_conditioned_accuracy=1.0,
        hallucination_rate=0.0,
        abstention_rate=0.0,
    )
    metrics = VqaReportMetrics(
        overall=group, answerable_by_ocr=None, not_answerable_by_ocr=group, wall_clock_sec=2.0
    )

    report = render_vqa_report_md(metrics)

    assert "Grounded Accuracy" in report
    assert "Hallucination Rate" in report
    assert "1.0000" in report


def test_render_vqa_report_md_shows_n_a_when_retrieval_conditioned_accuracy_is_none():
    group = VqaGroupMetrics(
        num_queries=1,
        exact_match=0.0,
        token_f1=0.0,
        grounded_accuracy=0.0,
        retrieval_conditioned_accuracy=None,
        hallucination_rate=1.0,
        abstention_rate=0.0,
    )
    metrics = VqaReportMetrics(
        overall=group, answerable_by_ocr=None, not_answerable_by_ocr=None, wall_clock_sec=0.0
    )

    report = render_vqa_report_md(metrics)

    assert "n/a" in report
