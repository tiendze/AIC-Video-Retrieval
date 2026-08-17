"""VQA evaluation orchestration for `eval/run.py` (plan §P3.4 wiring step,
PRD §6.7).

Kept as its own module -- `eval/run.py` is already past the 800-line file
ceiling in `.claude/rules/ecc/common/coding-style.md` (chủ dự án 2026-08-14:
split rather than grow it further). `eval/run.py` stays the single CLI entry
point (CLAUDE.md §6); this module supplies the functions it calls when
`score_vqa: true`, the same separation `eval/trake_run.py` already uses for
TRAKE.

One VQA query = one record in `configs/eval/aic_qa_v1.json` (P3.3's frozen,
already-reviewed QA eval set -- every record in that file is approved by
construction, so `load_qa_eval_set` does no filtering). `question` goes
through `QaService.answer_question(question, top_k)`
(`app/services/vqa_service.py`); `video_id`/`start_sec`/`end_sec`/
`reference_answer` become the `eval.metrics.vqa` ground truth `score_query`
scores against.

This module does NOT call Gemini, Milvus, or MongoDB itself -- it is driven
through the `QaAnswerer` protocol, satisfied by a Fake in tests
(`tests/test_vqa_run.py`) and by the real `QaService` in `eval/run.py`,
mirroring `eval/trake_run.py`'s `TrakeSearcher` protocol.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from eval.metrics.vqa import (
    VqaContextKeyframe,
    VqaGroundTruthQuery,
    VqaPrediction,
    VqaQueryResult,
    abstention_rate,
    exact_match_rate,
    grounded_accuracy,
    hallucination_rate,
    mean_token_f1,
    retrieval_conditioned_accuracy,
    score_query,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VqaEvalRecord:
    """One P3.3 frozen QA eval-set record (`configs/eval/aic_qa_v1.json`),
    reduced to what this module needs. `answerable_by_ocr` is carried
    through -- not used in scoring itself, but P3.3's plan note requires
    reporting Grounded Accuracy split by this flag (CLAUDE.md §4's
    OCR-leakage trap: a question answerable purely from on-screen text
    would credit the VLM for what the OCR channel actually found)."""

    video_id: str
    start_sec: float
    end_sec: float
    question: str
    reference_answer: str
    answerable_by_ocr: bool


def load_qa_eval_set(records: Sequence[dict[str, Any]]) -> list[VqaEvalRecord]:
    """Adapts raw dicts already parsed from `configs/eval/aic_qa_v1.json`
    (P3.3) -- the caller does the JSON read, mirroring how `eval/run.py`
    loads its own ground-truth JSON before handing it to an adapter.
    `keyframe_id` (present in the real file, used only to generate the
    question in P3.3) is intentionally not carried through -- scoring needs
    the segment window, not the specific keyframe that produced the
    caption."""
    return [
        VqaEvalRecord(
            video_id=r["video_id"],
            start_sec=r["start_sec"],
            end_sec=r["end_sec"],
            question=r["question"],
            reference_answer=r["reference_answer"],
            answerable_by_ocr=r["answerable_by_ocr"],
        )
        for r in records
    ]


class QaAnswerer(Protocol):
    """`QaService.answer_question`'s shape -- a Fake in tests, the real
    service in `eval/run.py`."""

    async def answer_question(self, question: str, top_k: int) -> Any: ...


def _to_context_keyframe(hit: Any) -> VqaContextKeyframe:
    """Normalizes one `QaResult.keyframes` entry (`SearchHit`,
    `app/services/search_service.py`) to what `eval.metrics.vqa` needs --
    mirrors `eval/run.py::_hit_to_retrieved`."""
    return VqaContextKeyframe(video_id=hit.video_id, timestamp=hit.timestamp)


def _to_prediction(result: Any) -> VqaPrediction:
    """`QaResult` -> `VqaPrediction`. `evidence_keyframe_ids` are strings;
    resolved to `(video_id, timestamp)` pairs by matching against
    `result.keyframes` -- the same grid the ids were cited from. An id with
    no match in the grid is DROPPED rather than raising: P3.2's grounding
    gate already guarantees every surviving citation is inside the set shown
    to the model, so a mismatch here would mean a QaService wiring bug, not
    bad eval data -- not this module's job to diagnose that, and treating
    the citation as ungrounded is the conservative (never overcounts
    Grounded Accuracy) outcome either way."""
    context = tuple(_to_context_keyframe(kf) for kf in result.keyframes)
    by_id = {kf.keyframe_id: _to_context_keyframe(kf) for kf in result.keyframes}
    evidence = tuple(by_id[kf_id] for kf_id in result.evidence_keyframe_ids if kf_id in by_id)
    return VqaPrediction(
        status=result.status,
        answer=result.answer,
        confidence=result.confidence,
        evidence_keyframes=evidence,
        context_keyframes=context,
    )


async def run_vqa_queries(
    answerer: QaAnswerer,
    records: Sequence[VqaEvalRecord],
    top_k: int,
    confidence_threshold: float,
) -> list[VqaQueryResult]:
    """Runs every record's `question` through `answerer.answer_question()`
    and scores each with `eval.metrics.vqa.score_query`, in order."""
    results: list[VqaQueryResult] = []
    for record in records:
        result = await answerer.answer_question(record.question, top_k)
        prediction = _to_prediction(result)
        ground_truth = VqaGroundTruthQuery(
            video_id=record.video_id,
            start_sec=record.start_sec,
            end_sec=record.end_sec,
            reference_answer=record.reference_answer,
        )
        results.append(score_query(prediction, ground_truth, confidence_threshold))
    return results


@dataclass(frozen=True)
class VqaGroupMetrics:
    """One group's (overall, or one `answerable_by_ocr` split) aggregated
    VQA metrics."""

    num_queries: int
    exact_match: float
    token_f1: float
    grounded_accuracy: float
    retrieval_conditioned_accuracy: float | None  # None when no query had a retrieval hit
    hallucination_rate: float
    abstention_rate: float


def _aggregate_group(results: Sequence[VqaQueryResult]) -> VqaGroupMetrics:
    has_retrieval_hit = any(r.retrieval_hit for r in results)
    return VqaGroupMetrics(
        num_queries=len(results),
        exact_match=exact_match_rate(results),
        token_f1=mean_token_f1(results),
        grounded_accuracy=grounded_accuracy(results),
        retrieval_conditioned_accuracy=(
            retrieval_conditioned_accuracy(results) if has_retrieval_hit else None
        ),
        hallucination_rate=hallucination_rate(results),
        abstention_rate=abstention_rate(results),
    )


@dataclass(frozen=True)
class VqaReportMetrics:
    """Full P3.4 report: `overall` is PRD §6.7's headline numbers;
    `answerable_by_ocr`/`not_answerable_by_ocr` isolate the OCR-leakage trap
    (P3.3's plan note) -- either is `None` when the eval set has zero
    records with that flag, not a misleading 0.0."""

    overall: VqaGroupMetrics
    answerable_by_ocr: VqaGroupMetrics | None
    not_answerable_by_ocr: VqaGroupMetrics | None
    wall_clock_sec: float


def aggregate_vqa_metrics(
    paired: Sequence[tuple[bool, VqaQueryResult]], wall_clock_sec: float
) -> VqaReportMetrics:
    """`paired` is `(record.answerable_by_ocr, result)` in the same order
    `run_vqa_queries` produced `result` for `record`."""
    all_results = [r for _, r in paired]
    ocr_results = [r for flag, r in paired if flag]
    non_ocr_results = [r for flag, r in paired if not flag]
    return VqaReportMetrics(
        overall=_aggregate_group(all_results),
        answerable_by_ocr=_aggregate_group(ocr_results) if ocr_results else None,
        not_answerable_by_ocr=_aggregate_group(non_ocr_results) if non_ocr_results else None,
        wall_clock_sec=wall_clock_sec,
    )


async def run_vqa_eval(
    answerer: QaAnswerer,
    records: Sequence[VqaEvalRecord],
    top_k: int,
    confidence_threshold: float,
) -> VqaReportMetrics:
    """Full VQA eval pass: run every record, score it, aggregate. Raises
    `ValueError` (not a silently empty report) when `records` is empty --
    CLAUDE.md §7: missing data must fail loudly."""
    if not records:
        raise ValueError("run_vqa_eval: records is empty -- nothing to evaluate")

    start = time.perf_counter()
    results = await run_vqa_queries(answerer, records, top_k, confidence_threshold)
    wall_clock_sec = time.perf_counter() - start

    paired = list(zip((r.answerable_by_ocr for r in records), results, strict=True))
    metrics = aggregate_vqa_metrics(paired, wall_clock_sec)
    logger.info(
        "vqa: %d quer(ies), Grounded Accuracy=%.4f, Hallucination Rate=%.4f, wall_clock=%.2fs",
        metrics.overall.num_queries,
        metrics.overall.grounded_accuracy,
        metrics.overall.hallucination_rate,
        wall_clock_sec,
    )
    return metrics


def _format_group_row(label: str, g: VqaGroupMetrics | None) -> str:
    if g is None:
        return f"| {label} | n/a | n/a | n/a | n/a | n/a | n/a | n/a |"
    rca = "n/a" if g.retrieval_conditioned_accuracy is None else f"{g.retrieval_conditioned_accuracy:.4f}"
    return (
        f"| {label} | {g.num_queries} | {g.exact_match:.4f} | {g.token_f1:.4f} | "
        f"{g.grounded_accuracy:.4f} | {rca} | {g.hallucination_rate:.4f} | "
        f"{g.abstention_rate:.4f} |"
    )


def render_vqa_report_md(metrics: VqaReportMetrics) -> str:
    lines = [
        "## VQA metrics (PRD §6.7)",
        "",
        "| group | queries | EM | Token-F1 | Grounded Accuracy | "
        "Retrieval-conditioned Accuracy | Hallucination Rate | Abstention Rate |",
        "|---|---|---|---|---|---|---|---|",
        _format_group_row("overall", metrics.overall),
        _format_group_row("answerable_by_ocr=true", metrics.answerable_by_ocr),
        _format_group_row("answerable_by_ocr=false", metrics.not_answerable_by_ocr),
        "",
        f"wall-clock: {metrics.wall_clock_sec:.2f}s",
        "",
        "Đọc `Grounded Accuracy` cùng `Hallucination Rate`: accuracy cao mà "
        "hallucination cũng cao nghĩa là hệ thống đang đoán bừa (CLAUDE.md §8). "
        "So sánh hàng `answerable_by_ocr=true` với `=false` để tách công của "
        "kênh OCR khỏi công của VLM (plan §P3.3).",
    ]
    return "\n".join(lines) + "\n"
