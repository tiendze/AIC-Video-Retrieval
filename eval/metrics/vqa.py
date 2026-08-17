"""Pure VQA-scoring functions for `eval/run.py` (plan §P3.4, PRD §6.7).

Mirrors `eval/metrics/retrieval.py` and `eval/metrics/trake.py`: dataclasses
reduced to only what scoring needs, pure aggregate functions, `ValueError` on
empty input rather than a silent 0.0 (CLAUDE.md §7 -- a metrics table full of
zeros must never be indistinguishable from "nothing was evaluated").

This module does NOT call Gemini, Milvus, or MongoDB. Wiring it against the
live `QaService` + frozen `configs/eval/aic_qa_v1.json` (P3.3) into a runnable
`eval/run.py` report table is a separate step, the same way `trake.py` has a
companion `eval/trake_run.py` for the I/O side.

Two distinct "grounded" concepts are scored, and conflating them would
silently misreport the headline metric:

- `retrieval_hit` -- does ANY keyframe shown to the model (the context/grid
  `QaService` retrieved) fall inside the ground-truth segment. Answers "did
  retrieval even give the model a chance to be right".
- `evidence_grounded` -- does ANY keyframe the model actually CITED
  (`evidence_keyframe_ids`, already filtered by P3.2's grounding gate to be a
  subset of context) fall inside the ground-truth segment. Answers "did the
  model point at the right one, out of what it was given".

`grounded_accuracy` (PRD §6.7's headline metric) requires both a text-correct
answer AND `evidence_grounded` -- a right answer citing the wrong keyframe is
not grounded, per the plan table ("Trả lời đúng VÀ >=1 keyframe bằng chứng
nằm trong segment đúng").

`hallucination_rate` is defined by the plan table as confident despite the
context never containing a correct keyframe at all (`confident AND NOT
retrieval_hit`) -- deliberately independent of whether the answer text
happens to be right. A confident answer with zero correct keyframes in
context could not have been produced from real evidence, so it is scored as
hallucination even in the (impossible under the P3.2 gate, but not assumed
here) case its text happens to match the reference.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

VqaStatus = Literal["answered", "insufficient_evidence", "unavailable"]

# SQuAD-style normalization stopword: English articles only. Reference
# answers in this eval set are Vietnamese (configs/eval/aic_qa_v1.json),
# which has no direct equivalent, so this is a no-op for that data and only
# matters if an English answer variant is ever scored.
_ENGLISH_ARTICLES = frozenset({"a", "an", "the"})

# \w with Python's default UNICODE str matching keeps Vietnamese diacritics
# (they are Unicode word characters) -- this strips punctuation only, never
# tone marks. "bỏ dấu câu" in the plan means "remove punctuation", not
# "remove diacritics".
_PUNCTUATION_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)


def normalize_answer(text: str) -> str:
    """Lowercase, strip punctuation, drop English articles, collapse
    whitespace (plan §P3.4: "lowercase, bỏ dấu câu, bỏ mạo từ")."""
    lowered = text.lower()
    no_punctuation = _PUNCTUATION_RE.sub(" ", lowered)
    tokens = [t for t in no_punctuation.split() if t not in _ENGLISH_ARTICLES]
    return " ".join(tokens)


def exact_match(prediction: str, reference: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(reference)


def token_f1(prediction: str, reference: str) -> float:
    """Standard SQuAD-style token-F1 over normalized token multisets --
    accepts different phrasing of the same answer, unlike `exact_match`."""
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


@dataclass(frozen=True)
class VqaGroundTruthQuery:
    """One QA eval record's answer key (`configs/eval/aic_qa_v1.json`, P3.3),
    narrowed to what scoring needs."""

    video_id: str
    start_sec: float
    end_sec: float
    reference_answer: str


@dataclass(frozen=True)
class VqaContextKeyframe:
    """One keyframe involved in a query -- either shown to the model
    (context/grid) or cited as evidence. Mirrors `RetrievedHit`
    (`eval/metrics/retrieval.py`)."""

    video_id: str
    timestamp: float


@dataclass(frozen=True)
class VqaPrediction:
    """One system answer for one query -- mirrors `QaResult`
    (`app/services/vqa_service.py`) narrowed to eval-scoring fields.
    `evidence_keyframes` is expected to already be a subset of
    `context_keyframes` (P3.2's grounding gate guarantees this upstream);
    this module does not re-check that subset relationship."""

    status: VqaStatus
    answer: str | None
    confidence: float | None
    evidence_keyframes: tuple[VqaContextKeyframe, ...]
    context_keyframes: tuple[VqaContextKeyframe, ...]


@dataclass(frozen=True)
class VqaQueryResult:
    """One query's full scoring outcome, ready for the aggregate functions
    below."""

    status: VqaStatus
    exact_match: bool
    token_f1: float
    retrieval_hit: bool
    evidence_grounded: bool
    grounded_correct: bool
    confident: bool


def _any_in_segment(
    keyframes: Sequence[VqaContextKeyframe], ground_truth: VqaGroundTruthQuery
) -> bool:
    return any(
        kf.video_id == ground_truth.video_id
        and ground_truth.start_sec <= kf.timestamp <= ground_truth.end_sec
        for kf in keyframes
    )


def score_query(
    prediction: VqaPrediction, ground_truth: VqaGroundTruthQuery, confidence_threshold: float
) -> VqaQueryResult:
    """One query's full outcome. A `status != "answered"` prediction (model
    abstained or Gemini was unavailable) scores as text-wrong by construction
    -- `answer` is `None`/the fixed abstention string, which cannot match a
    real `reference_answer`, so no special-casing is needed to keep
    abstention from ever counting as a correct answer."""
    answer_text = prediction.answer or ""
    em = exact_match(answer_text, ground_truth.reference_answer)
    f1 = token_f1(answer_text, ground_truth.reference_answer)
    retrieval_hit = _any_in_segment(prediction.context_keyframes, ground_truth)
    evidence_grounded = _any_in_segment(prediction.evidence_keyframes, ground_truth)
    confident = (
        prediction.status == "answered"
        and prediction.confidence is not None
        and prediction.confidence >= confidence_threshold
    )

    return VqaQueryResult(
        status=prediction.status,
        exact_match=em,
        token_f1=f1,
        retrieval_hit=retrieval_hit,
        evidence_grounded=evidence_grounded,
        grounded_correct=em and evidence_grounded,
        confident=confident,
    )


def _require_nonempty(results: Sequence[VqaQueryResult], fn_name: str) -> None:
    if not results:
        raise ValueError(f"{fn_name}: results is empty -- nothing was evaluated")


def exact_match_rate(results: Sequence[VqaQueryResult]) -> float:
    _require_nonempty(results, "exact_match_rate")
    return sum(1 for r in results if r.exact_match) / len(results)


def mean_token_f1(results: Sequence[VqaQueryResult]) -> float:
    _require_nonempty(results, "mean_token_f1")
    return sum(r.token_f1 for r in results) / len(results)


def grounded_accuracy(results: Sequence[VqaQueryResult]) -> float:
    """PRD §6.7's headline metric: `grounded_correct` already encodes "text
    correct AND evidence in the right segment" (see `score_query`)."""
    _require_nonempty(results, "grounded_accuracy")
    return sum(1 for r in results if r.grounded_correct) / len(results)


def retrieval_conditioned_accuracy(results: Sequence[VqaQueryResult]) -> float:
    """`grounded_accuracy` restricted to queries where retrieval already
    found a correct keyframe -- isolates VLM reasoning error from retrieval
    error (plan §P3.4). Raises when no query in `results` ever had a
    retrieval hit, same "no silent 0.0" rule as the other aggregates, since
    an empty conditioning subset is a different failure than a real 0%."""
    conditioned = [r for r in results if r.retrieval_hit]
    if not conditioned:
        raise ValueError(
            "retrieval_conditioned_accuracy: no query had a retrieval hit -- nothing to condition on"
        )
    return sum(1 for r in conditioned if r.grounded_correct) / len(conditioned)


def hallucination_rate(results: Sequence[VqaQueryResult]) -> float:
    """Plan §P3.4: fraction of queries answered confidently while the
    context never contained a correct keyframe at all (`confident AND NOT
    retrieval_hit`) -- read together with `grounded_accuracy`, not alone
    (CLAUDE.md §8: high accuracy with high hallucination means the system is
    guessing, not seeing)."""
    _require_nonempty(results, "hallucination_rate")
    return sum(1 for r in results if r.confident and not r.retrieval_hit) / len(results)


def abstention_rate(results: Sequence[VqaQueryResult]) -> float:
    """Fraction of queries where the model answered "Không đủ căn cứ" --
    read alongside `hallucination_rate` (plan §P3.4)."""
    _require_nonempty(results, "abstention_rate")
    return sum(1 for r in results if r.status == "insufficient_evidence") / len(results)


def mean_llm_judge_score(scores: Sequence[int]) -> float:
    """Mean of Gemini's 1-5 rubric scores (plan §P3.4: "LLM-as-Judge --
    Gemini chấm rubric 1-5. Chỉ tham khảo"). Pure aggregate only -- the
    Gemini-calling judge client is I/O and belongs in a separate module that
    produces this function's input, the same way `retrieval.py` never calls
    Milvus itself. Not used for the main metric comparisons (self-evaluation
    bias, per the plan), so it is reported but never gates a decision."""
    if not scores:
        raise ValueError("mean_llm_judge_score: scores is empty -- nothing was judged")
    for score in scores:
        if not 1 <= score <= 5:
            raise ValueError(f"mean_llm_judge_score: score {score} out of the 1-5 rubric range")
    return sum(scores) / len(scores)
