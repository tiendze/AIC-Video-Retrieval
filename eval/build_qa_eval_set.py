"""CLI: sinh bộ nhãn eval QA cho AIC 2026 (P3.3, .claude/plans/
p3-vqa-submission.plan.md) -- dataset không có nhãn QA nào, giống mọi nhãn
khác của dự án (PRD §1.4/§6.5).

Reuses the ALREADY-SAMPLED, ALREADY-REVIEWED KIS/TRAKE shot universe from
`configs/eval/aic_eval_v1.candidates.json` (P0.4', eval/build_eval_set.py)
instead of re-sampling -- "cùng cách lấy mẫu, cùng seed" per the plan. Each
approved record there already carries a caption, a keyframe_id, and an
image_path, so this module only needs the `approved: true` records, not
MongoDB/keyframe access again.

Why a SEPARATE file, not literally appended to build_eval_set.py: that file
already sits near the CLAUDE.md 800-line file-size ceiling. Shared
primitives (`EvalSetBuildError`, `_git_sha`, `_load_yaml`, `_shot_id_key`)
are imported from there, not duplicated.

Generation is text-only (reads the shot's existing caption, NOT the
keyframe image) via a LOCAL Ollama model, deliberately a DIFFERENT model
family than the VQA answering model (`GeminiVqaClient`, gemini-3.6-flash,
`configs/qa/vqa.yaml`, P3.1/P3.2) -- otherwise Grounded Accuracy would
measure Gemini grading itself (CLAUDE.md §3 "Model sinh nhãn ≠ model bị
đánh giá"). Chủ dự án has no budget for a paid LLM API (2026-08-16
decision) and declined a local VISION model to keep VRAM usage low and
sequential with BEiT-3 (CLAUDE.md §1.4) -- see references.md §5.6.

Two-phase, same reason as build_eval_set.py: the manual review step (loại
câu hỏi trả lời được bằng kiến thức nền, gắn cờ `answerable_by_ocr`) cannot
be automated (CLAUDE.md §3 "hệ thống interactive -- con người chốt đáp án
cuối"):

    generate  read APPROVED aic_eval_v1.candidates.json records -> local
              Ollama model generates 1-2 visually-grounded questions +
              reference answers per shot -> configs/eval/aic_qa_v1.
              candidates.json for manual review (reviewer fills "approved"
              AND "answerable_by_ocr")
    freeze    read reviewed candidates -> quality gate (>=100 reviewed
              samples per plan §P3.3, self-grading guard) -> configs/eval/
              aic_qa_v1.json + .sha256 + .quality_report.md

WARNING: `generate` calls a local Ollama server per shot. Per CLAUDE.md
§10, this is prepared for the project owner to run, NOT auto-executed.
Same resume/checkpoint discipline as build_eval_set.py: `--out` existing ->
already-generated shots are kept, not regenerated; checkpointed after EACH
shot so a mid-run failure does not discard prior generations.

Usage:
  python -m eval.build_qa_eval_set generate --limit-shots 5
  python -m eval.build_qa_eval_set generate
  # ... người review điền "approved": true/false VÀ "answerable_by_ocr" ...
  python -m eval.build_qa_eval_set freeze
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field

from eval.build_eval_set import EvalSetBuildError, _git_sha, _load_yaml, _shot_id_key

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG_PATH = _REPO_ROOT / "configs" / "eval" / "build_qa_eval_set.yaml"
_DEFAULT_SOURCE_PATH = _REPO_ROOT / "configs" / "eval" / "aic_eval_v1.candidates.json"
_DEFAULT_CANDIDATES_PATH = _REPO_ROOT / "configs" / "eval" / "aic_qa_v1.candidates.json"
_DEFAULT_FROZEN_PATH = _REPO_ROOT / "configs" / "eval" / "aic_qa_v1.json"
_DEFAULT_VQA_CONFIG_PATH = _REPO_ROOT / "configs" / "qa" / "vqa.yaml"

# PRD §6.5 / plan §P3.3: "1-2 câu hỏi" per shot -- a fixed protocol
# invariant from the plan itself, not an operator tuning knob (same pattern
# as build_eval_set.py's _QUALITY_GATE_MAX_REJECTION_RATE).
_MIN_QUESTIONS_PER_SHOT = 1
_MAX_QUESTIONS_PER_SHOT = 2

_QA_SYSTEM_PROMPT = (
    "Bạn nhận được MỘT câu mô tả (caption) tiếng Việt về một cảnh (shot) "
    "trong video tin tức. Hãy sinh 1 đến 2 câu hỏi, MỖI câu hỏi kèm đáp án "
    "tham chiếu ngắn gọn, sao cho câu hỏi CHỈ trả lời được bằng cách xem "
    "chính cảnh đó -- không phải bằng kiến thức nền chung chung. TUYỆT ĐỐI "
    "KHÔNG hỏi những điều có thể trả lời mà không cần xem video (ví dụ: "
    "kiến thức phổ thông, định nghĩa, sự kiện không được caption nhắc "
    "tới). KHÔNG bịa chi tiết không có trong caption. Trả lời bằng tiếng "
    "Việt."
)

_QA_GROUND_TRUTH_FIELDS = (
    "video_id",
    "start_sec",
    "end_sec",
    "keyframe_id",
    "question",
    "reference_answer",
    "answerable_by_ocr",
)


@dataclass(frozen=True)
class BuildQaEvalSetConfig:
    """Mirrors `configs/eval/build_qa_eval_set.yaml` -- see that file for
    the rationale behind each default."""

    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen3:8b"
    max_retries: int = 3
    retry_base_delay_seconds: float = 1.0
    min_reviewed_samples: int = 100


_DEFAULT_CONFIG = BuildQaEvalSetConfig()


class QaItem(BaseModel):
    question: str
    reference_answer: str


class QaGeneration(BaseModel):
    items: list[QaItem] = Field(
        min_length=_MIN_QUESTIONS_PER_SHOT, max_length=_MAX_QUESTIONS_PER_SHOT
    )


@dataclass(frozen=True)
class QaQualityReport:
    total_candidates: int
    approved_count: int
    rejection_rate: float
    answerable_by_ocr_rate: float  # among APPROVED records


# --------------------------------------------------------------------------
# Source loading -- the shot universe is P0.4''s already-sampled, already-
# reviewed KIS/TRAKE candidates, not a fresh sample.
# --------------------------------------------------------------------------


def load_approved_source_records(candidates_path: Path) -> list[dict[str, Any]]:
    doc = json.loads(candidates_path.read_text(encoding="utf-8"))
    return [r for r in doc["records"] if r.get("approved") is True]


# --------------------------------------------------------------------------
# I/O boundary -- Protocol + local Ollama implementation (tests/test_ocr.py
# pattern: Protocol-faked in tests, no server needed to run the suite).
# --------------------------------------------------------------------------


class QaCandidateGenerator(Protocol):
    def generate(self, caption: str) -> list[QaItem]: ...


class OllamaQaGenerator:
    """Wraps `ollama.Client.chat(..., format=QaGeneration.model_json_schema())`
    -- structured output via JSON-schema-constrained local generation
    (references.md §5.6), deliberately NOT Gemini/OpenAI (self-grading
    guard, see module docstring). Text-only: reads the shot's existing
    caption, not the keyframe image."""

    def __init__(self, config: BuildQaEvalSetConfig = _DEFAULT_CONFIG) -> None:
        from ollama import Client  # lazy: only needed when this class is used

        self._client = Client(host=config.ollama_host)
        self._config = config

    def generate(self, caption: str) -> list[QaItem]:
        import httpx

        last_exc: Exception | None = None
        for attempt in range(self._config.max_retries):
            try:
                return self._call_once(caption)
            except (httpx.ConnectError, httpx.TimeoutException) as exc:
                last_exc = exc
                time.sleep(self._config.retry_base_delay_seconds * (2**attempt))
        raise RuntimeError(
            f"Ollama QA generation failed after {self._config.max_retries} attempts"
        ) from last_exc

    def _call_once(self, caption: str) -> list[QaItem]:
        response = self._client.chat(
            model=self._config.ollama_model,
            messages=[
                {"role": "system", "content": _QA_SYSTEM_PROMPT},
                {"role": "user", "content": f"Caption: {caption}"},
            ],
            format=QaGeneration.model_json_schema(),
        )
        return QaGeneration.model_validate_json(response.message.content).items


# --------------------------------------------------------------------------
# Orchestration -- pure aside from qa_generator.generate() calls.
# --------------------------------------------------------------------------


def _qa_source_key(record: Mapping[str, Any]) -> tuple[str, int]:
    return (record["video_id"], record["shot_id"])


def _generate_qa_candidates(
    config: BuildQaEvalSetConfig,
    *,
    source_records: Sequence[Mapping[str, Any]],
    qa_generator: QaCandidateGenerator,
    limit_shots: int = 0,
    existing_records: Sequence[Mapping[str, Any]] = (),
    on_progress: Callable[[list[dict[str, Any]], list[str]], None] | None = None,
) -> dict[str, Any]:
    """`existing_records` (loaded from a prior, interrupted `generate` run)
    are kept as-is; their shots are NOT re-sent to Ollama -- resuming after
    a crash must not regenerate shots already done. `on_progress`, called
    after EACH source shot, lets the caller checkpoint to disk (same
    per-shot rationale as build_eval_set.py: losing a whole video's worth
    of generations to one failure partway through is unacceptable)."""
    already_done = {_qa_source_key(r) for r in existing_records}
    tasks = [r for r in source_records if _qa_source_key(r) not in already_done]
    if limit_shots > 0:
        tasks = tasks[:limit_shots]

    records: list[dict[str, Any]] = list(existing_records)
    failed: list[str] = []

    for source in tasks:
        label = _shot_id_key(source["video_id"], source["shot_id"])
        try:
            items = qa_generator.generate(source["caption"])
        except Exception as exc:  # noqa: BLE001 -- ollama.ResponseError and
            # pydantic.ValidationError (malformed JSON despite the schema
            # constraint) are both plain Exception subclasses, not
            # OSError/RuntimeError/ValueError -- same rationale as
            # build_eval_set.py's GeminiCaptionGenerator handling: one
            # shot's failure must not abort the whole batch. Logged with
            # the failing shot's id, recorded in `failed` -- visible in the
            # output, not swallowed (CLAUDE.md §7).
            logger.error("%s: QA generation failed: %s", label, exc)
            failed.append(label)
            if on_progress is not None:
                on_progress(records, failed)
            continue

        for item in items:
            records.append(
                {
                    "video_id": source["video_id"],
                    "shot_id": source["shot_id"],
                    "start_sec": source["start_sec"],
                    "end_sec": source["end_sec"],
                    "keyframe_id": source["keyframe_id"],
                    "question": item.question,
                    "reference_answer": item.reference_answer,
                    "source_caption": source["caption"],
                    "generator_model": config.ollama_model,
                    "answerable_by_ocr": None,
                    "approved": None,
                }
            )
        if on_progress is not None:
            on_progress(records, failed)

    logger.info(
        "generate: %d QA candidate(s) from %d shot(s), %d shot(s) failed",
        len(records) - len(existing_records),
        len(tasks),
        len(failed),
    )

    return {
        "eval_set_version": "v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "config": asdict(config),
        "records": records,
        "failed": failed,
    }


# --------------------------------------------------------------------------
# Freeze -- candidates (with human "approved"/"answerable_by_ocr" decisions)
# -> frozen artifact.
# --------------------------------------------------------------------------


def compute_qa_quality_report(
    candidates: Sequence[Mapping[str, Any]], min_reviewed_samples: int
) -> QaQualityReport:
    """Raises `EvalSetBuildError` (not a silent skip) if any candidate is
    still unreviewed, if fewer than `min_reviewed_samples` were reviewed
    (plan §P3.3: "Review thủ công ít nhất 100 mẫu"), or if an approved
    record is missing its `answerable_by_ocr` flag -- "thiếu dữ liệu phải
    fail rõ ràng" (CLAUDE.md §7)."""
    total = len(candidates)
    unreviewed = [c for c in candidates if c.get("approved") is None]
    if unreviewed:
        sample_ids = [_shot_id_key(c["video_id"], c["shot_id"]) for c in unreviewed[:5]]
        raise EvalSetBuildError(
            f"{len(unreviewed)}/{total} candidate(s) still unreviewed (approved=null), "
            f'e.g. {sample_ids} -- review every record\'s "approved" field before freezing'
        )
    if total < min_reviewed_samples:
        raise EvalSetBuildError(
            f"only {total} reviewed sample(s), need >= {min_reviewed_samples} "
            '(plan §P3.3: "Review thủ công ít nhất 100 mẫu") -- generate more '
            "before freezing"
        )

    approved = [c for c in candidates if c["approved"] is True]
    missing_ocr_flag = [c for c in approved if c.get("answerable_by_ocr") is None]
    if missing_ocr_flag:
        sample_ids = [_shot_id_key(c["video_id"], c["shot_id"]) for c in missing_ocr_flag[:5]]
        raise EvalSetBuildError(
            f"{len(missing_ocr_flag)} approved candidate(s) missing 'answerable_by_ocr', "
            f"e.g. {sample_ids} -- set it at review time (plan §P3.3: tách công OCR "
            "khỏi công VLM khi báo cáo Grounded Accuracy)"
        )

    rejected = total - len(approved)
    ocr_answerable = sum(1 for c in approved if c["answerable_by_ocr"] is True)
    return QaQualityReport(
        total_candidates=total,
        approved_count=len(approved),
        rejection_rate=rejected / total if total else 0.0,
        answerable_by_ocr_rate=ocr_answerable / len(approved) if approved else 0.0,
    )


def assert_generator_differs_from_vqa_model(
    records: Sequence[Mapping[str, Any]], vqa_model: str
) -> None:
    """CLAUDE.md §3 self-grading guard: the model that generated a
    question/reference-answer must not be the same model that answers it at
    eval time (`VqaService`/`GeminiVqaClient`, `configs/qa/vqa.yaml`)."""
    offending = {r["generator_model"] for r in records if r["generator_model"] == vqa_model}
    if offending:
        raise EvalSetBuildError(
            f"QA generator_model '{vqa_model}' matches the VQA answering model -- "
            "this would be the model grading itself (CLAUDE.md §3 'Model sinh nhãn "
            "≠ model bị đánh giá')"
        )


def freeze_qa_records(approved: Sequence[Mapping[str, Any]]) -> tuple[str, str]:
    """Keeps exactly the ground-truth fields consumed by `eval/metrics/
    vqa.py` (P3.4) -- provenance fields (`generator_model`, `source_caption`,
    `approved`) are review-time metadata, not part of the frozen ground
    truth."""
    minimal = [{field: record[field] for field in _QA_GROUND_TRUTH_FIELDS} for record in approved]
    json_text = json.dumps(minimal, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    digest = hashlib.sha256(json_text.encode("utf-8")).hexdigest()
    return json_text, digest


def _render_quality_report(quality: QaQualityReport, candidates_doc: Mapping[str, Any]) -> str:
    return (
        "# aic_qa_v1 quality report\n\n"
        f"- eval_set_version: {candidates_doc['eval_set_version']}\n"
        f"- git_sha: {candidates_doc['git_sha']}\n"
        f"- generated_at: {candidates_doc['generated_at']}\n"
        f"- total candidates: {quality.total_candidates}\n"
        f"- approved: {quality.approved_count}\n"
        f"- tỉ lệ loại khi review: {quality.rejection_rate:.1%}\n"
        f"- tỉ lệ trả lời được bằng OCR (answerable_by_ocr, trong số đã duyệt): "
        f"{quality.answerable_by_ocr_rate:.1%}\n"
    )


def _freeze_qa_candidates(
    candidates_doc: Mapping[str, Any], *, vqa_model: str, min_reviewed_samples: int
) -> tuple[str, str, str]:
    records = candidates_doc["records"]
    quality = compute_qa_quality_report(records, min_reviewed_samples)
    approved = [r for r in records if r["approved"] is True]
    assert_generator_differs_from_vqa_model(approved, vqa_model)
    json_text, digest = freeze_qa_records(approved)
    report_md = _render_quality_report(quality, candidates_doc)
    return json_text, digest, report_md


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _load_config(path: Path) -> BuildQaEvalSetConfig:
    return BuildQaEvalSetConfig(**_load_yaml(path))


def _write_qa_candidates_doc(
    out_path: Path, config: BuildQaEvalSetConfig, records: list[dict[str, Any]], failed: list[str]
) -> None:
    """Shared by the mid-run checkpoint and the final write -- a checkpoint
    left behind by a crash is itself a valid input to resume from."""
    doc = {
        "eval_set_version": "v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "config": asdict(config),
        "records": records,
        "failed": failed,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser(
        "generate", help="Read approved KIS/TRAKE candidates -> local Ollama -> QA candidates JSON"
    )
    generate.add_argument("--config", type=Path, default=_DEFAULT_CONFIG_PATH)
    generate.add_argument("--source", type=Path, default=_DEFAULT_SOURCE_PATH)
    generate.add_argument("--out", type=Path, default=_DEFAULT_CANDIDATES_PATH)
    generate.add_argument(
        "--limit-shots", type=int, default=0, help="0 = no limit; >0 to eyeball a few first"
    )

    freeze = subparsers.add_parser(
        "freeze", help="Read reviewed QA candidates -> freeze aic_qa_v1.json + hash"
    )
    freeze.add_argument("--candidates", type=Path, default=_DEFAULT_CANDIDATES_PATH)
    freeze.add_argument("--out", type=Path, default=_DEFAULT_FROZEN_PATH)
    freeze.add_argument("--vqa-config", type=Path, default=_DEFAULT_VQA_CONFIG_PATH)
    freeze.add_argument("--config", type=Path, default=_DEFAULT_CONFIG_PATH)

    return parser.parse_args()


def _run_generate(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    source_records = load_approved_source_records(args.source)

    existing_records: list[dict[str, Any]] = []
    if args.out.exists():
        existing_records = json.loads(args.out.read_text(encoding="utf-8"))["records"]
        logger.info(
            "%s already has %d record(s) -- resuming, not regenerating them",
            args.out,
            len(existing_records),
        )

    qa_generator = OllamaQaGenerator(config)

    def _checkpoint(records: list[dict[str, Any]], failed: list[str]) -> None:
        _write_qa_candidates_doc(args.out, config, records, failed)

    result = _generate_qa_candidates(
        config,
        source_records=source_records,
        qa_generator=qa_generator,
        limit_shots=args.limit_shots,
        existing_records=existing_records,
        on_progress=_checkpoint,
    )
    _write_qa_candidates_doc(args.out, config, result["records"], result["failed"])
    logger.info("Wrote %d QA candidate(s) to %s", len(result["records"]), args.out)
    return 0


def _run_freeze(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    candidates_doc = json.loads(args.candidates.read_text(encoding="utf-8"))
    vqa_model = _load_yaml(args.vqa_config)["gemini_model"]

    try:
        json_text, digest, report_md = _freeze_qa_candidates(
            candidates_doc, vqa_model=vqa_model, min_reviewed_samples=config.min_reviewed_samples
        )
    except EvalSetBuildError as exc:
        logger.error("freeze refused: %s", exc)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json_text, encoding="utf-8")
    args.out.with_name(args.out.name + ".sha256").write_text(digest + "\n", encoding="utf-8")
    args.out.with_name(args.out.stem + ".quality_report.md").write_text(report_md, encoding="utf-8")
    logger.info("Froze %s (sha256=%s)", args.out, digest)
    return 0


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = _parse_args()
    if args.command == "generate":
        sys.exit(_run_generate(args))
    sys.exit(_run_freeze(args))


if __name__ == "__main__":
    main()
