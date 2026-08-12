"""CLI: sinh bộ nhãn eval tự động cho AIC 2026 (PRD §6.1, .claude/plans/
p0-4-build-eval-set.plan.md) -- dataset không kèm nhãn truy vấn nào, nên đây
là điều kiện để `eval/run.py` (P0.9) có gì để chấm.

Hai giai đoạn, vì bước review tay (PRD §6.1.2 bước 4) không tự động hoá
được (CLAUDE.md §3 "hệ thống interactive -- con người chốt đáp án cuối"):

    generate  lấy mẫu shot (căn bậc hai theo batch+author, seed cố định) ->
              Gemini sinh caption tiếng Việt -> lọc tự động caption chung
              chung -> ghi configs/eval/aic_eval_v1.candidates.json cho
              người review điền field "approved"
    freeze    đọc candidates đã review -> tính chỉ số chất lượng (PRD
              §6.1.3) -> cổng chất lượng (loại > 30% thì từ chối đóng băng)
              -> ghi configs/eval/aic_eval_v1.json + .sha256 + .quality_report.md

WARNING: `generate` gọi Gemini hàng loạt (~800 ảnh) và cần MongoDB đang
chạy -- theo CLAUDE.md §10, lệnh này chuẩn bị sẵn để chủ dự án tự chạy,
KHÔNG tự động thực thi.

`generate` tự resume: nếu `--out` đã tồn tại (vd. lần chạy trước bị dừng vì
hết quota Gemini), nó đọc file đó, giữ nguyên caption đã có, và CHỈ gọi
Gemini cho những shot chưa có -- không trả tiền lại cho phần đã xong. File
cũng được ghi (checkpoint) sau MỖI video, không phải chỉ một lần ở cuối, vì
lỗi hết quota/API của Gemini không phải `OSError`/`RuntimeError`/`ValueError`
nên không bị bắt ở vòng lặp per-shot -- nếu không checkpoint, một lỗi như
vậy sẽ làm mất TOÀN BỘ caption đã sinh (đã trả tiền) trong lần chạy đó. Gặp
lỗi giữa chừng -> chạy lại đúng lệnh cũ là tiếp tục, không sinh lại từ đầu.

Usage:
  python -m eval.build_eval_set generate --limit-shots 5   # xem thử vài mẫu trước
  python -m eval.build_eval_set generate
  # ... người review điền "approved": true/false trong file candidates ...
  python -m eval.build_eval_set freeze
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import io
import json
import logging
import math
import random
import re
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import yaml
from PIL import Image
from pydantic import BaseModel, Field

from app.config import Settings
from app.stores.mongo_client import MongoStore
from datasets.aic2026 import AIC2026Adapter
from datasets.base import GroundTruthSegment, batch_of

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG_PATH = _REPO_ROOT / "configs" / "eval" / "build_eval_set.yaml"
_DEFAULT_SHOTS_PATH = _REPO_ROOT / "data" / "shots.json"
_DEFAULT_CANDIDATES_PATH = _REPO_ROOT / "configs" / "eval" / "aic_eval_v1.candidates.json"
_DEFAULT_FROZEN_PATH = _REPO_ROOT / "configs" / "eval" / "aic_eval_v1.json"

# PRD §6.1.3 / R6: "Nếu tỉ lệ caption bị loại khi review > 30%, dừng lại báo
# cáo -- bộ nhãn kém thì mọi metric phía sau chỉ là nhiễu." This is a fixed
# protocol invariant from the PRD itself (not an operator tuning knob), so
# it lives here as a named constant rather than in build_eval_set.yaml.
_QUALITY_GATE_MAX_REJECTION_RATE = 0.30

_GROUND_TRUTH_FIELDS = ("video_id", "start_sec", "end_sec", "order_index", "caption")

_WORD_PATTERN = re.compile(r"\w+", re.UNICODE)


class EvalSetBuildError(Exception):
    """Raised when candidate/eval-set data fails a build_eval_set.py invariant."""


@dataclass(frozen=True)
class BuildEvalSetConfig:
    """Mirrors `configs/eval/build_eval_set.yaml` -- see that file for the
    rationale behind each default."""

    seed: int = 20260809
    split: str = "all"
    kis_sample_size: int = 600
    trake_sequence_count: int = 40
    trake_sequence_length: int = 5
    min_shot_duration_sec: float = 1.0
    max_image_side: int = 384
    min_content_words: int = 4
    near_duplicate_jaccard: float = 0.6
    max_near_duplicates: int = 3
    vietnamese_stopwords: tuple[str, ...] = ()
    max_retries: int = 3
    retry_base_delay_seconds: float = 1.0
    gemini_model: str = "gemini-3.6-flash"


_DEFAULT_CONFIG = BuildEvalSetConfig()


@dataclass(frozen=True)
class ShotRef:
    """One shot, joined from `data/shots.json` (boundaries) with the
    video's metadata (batch/author, from `AIC2026Adapter`)."""

    video_id: str
    shot_id: int
    start_sec: float
    end_sec: float
    batch: str
    author: str


@dataclass(frozen=True)
class VideoMeta:
    batch: str
    author: str
    title: str


@dataclass(frozen=True)
class QualityReport:
    total_candidates: int
    approved_count: int
    rejection_rate: float
    generic_filter_rate: float
    inter_reviewer_agreement: float | None  # None = not_measured (v1: 1 reviewer)


# --------------------------------------------------------------------------
# Data loading -- joins data/shots.json with AIC2026Adapter metadata into
# ShotRef, so every downstream pure function works on one typed shape.
# --------------------------------------------------------------------------


def load_shots_json(path: Path) -> dict[str, list[dict[str, Any]]]:
    """`{"videos": [{"video_name": "L21_V001.mp4", "shots": [...]}]}` ->
    `{video_id: [shot_dict, ...]}`. Authoritative shot-boundary source
    (`pipelines/shot_detect.py`'s TransNetV2 output), not reconstructed."""
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return {entry["video_name"].removesuffix(".mp4"): entry["shots"] for entry in data["videos"]}


def _shot_id_key(video_id: str, shot_id: int) -> str:
    """Matches the `shot_id` string convention already written by
    `pipelines/ingest.py` into MongoDB's `keyframes` collection."""
    return f"{video_id}_shot{shot_id}"


def build_video_meta(adapter: AIC2026Adapter, split: str) -> dict[str, VideoMeta]:
    return {
        v.video_id: VideoMeta(
            batch=batch_of(v.video_id), author=v.category or "unknown", title=v.title or ""
        )
        for v in adapter.iter_videos(split)
    }


def build_shot_refs(
    shots_by_video_name: Mapping[str, Sequence[Mapping[str, Any]]],
    video_meta: Mapping[str, VideoMeta],
) -> dict[str, list[ShotRef]]:
    """Only videos present in `video_meta` (i.e. within the requested split)
    are kept -- `data/shots.json` covers the full 873-video corpus regardless
    of split."""
    result: dict[str, list[ShotRef]] = {}
    for video_id, meta in video_meta.items():
        raw_shots = shots_by_video_name.get(video_id)
        if not raw_shots:
            continue
        result[video_id] = [
            ShotRef(
                video_id=video_id,
                shot_id=s["shot_id"],
                start_sec=s["start_sec"],
                end_sec=s["end_sec"],
                batch=meta.batch,
                author=meta.author,
            )
            for s in sorted(raw_shots, key=lambda x: x["shot_id"])
        ]
    return result


# --------------------------------------------------------------------------
# Sampling -- pure functions, no mocking needed to test.
# --------------------------------------------------------------------------


def eligible_shots(shots: Sequence[ShotRef], min_duration_sec: float) -> list[ShotRef]:
    """Drops flash-cut shots (< p10 of the observed shot-length distribution,
    verify 2026-08-09) that are too short to be a usable retrieval target."""
    return [shot for shot in shots if (shot.end_sec - shot.start_sec) >= min_duration_sec]


def stratum_key(shot: ShotRef) -> tuple[str, str]:
    return (shot.batch, shot.author)


def allocate_by_sqrt(
    stratum_sizes: Mapping[tuple[str, str], int], total: int
) -> dict[tuple[str, str], int]:
    """Largest-remainder apportionment on sqrt(stratum size) weights: a
    stratum with 100x the shots of another gets ~10x the quota (sqrt(100)),
    not 100x -- keeps small strata (e.g. a sparsely-shot batch) from being
    crowded out by a large one (verify 2026-08-09: one batch alone holds
    ~50% of all shots in the corpus). Deterministic given the same inputs
    (stable tie-break by sorted key); `sum(result.values()) == total` unless
    `total` exceeds the number of strata with nonzero weight."""
    if not stratum_sizes or total <= 0:
        return dict.fromkeys(stratum_sizes, 0)

    weights = {key: math.sqrt(size) for key, size in stratum_sizes.items() if size > 0}
    weight_sum = sum(weights.values())
    if weight_sum == 0:
        return dict.fromkeys(stratum_sizes, 0)

    raw = {key: total * weight / weight_sum for key, weight in weights.items()}
    quotas = {key: int(value) for key, value in raw.items()}
    remaining = total - sum(quotas.values())
    remainder_order = sorted(raw, key=lambda key: (-(raw[key] - quotas[key]), key))
    for key in remainder_order[:remaining]:
        quotas[key] += 1

    for key in stratum_sizes:
        quotas.setdefault(key, 0)
    return quotas


def sample_kis_shots(
    shots_by_video: Mapping[str, list[ShotRef]],
    config: BuildEvalSetConfig,
    rng: random.Random,
) -> list[ShotRef]:
    """Textual-KIS sample: stratify eligible shots by (batch, author),
    allocate quotas by `allocate_by_sqrt`, sample within each stratum."""
    by_stratum: dict[tuple[str, str], list[ShotRef]] = defaultdict(list)
    for shots in shots_by_video.values():
        for shot in eligible_shots(shots, config.min_shot_duration_sec):
            by_stratum[stratum_key(shot)].append(shot)

    sizes = {key: len(pool) for key, pool in by_stratum.items()}
    quotas = allocate_by_sqrt(sizes, config.kis_sample_size)

    sampled: list[ShotRef] = []
    for key in sorted(by_stratum):
        pool = sorted(by_stratum[key], key=lambda s: (s.video_id, s.shot_id))
        quota = min(quotas.get(key, 0), len(pool))
        sampled.extend(rng.sample(pool, quota))
    return sampled


def sample_trake_sequences(
    shots_by_video: Mapping[str, list[ShotRef]],
    config: BuildEvalSetConfig,
    rng: random.Random,
) -> list[list[ShotRef]]:
    """Picks `config.trake_sequence_count` videos, each contributing one
    window of `config.trake_sequence_length` CONSECUTIVE shots (by shot_id,
    already contiguous -- TransNetV2 numbers shots 1..N per video). Unlike
    KIS sampling, no `min_shot_duration_sec` filter here: TRAKE needs real
    temporal contiguity, not a duration-cleaned subset."""
    eligible_videos = sorted(
        video_id
        for video_id, shots in shots_by_video.items()
        if len(shots) >= config.trake_sequence_length
    )
    chosen = sorted(
        rng.sample(eligible_videos, min(config.trake_sequence_count, len(eligible_videos)))
    )

    sequences: list[list[ShotRef]] = []
    for video_id in chosen:
        shots = shots_by_video[video_id]  # already sorted by shot_id (build_shot_refs)
        max_start = len(shots) - config.trake_sequence_length
        start = rng.randint(0, max_start)
        sequences.append(shots[start : start + config.trake_sequence_length])
    return sequences


def select_middle_keyframe(keyframe_docs: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """The shot's representative image for the Gemini prompt -- the sampled
    keyframe closest to the shot's temporal middle, found by sorting the
    shot's own keyframes (≤4, per the `k_extract` formula) by `frame_idx`
    and taking the middle element."""
    if not keyframe_docs:
        raise EvalSetBuildError("select_middle_keyframe: keyframe_docs is empty")
    ordered = sorted(keyframe_docs, key=lambda d: d["frame_idx"])
    return ordered[len(ordered) // 2]


# --------------------------------------------------------------------------
# Generic-caption filter (PRD §6.1.2 step 3 / §6.1.3) -- computable proxy for
# "matches thousands of shots", since no real search engine exists yet at P0.
# --------------------------------------------------------------------------


def content_words(caption: str, stopwords: frozenset[str]) -> frozenset[str]:
    tokens = _WORD_PATTERN.findall(caption.lower())
    return frozenset(token for token in tokens if token not in stopwords)


def flag_generic_captions(
    records: Sequence[Mapping[str, Any]], config: BuildEvalSetConfig
) -> list[str]:
    """One filter tag per record (same order/length as `records`):
    "kept" | "generic_short" | "generic_duplicate". A caption below
    `min_content_words` is flagged directly; otherwise a caption whose
    content-word Jaccard overlap with more than `max_near_duplicates` other
    captions in the same batch exceeds `near_duplicate_jaccard` is flagged
    -- the computable stand-in for "khớp > X shot khi tìm ngược"."""
    stopwords = frozenset(config.vietnamese_stopwords)
    word_sets = [content_words(record["caption"], stopwords) for record in records]

    tags: list[str] = []
    for i, words in enumerate(word_sets):
        if len(words) < config.min_content_words:
            tags.append("generic_short")
            continue
        near_duplicates = 0
        for j, other in enumerate(word_sets):
            if i == j or len(other) < config.min_content_words:
                continue
            union = words | other
            jaccard = len(words & other) / len(union) if union else 0.0
            if jaccard >= config.near_duplicate_jaccard:
                near_duplicates += 1
        tags.append("generic_duplicate" if near_duplicates > config.max_near_duplicates else "kept")
    return tags


# --------------------------------------------------------------------------
# TRAKE reconstruction -- reused later by the P2 DANTE/TRAKE eval harness,
# not called by this module's own CLI flow. No separate TRAKE file is
# needed: sampled TRAKE shots already carry contiguous `order_index` values
# within their `video_id` in the frozen GroundTruthSegment list.
# --------------------------------------------------------------------------


def extract_trake_sequences(
    segments: Sequence[GroundTruthSegment], n: int
) -> list[list[GroundTruthSegment]]:
    by_video: dict[str, list[GroundTruthSegment]] = defaultdict(list)
    for segment in segments:
        by_video[segment.video_id].append(segment)

    sequences: list[list[GroundTruthSegment]] = []
    for video_id in sorted(by_video):
        ordered = sorted(by_video[video_id], key=lambda s: s.order_index)
        for start in range(len(ordered) - n + 1):
            window = ordered[start : start + n]
            first_index = window[0].order_index
            if [s.order_index for s in window] == list(range(first_index, first_index + n)):
                sequences.append(window)
    return sequences


# --------------------------------------------------------------------------
# Freeze -- candidates (with human "approved" decisions) -> frozen artifact.
# --------------------------------------------------------------------------


def freeze_records(approved: Sequence[Mapping[str, Any]]) -> tuple[str, str]:
    """Keeps exactly the `GroundTruthSegment` fields (`datasets/base.py`) so
    `AIC2026Adapter.load_ground_truth()` reads the result unmodified. Hash
    convention mirrors `scripts/run_ingest.py`'s `_write_pilot_available_list`
    (bare hex digest, no filename, no JSON wrapper)."""
    minimal = [{field: record[field] for field in _GROUND_TRUTH_FIELDS} for record in approved]
    json_text = json.dumps(minimal, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    digest = hashlib.sha256(json_text.encode("utf-8")).hexdigest()
    return json_text, digest


def compute_quality_report(candidates: Sequence[Mapping[str, Any]]) -> QualityReport:
    """PRD §6.1.3's required label-quality metrics. Raises `EvalSetBuildError`
    (not a silent skip) if any candidate is still unreviewed -- "thiếu dữ
    liệu phải fail rõ ràng" (CLAUDE.md §7)."""
    total = len(candidates)
    unreviewed = [c for c in candidates if c.get("approved") is None]
    if unreviewed:
        sample_ids = [_shot_id_key(c["video_id"], c["shot_id"]) for c in unreviewed[:5]]
        raise EvalSetBuildError(
            f"{len(unreviewed)}/{total} candidate(s) still unreviewed (approved=null), "
            f'e.g. {sample_ids} -- review every record\'s "approved" field before freezing'
        )

    approved = [c for c in candidates if c["approved"] is True]
    rejected = total - len(approved)
    generic_filtered = [c for c in candidates if c.get("auto_filter") != "kept"]
    return QualityReport(
        total_candidates=total,
        approved_count=len(approved),
        rejection_rate=rejected / total if total else 0.0,
        generic_filter_rate=len(generic_filtered) / total if total else 0.0,
        # v1 uses a single reviewer (chủ dự án, 2026-08-09) -- the PRD §6.1.3
        # 2-reviewer overlap metric genuinely cannot be computed, not omitted.
        inter_reviewer_agreement=None,
    )


def _render_quality_report(quality: QualityReport, candidates_doc: Mapping[str, Any]) -> str:
    records = candidates_doc["records"]
    unique_videos = len({r["video_id"] for r in records if r["approved"] is True})
    agreement = (
        f"{quality.inter_reviewer_agreement:.1%}"
        if quality.inter_reviewer_agreement is not None
        else "not_measured (chỉ 1 reviewer ở v1 -- PRD §6.1.3)"
    )
    return (
        "# aic_eval_v1 quality report\n\n"
        f"- eval_set_version: {candidates_doc['eval_set_version']}\n"
        f"- git_sha: {candidates_doc['git_sha']}\n"
        f"- generated_at: {candidates_doc['generated_at']}\n"
        f"- seed: {candidates_doc['config']['seed']}\n"
        f"- total candidates: {quality.total_candidates}\n"
        f"- approved: {quality.approved_count}\n"
        f"- unique video_id (đã duyệt): {unique_videos}\n"
        f"- tỉ lệ loại khi review: {quality.rejection_rate:.1%}\n"
        f"- tỉ lệ bị lọc chung chung (auto_filter): {quality.generic_filter_rate:.1%}\n"
        f"- độ đồng thuận 2 reviewer: {agreement}\n"
    )


def _freeze_candidates(candidates_doc: Mapping[str, Any]) -> tuple[str, str, str]:
    records = candidates_doc["records"]
    quality = compute_quality_report(records)
    if quality.rejection_rate > _QUALITY_GATE_MAX_REJECTION_RATE:
        raise EvalSetBuildError(
            f"rejection rate {quality.rejection_rate:.1%} exceeds the "
            f"{_QUALITY_GATE_MAX_REJECTION_RATE:.0%} quality gate (PRD §6.1.3/R6) -- "
            "stop and report, do not freeze a bad label set"
        )
    approved = [r for r in records if r["approved"] is True]
    json_text, digest = freeze_records(approved)
    report_md = _render_quality_report(quality, candidates_doc)
    return json_text, digest, report_md


# --------------------------------------------------------------------------
# I/O boundary -- Protocol + Fake in tests (tests/test_ocr.py:50-67 pattern),
# real Gemini/Mongo implementations here.
# --------------------------------------------------------------------------


class CaptionGenerator(Protocol):
    def generate(self, image: Image.Image, title: str) -> str: ...


class KeyframeSource(Protocol):
    async def get_keyframes_for_video(self, video_id: str) -> list[dict[str, Any]]: ...


class CaptionGeneration(BaseModel):
    caption: str = Field(
        description=(
            "One Vietnamese sentence describing what is visually shown in this "
            "news-broadcast keyframe (people, objects, actions, setting). Do NOT "
            "transcribe on-screen text, subtitles, banners, or running captions "
            "verbatim, and do not state anything not visible in the frame."
        )
    )


class GeminiCaptionGenerator:
    """Wraps `client.interactions.create(...)` -- the current Gemini
    Interactions API (references.md §5.1), never `generate_content`. Mirrors
    `pipelines/ocr.py:GeminiOcrClient` exactly (retry policy, lazy import,
    structured-output schema pattern)."""

    def __init__(self, api_key: str, config: BuildEvalSetConfig = _DEFAULT_CONFIG) -> None:
        from google import genai  # lazy: only needed when this class is used

        self._client = genai.Client(api_key=api_key)
        self._config = config

    def generate(self, image: Image.Image, title: str) -> str:
        resized = image.copy()
        resized.thumbnail((self._config.max_image_side, self._config.max_image_side))
        buffer = io.BytesIO()
        resized.convert("RGB").save(buffer, format="JPEG")
        image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        last_exc: Exception | None = None
        for attempt in range(self._config.max_retries):
            try:
                return self._call_once(image_b64, title)
            except (TimeoutError, ConnectionError, OSError) as exc:
                last_exc = exc
                time.sleep(self._config.retry_base_delay_seconds * (2**attempt))
        raise RuntimeError(
            f"Gemini caption generation failed after {self._config.max_retries} attempts"
        ) from last_exc

    def _call_once(self, image_b64: str, title: str) -> str:
        prompt = (
            "Đây là một keyframe trích từ video tin tức tiếng Việt"
            + (f' có tiêu đề "{title}"' if title else "")
            + ". Hãy mô tả bằng MỘT câu tiếng Việt những gì thấy được trong ảnh "
            "(người, vật, hành động, bối cảnh). KHÔNG chép lại chữ hiển thị trên "
            "màn hình (phụ đề, banner, số liệu chạy chữ) -- chỉ mô tả nội dung "
            "hình ảnh nhìn thấy được."
        )
        interaction = self._client.interactions.create(
            model=self._config.gemini_model,
            input=[
                {"type": "text", "text": prompt},
                {"type": "image", "data": image_b64, "mime_type": "image/jpeg"},
            ],
            response_format={
                "type": "text",
                "mime_type": "application/json",
                "schema": CaptionGeneration.model_json_schema(),
            },
        )
        return CaptionGeneration.model_validate_json(interaction.output_text).caption


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


async def _generate_candidates(
    config: BuildEvalSetConfig,
    *,
    shots_by_video: Mapping[str, list[ShotRef]],
    video_meta: Mapping[str, VideoMeta],
    caption_generator: CaptionGenerator,
    keyframe_source: KeyframeSource,
    limit_shots: int = 0,
    existing_records: Sequence[Mapping[str, Any]] = (),
    on_progress: Callable[[list[dict[str, Any]], list[str], list[str]], None] | None = None,
) -> dict[str, Any]:
    """`existing_records` (loaded from a prior, interrupted `generate` run)
    are kept as-is; their shots are NOT re-sent to Gemini -- resuming after a
    quota/API error must not re-pay for captions already generated. Sampling
    itself stays over the FULL universe regardless of `existing_records` (same
    seed -> same ~640-shot selection every time), only the *generation* step
    for already-done shots is skipped.

    `on_progress`, called after each video's shots finish, lets the caller
    checkpoint to disk -- Gemini quota/API errors (`google.genai.errors.
    ClientError` and friends) are NOT subclasses of `OSError`/`RuntimeError`/
    `ValueError` (verified against the SDK source, 2026-08-09), so they are
    deliberately NOT caught here and will propagate out of this function;
    without a checkpoint, that would silently discard every caption paid for
    so far, not just the one that failed."""
    rng = random.Random(config.seed)
    kis_shots = sample_kis_shots(shots_by_video, config, rng)
    trake_sequences = sample_trake_sequences(shots_by_video, config, rng)

    tasks: list[tuple[ShotRef, str]] = [(shot, "kis") for shot in kis_shots]
    for sequence in trake_sequences:
        tasks.extend((shot, "trake") for shot in sequence)
    if limit_shots > 0:
        tasks = tasks[:limit_shots]

    already_done = {(r["video_id"], r["shot_id"]) for r in existing_records}
    if already_done:
        logger.info(
            "Resuming: %d shot(s) already have a caption from a prior run -- not re-billed",
            len(already_done),
        )
    tasks = [
        (shot, kind) for shot, kind in tasks if (shot.video_id, shot.shot_id) not in already_done
    ]

    by_video_tasks: dict[str, list[tuple[ShotRef, str]]] = defaultdict(list)
    for shot, task_kind in tasks:
        by_video_tasks[shot.video_id].append((shot, task_kind))

    records: list[dict[str, Any]] = list(existing_records)
    skipped_no_keyframe: list[str] = []
    failed: list[str] = []

    for video_id in sorted(by_video_tasks):
        keyframe_docs = await keyframe_source.get_keyframes_for_video(video_id)
        by_shot_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for doc in keyframe_docs:
            by_shot_id[doc["shot_id"]].append(doc)

        title = video_meta[video_id].title
        for shot, task_kind in by_video_tasks[video_id]:
            label = _shot_id_key(shot.video_id, shot.shot_id)
            shot_keyframes = by_shot_id.get(label, [])
            if not shot_keyframes:
                skipped_no_keyframe.append(label)
                logger.warning("%s: no keyframe for this shot -- skipping", label)
                continue

            keyframe = select_middle_keyframe(shot_keyframes)
            try:
                with Image.open(keyframe["image_path"]) as image:
                    caption = caption_generator.generate(image, title)
            except Exception as exc:  # noqa: BLE001 -- see rationale below
                # Deliberately broad, not "except Exception: pass" (CLAUDE.md
                # §7 still applies: logged with the failing shot's id, recorded
                # in `failed` -- visible in the output, not swallowed).
                # Verified 2026-08-09 against the installed google-genai SDK:
                # API-level failures (auth, quota, rate-limit, server errors)
                # come from TWO unrelated exception hierarchies in this SDK --
                # `google.genai.errors.{ClientError,ServerError}` (documented,
                # subclass of plain `Exception`) and, separately, what
                # `interactions.create()` actually raises in practice,
                # `google.genai._gaos.lib.compat_errors.RateLimitError` (also
                # `Exception`, NOT the same class, NOT overlapping with the
                # public `google.genai.errors` module at all). Neither is an
                # `OSError`/`RuntimeError`/`ValueError`/`TimeoutError`/
                # `ConnectionError`, and the SDK gives no stable narrower type
                # to catch. A batch of ~640 independent Gemini calls must not
                # let one bad/rate-limited call abort every caption already
                # paid for -- one shot failing is expected and recoverable
                # (re-run resumes it), the whole run crashing is not.
                logger.error("%s: caption generation failed: %s", label, exc)
                failed.append(label)
                if on_progress is not None:
                    on_progress(records, skipped_no_keyframe, failed)
                continue

            records.append(
                {
                    "video_id": shot.video_id,
                    "shot_id": shot.shot_id,
                    "start_sec": shot.start_sec,
                    "end_sec": shot.end_sec,
                    "order_index": shot.shot_id,
                    "caption": caption,
                    "task": task_kind,
                    "keyframe_id": keyframe["keyframe_id"],
                    "image_path": keyframe["image_path"],
                    "batch": shot.batch,
                    "author": shot.author,
                    "auto_filter": "kept",
                    "approved": None,
                }
            )
            if on_progress is not None:
                on_progress(records, skipped_no_keyframe, failed)

    for record, tag in zip(records, flag_generic_captions(records, config), strict=True):
        record["auto_filter"] = tag

    logger.info(
        "generate: %d candidate(s) (%d kis, %d trake), %d skipped (no keyframe), %d failed",
        len(records),
        sum(1 for r in records if r["task"] == "kis"),
        sum(1 for r in records if r["task"] == "trake"),
        len(skipped_no_keyframe),
        len(failed),
    )

    return {
        "eval_set_version": "v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "config": asdict(config),
        "records": records,
        # Durable visibility (not just a transient log line) that a shot was
        # dropped, and why -- CLAUDE.md §7 "thiếu dữ liệu phải fail rõ ràng,
        # không trả rỗng" applies to partial gaps too, not just hard errors.
        "skipped_no_keyframe": skipped_no_keyframe,
        "failed": failed,
    }


def _git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, OSError, subprocess.TimeoutExpired):
        return "unknown"


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_config(path: Path) -> BuildEvalSetConfig:
    raw = _load_yaml(path)
    raw["vietnamese_stopwords"] = tuple(raw.get("vietnamese_stopwords", ()))
    return BuildEvalSetConfig(**raw)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser(
        "generate", help="Sample shots + call Gemini -> candidates JSON for manual review"
    )
    generate.add_argument("--config", type=Path, default=_DEFAULT_CONFIG_PATH)
    generate.add_argument("--dataset-root", type=Path, default=_REPO_ROOT / "data")
    generate.add_argument("--shots-path", type=Path, default=_DEFAULT_SHOTS_PATH)
    generate.add_argument("--out", type=Path, default=_DEFAULT_CANDIDATES_PATH)
    generate.add_argument(
        "--limit-shots", type=int, default=0, help="0 = no limit; >0 to eyeball a few first"
    )

    freeze = subparsers.add_parser(
        "freeze", help="Read reviewed candidates -> freeze aic_eval_v1.json + hash"
    )
    freeze.add_argument("--candidates", type=Path, default=_DEFAULT_CANDIDATES_PATH)
    freeze.add_argument("--out", type=Path, default=_DEFAULT_FROZEN_PATH)

    return parser.parse_args()


def _write_candidates_doc(
    out_path: Path,
    config: BuildEvalSetConfig,
    records: list[dict[str, Any]],
    skipped_no_keyframe: list[str],
    failed: list[str],
) -> None:
    """Shared by the mid-run checkpoint and the final write -- same shape
    either way, so a checkpoint file left behind by a crash is itself a
    valid input to resume from (`_run_generate` re-reads `out_path`)."""
    doc = {
        "eval_set_version": "v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "config": asdict(config),
        "records": records,
        "skipped_no_keyframe": skipped_no_keyframe,
        "failed": failed,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")


async def _run_generate(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    settings = Settings()

    adapter = AIC2026Adapter(
        dataset_root=args.dataset_root, eval_set_path=args.out.with_name("aic_eval_v1.json")
    )
    video_meta = build_video_meta(adapter, config.split)
    shots_raw = load_shots_json(args.shots_path)
    shots_by_video = build_shot_refs(shots_raw, video_meta)

    existing_records: list[dict[str, Any]] = []
    if args.out.exists():
        existing_records = json.loads(args.out.read_text(encoding="utf-8"))["records"]
        logger.info(
            "%s already has %d record(s) -- resuming, not re-billing Gemini for them",
            args.out,
            len(existing_records),
        )

    mongo = MongoStore(uri=settings.mongo_uri)
    caption_generator = GeminiCaptionGenerator(
        api_key=settings.gemini_api_key.get_secret_value(), config=config
    )

    def _checkpoint(records: list[dict[str, Any]], skipped: list[str], failed: list[str]) -> None:
        _write_candidates_doc(args.out, config, records, skipped, failed)

    try:
        candidates_doc = await _generate_candidates(
            config,
            shots_by_video=shots_by_video,
            video_meta=video_meta,
            caption_generator=caption_generator,
            keyframe_source=mongo,
            limit_shots=args.limit_shots,
            existing_records=existing_records,
            on_progress=_checkpoint,
        )
    finally:
        await mongo.close()

    _write_candidates_doc(
        args.out,
        config,
        candidates_doc["records"],
        candidates_doc["skipped_no_keyframe"],
        candidates_doc["failed"],
    )
    logger.info("Wrote %d candidate(s) to %s", len(candidates_doc["records"]), args.out)
    return 0


def _run_freeze(args: argparse.Namespace) -> int:
    candidates_doc = json.loads(args.candidates.read_text(encoding="utf-8"))
    try:
        json_text, digest, report_md = _freeze_candidates(candidates_doc)
    except EvalSetBuildError as exc:
        logger.error("freeze refused: %s", exc)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json_text, encoding="utf-8")
    args.out.with_name(args.out.name + ".sha256").write_text(digest + "\n", encoding="utf-8")
    args.out.with_name(args.out.stem + ".quality_report.md").write_text(report_md, encoding="utf-8")
    logger.info("Froze %s (sha256=%s)", args.out, digest)
    return 0


async def _main() -> int:
    args = _parse_args()
    if args.command == "generate":
        return await _run_generate(args)
    return _run_freeze(args)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    sys.exit(asyncio.run(_main()))


if __name__ == "__main__":
    main()
