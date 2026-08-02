"""Ingest orchestrator (P0.7) — all three ingest branches for one video:

    VISION   shot detection -> keyframe sampling -> embedding -> Milvus/MongoDB
    TEXT/OCR keyframes -> pipelines/ocr.py (text filter -> phash dedupe ->
             Gemini OCR -> Elasticsearch), gated by a run-wide token budget
    METADATA folded into VideoDocument, written alongside the VISION branch

`ingest_status` walks `pending -> decoding -> embedded -> ocr_done ->
complete` (`failed` on any per-video exception) — architecture.md §4.3.
`run_ingest()` skips a video only once it has reached `ocr_done`/`complete`
(`_TERMINAL_STATUSES`); a video already at `embedded` from an earlier run
(VISION finished, OCR not yet attempted or budget-exhausted) is resumed
through `resume_ocr_for_video()`, which reads its keyframes back from
MongoDB instead of redoing shot detection/embedding — CLAUDE.md T1: a
killed-and-rerun ingest must not duplicate records and must not reprocess
finished work.

`global_index` stays contiguous across the whole run, and across resumed
runs, by recovering the next starting point from the highest `global_index`
already claimed by a written keyframe row (not from `VideoDocument.e_v`,
which is only written once a video's VISION branch fully finishes and so
would miss keyframes written just before a crash) rather than an in-memory
counter a crash would lose.

The Gemini OCR token budget (PRD risk R2, references.md §5.1) is a cap for
the *entire* `run_ingest()` call, not per video — `run_ingest` decrements a
running counter by each video's actual `tokens_used` and passes only the
remainder into the next video's OCR call, so a multi-video run can never
spend more than the configured cap in total. Once the remainder hits zero,
later videos simply stay at `embedded` (a legitimate, resumable state) until
a future run has budget again.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from PIL import Image

from app.encoders.base import MultimodalEncoder
from app.stores.es_client import EsStore
from app.stores.milvus_client import MilvusStore
from app.stores.mongo_client import KeyframeDocument, MongoStore, VideoDocument
from datasets.base import VideoRef
from pipelines.embed import KeyframeToEmbed, embed_and_write
from pipelines.keyframe import keyframe_frame_indices
from pipelines.ocr import OcrCandidate, OcrClient, OcrConfig, OcrRunSummary, run_ocr
from pipelines.shot_detect import Shot, ShotDetector
from pipelines.video_io import frame_idx_to_timestamp, iter_frames_rgb, read_video_info

logger = logging.getLogger(__name__)

STATUS_PENDING = "pending"
STATUS_DECODING = "decoding"
STATUS_EMBEDDED = "embedded"
STATUS_OCR_DONE = "ocr_done"
STATUS_COMPLETE = "complete"
STATUS_FAILED = "failed"

# Only these mean "nothing left to do for this video" on resume. STATUS_EMBEDDED
# is deliberately excluded: a video can legitimately sit there while waiting
# for OCR (budget exhausted, or the process was killed between the VISION and
# TEXT branches) — see resume_ocr_for_video().
_TERMINAL_STATUSES = frozenset({STATUS_OCR_DONE, STATUS_COMPLETE})

_THUMBNAIL_MAX_WIDTH = 320
_DEFAULT_OCR_CONFIG = OcrConfig()


@dataclass(frozen=True)
class ExtractedKeyframe:
    """One sampled keyframe with everything both Mongo and Milvus need."""

    keyframe_id: str
    video_id: str
    shot_id: str
    frame_idx: int
    global_index: int
    timestamp: float
    image: Image.Image
    author: str
    batch: str


@dataclass(frozen=True)
class IngestSummary:
    completed_video_ids: list[str]
    skipped_video_ids: list[str]  # already past _TERMINAL_STATUSES, not reprocessed
    failed_video_ids: list[str]
    ocr_tokens_used: int
    ocr_gemini_calls: int
    ocr_after_text_filter: int  # sum across videos — R2 cost evidence per filter step
    ocr_after_dedupe: int
    ocr_failed_keyframe_ids: list[str]


async def next_global_index_start(mongo: MongoStore) -> int:
    max_global_index = await mongo.max_keyframe_global_index()
    return 0 if max_global_index is None else max_global_index + 1


def extract_keyframe_images(
    video_path: str, target_frame_indices: Sequence[int]
) -> dict[int, Image.Image]:
    """A second sequential pass over the same decoder shot_detect used
    (pipelines/video_io — CLAUDE.md §3), picking out only the frames
    pipelines/keyframe.py selected."""
    wanted = set(target_frame_indices)
    images: dict[int, Image.Image] = {}
    for frame_idx, frame_rgb in iter_frames_rgb(video_path):
        if frame_idx in wanted:
            images[frame_idx] = Image.fromarray(frame_rgb)
        if len(images) == len(wanted):
            break
    missing = wanted - images.keys()
    if missing:
        raise ValueError(f"{video_path}: decoder never produced frame_idx {sorted(missing)}")
    return images


def _batch_of(video_id: str) -> str:
    """Derives the release batch (e.g. "L21") from a video_id like
    "L21_V001" — same convention as datasets/aic2026.py. A single
    `run_ingest` call processes the whole pilot split (L21-L24 together), so
    `batch` cannot be a constant passed in for the entire run; it must be
    read per video from the video_id itself."""
    return video_id.split("_V", 1)[0]


def build_extracted_keyframes(
    video: VideoRef,
    shots: Sequence[Shot],
    images_by_frame_idx: dict[int, Image.Image],
    fps: float,
    global_index_start: int,
) -> list[ExtractedKeyframe]:
    """Pairs each shot's sampled frames with a contiguous `global_index`, in
    shot order — this order is also what `s_v`/`e_v` are derived from."""
    records: list[ExtractedKeyframe] = []
    global_index = global_index_start
    batch = _batch_of(video.video_id)
    for shot_index, shot in enumerate(shots):
        shot_id = f"{video.video_id}_shot{shot_index}"
        for frame_idx in keyframe_frame_indices(shot):
            records.append(
                ExtractedKeyframe(
                    keyframe_id=f"{video.video_id}_{global_index}",
                    video_id=video.video_id,
                    shot_id=shot_id,
                    frame_idx=frame_idx,
                    global_index=global_index,
                    timestamp=frame_idx_to_timestamp(frame_idx, fps),
                    image=images_by_frame_idx[frame_idx],
                    author=video.category or "",
                    batch=batch,
                )
            )
            global_index += 1
    return records


def save_keyframe_images(keyframes: Sequence[ExtractedKeyframe], image_dir: Path) -> None:
    """Writes each keyframe's full image + a resized thumbnail to
    `image_dir/<video_id>/`. Paths are deterministic from `keyframe_id`, so
    `image_path`/`thumbnail_path` never need to be persisted separately."""
    for keyframe in keyframes:
        video_dir = image_dir / keyframe.video_id
        video_dir.mkdir(parents=True, exist_ok=True)
        keyframe.image.save(image_path_for(image_dir, keyframe))
        thumbnail = keyframe.image.copy()
        thumbnail.thumbnail((_THUMBNAIL_MAX_WIDTH, _THUMBNAIL_MAX_WIDTH))
        thumbnail.save(thumbnail_path_for(image_dir, keyframe))


def image_path_for(image_dir: Path, keyframe: ExtractedKeyframe) -> Path:
    return image_dir / keyframe.video_id / f"{keyframe.keyframe_id}.jpg"


def thumbnail_path_for(image_dir: Path, keyframe: ExtractedKeyframe) -> Path:
    return image_dir / keyframe.video_id / f"{keyframe.keyframe_id}_thumb.jpg"


def _pending_video_document(video: VideoRef, status: str) -> VideoDocument:
    """Placeholder fields (0/"") for stages before shot detection has run —
    overwritten with real values once known; `ingest_status` is what matters
    for resume, not these interim zeros."""
    return VideoDocument(
        video_id=video.video_id,
        s_v=0,
        e_v=0,
        duration=int(video.duration_sec),
        fps=0.0,
        num_keyframes=0,
        ingest_status=status,
        author=video.category or "",
        batch=_batch_of(video.video_id),
        title=video.title or "",
        publish_date=video.publish_date or "",
        watch_url=video.watch_url or "",
        channel_id=video.channel_id or "",
    )


def _finalized_video_document(
    video: VideoRef,
    status: str,
    keyframes: Sequence[ExtractedKeyframe],
    fps: float,
) -> VideoDocument:
    base = _pending_video_document(video, status)
    global_indices = [k.global_index for k in keyframes]
    return VideoDocument(
        **{
            **base.__dict__,
            "s_v": min(global_indices),
            "e_v": max(global_indices),
            "fps": fps,
            "num_keyframes": len(keyframes),
        }
    )


def _video_document_from_mongo(doc: dict[str, Any]) -> VideoDocument:
    """Reconstructs a `VideoDocument` from a raw MongoDB find result (which
    also carries Mongo's own `_id`) — used when resuming OCR for a video
    whose VISION branch was already finalized in an earlier run."""
    return VideoDocument(
        video_id=doc["video_id"],
        s_v=doc["s_v"],
        e_v=doc["e_v"],
        duration=doc["duration"],
        fps=doc["fps"],
        num_keyframes=doc["num_keyframes"],
        ingest_status=doc["ingest_status"],
        author=doc["author"],
        batch=doc["batch"],
        title=doc["title"],
        publish_date=doc["publish_date"],
        watch_url=doc["watch_url"],
        channel_id=doc["channel_id"],
    )


def _to_keyframe_document(keyframe: ExtractedKeyframe, image_dir: Path) -> KeyframeDocument:
    return KeyframeDocument(
        keyframe_id=keyframe.keyframe_id,
        video_id=keyframe.video_id,
        shot_id=keyframe.shot_id,
        frame_idx=keyframe.frame_idx,
        timestamp=keyframe.timestamp,
        global_index=keyframe.global_index,
        image_path=str(image_path_for(image_dir, keyframe)),
        thumbnail_path=str(thumbnail_path_for(image_dir, keyframe)),
    )


def _to_keyframe_to_embed(keyframe: ExtractedKeyframe) -> KeyframeToEmbed:
    return KeyframeToEmbed(
        keyframe_id=keyframe.keyframe_id,
        image=keyframe.image,
        video_id=keyframe.video_id,
        global_index=keyframe.global_index,
        frame_idx=keyframe.frame_idx,
        timestamp=keyframe.timestamp,
        author=keyframe.author,
        batch=keyframe.batch,
    )


def _ocr_candidates_from_extracted(keyframes: Sequence[ExtractedKeyframe]) -> list[OcrCandidate]:
    return [
        OcrCandidate(
            keyframe_id=k.keyframe_id,
            video_id=k.video_id,
            timestamp=k.timestamp,
            image=k.image,
        )
        for k in keyframes
    ]


def _ocr_candidates_from_mongo_docs(docs: Sequence[dict[str, Any]]) -> list[OcrCandidate]:
    return [
        OcrCandidate(
            keyframe_id=doc["keyframe_id"],
            video_id=doc["video_id"],
            timestamp=doc["timestamp"],
            image=Image.open(doc["image_path"]),
        )
        for doc in docs
    ]


async def _run_ocr_branch(
    video: VideoRef,
    embedded_doc: VideoDocument,
    candidates: Sequence[OcrCandidate],
    *,
    mongo: MongoStore,
    ocr_client: OcrClient,
    es: EsStore,
    ocr_token_budget: int,
    ocr_config: OcrConfig,
) -> OcrRunSummary | None:
    """Runs the TEXT branch for one video's keyframes and advances its
    status `embedded -> ocr_done -> complete`. Returns `None` (status stays
    `embedded`, unchanged) without calling Gemini at all when the run-wide
    token budget is already exhausted — CLAUDE.md §4: 'vượt trần thì dừng
    và báo, đừng đốt tiếp'."""
    if ocr_token_budget <= 0:
        logger.warning(
            "OCR token budget exhausted — video_id=%s stays at %s, resume on a later run",
            video.video_id,
            STATUS_EMBEDDED,
        )
        return None

    try:
        summary = await run_ocr(
            candidates,
            ocr_client=ocr_client,
            es=es,
            token_budget=ocr_token_budget,
            config=ocr_config,
        )
    except Exception as exc:
        # Deliberately broad, mirroring run_ingest's own outer boundary: the
        # google-genai SDK's exception hierarchy for quota/rate-limit/auth
        # failures isn't verified yet (GeminiOcrClient's docstring), and an
        # EsStore outage raises whatever the elasticsearch client raises.
        # Letting either propagate to run_ingest's per-video except clause
        # would overwrite this video's already-correct VISION VideoDocument
        # (s_v/e_v/num_keyframes) with a zeroed STATUS_FAILED placeholder,
        # and force a full shot-detect/embed redo next run instead of a cheap
        # OCR-only resume (CLAUDE.md §7: an OCR failure must not kill the
        # rest of the pipeline). Leaving the video at STATUS_EMBEDDED here
        # means resume_ocr_for_video() picks it back up untouched.
        logger.error(
            "OCR branch failed for video_id=%s — stays at %s: %s",
            video.video_id,
            STATUS_EMBEDDED,
            exc,
        )
        return None
    logger.info(
        "OCR summary video_id=%s: candidates=%d after_text_filter=%d after_dedupe=%d "
        "gemini_calls=%d tokens_used=%d stopped_on_budget=%s failed_keyframe_ids=%s",
        video.video_id,
        summary.total_candidates,
        summary.after_text_filter,
        summary.after_dedupe,
        summary.gemini_calls,
        summary.tokens_used,
        summary.stopped_on_budget,
        summary.failed_keyframe_ids,
    )
    await mongo.upsert_video(replace(embedded_doc, ingest_status=STATUS_OCR_DONE))
    await mongo.upsert_video(replace(embedded_doc, ingest_status=STATUS_COMPLETE))
    return summary


async def ingest_video(
    video: VideoRef,
    *,
    shot_detector: ShotDetector,
    encoder: MultimodalEncoder,
    mongo: MongoStore,
    milvus: MilvusStore,
    image_dir: Path,
    ocr_client: OcrClient,
    es: EsStore,
    ocr_token_budget: int,
    ocr_config: OcrConfig = _DEFAULT_OCR_CONFIG,
) -> OcrRunSummary | None:
    """Ingests one video end-to-end: VISION branch, then TEXT/OCR branch.
    Caller (`run_ingest`) is responsible for skipping videos already past
    `_TERMINAL_STATUSES`."""
    # Must run before the pending-status write below: once that placeholder
    # (e_v=0) lands in MongoDB, max_video_e_v() would see this video's own
    # not-yet-real e_v and could return a stale/wrong starting point.
    global_index_start = await next_global_index_start(mongo)
    await mongo.upsert_video(_pending_video_document(video, STATUS_DECODING))

    shots = shot_detector.detect_shots(video.source_path)
    video_info = read_video_info(video.source_path)

    target_frame_indices = {idx for shot in shots for idx in keyframe_frame_indices(shot)}
    images_by_frame_idx = extract_keyframe_images(video.source_path, sorted(target_frame_indices))
    keyframes = build_extracted_keyframes(
        video, shots, images_by_frame_idx, video_info.fps, global_index_start
    )

    save_keyframe_images(keyframes, image_dir)
    await mongo.bulk_insert_keyframes([_to_keyframe_document(k, image_dir) for k in keyframes])

    embed_and_write([_to_keyframe_to_embed(k) for k in keyframes], encoder, milvus)

    embedded_doc = _finalized_video_document(video, STATUS_EMBEDDED, keyframes, video_info.fps)
    await mongo.upsert_video(embedded_doc)

    return await _run_ocr_branch(
        video,
        embedded_doc,
        _ocr_candidates_from_extracted(keyframes),
        mongo=mongo,
        ocr_client=ocr_client,
        es=es,
        ocr_token_budget=ocr_token_budget,
        ocr_config=ocr_config,
    )


async def resume_ocr_for_video(
    video: VideoRef,
    video_doc: dict[str, Any],
    *,
    mongo: MongoStore,
    ocr_client: OcrClient,
    es: EsStore,
    ocr_token_budget: int,
    ocr_config: OcrConfig = _DEFAULT_OCR_CONFIG,
) -> OcrRunSummary | None:
    """For a video already at STATUS_EMBEDDED from an earlier run — reads its
    keyframes back from MongoDB (never redoes shot detection/embedding) and
    runs just the TEXT/OCR branch."""
    keyframe_docs = await mongo.get_keyframes_for_video(video.video_id)
    candidates = _ocr_candidates_from_mongo_docs(keyframe_docs)
    embedded_doc = _video_document_from_mongo(video_doc)

    return await _run_ocr_branch(
        video,
        embedded_doc,
        candidates,
        mongo=mongo,
        ocr_client=ocr_client,
        es=es,
        ocr_token_budget=ocr_token_budget,
        ocr_config=ocr_config,
    )


async def run_ingest(
    videos: Iterable[VideoRef],
    *,
    shot_detector: ShotDetector,
    encoder: MultimodalEncoder,
    mongo: MongoStore,
    milvus: MilvusStore,
    image_dir: Path,
    ocr_client: OcrClient,
    es: EsStore,
    ocr_token_budget: int,
    ocr_config: OcrConfig = _DEFAULT_OCR_CONFIG,
) -> IngestSummary:
    """Iterates every video, skipping ones already past `_TERMINAL_STATUSES`.
    A failing video is recorded and does not stop the rest of the run
    (CLAUDE.md §4: 'video lỗi -> error list riêng, pipeline chạy tiếp').

    `ocr_token_budget` is a single pool shared across every video in this
    call, decremented by each video's actual spend — never reset per video."""
    completed: list[str] = []
    skipped: list[str] = []
    failed: list[str] = []
    tokens_remaining = ocr_token_budget
    ocr_gemini_calls = 0
    ocr_after_text_filter = 0
    ocr_after_dedupe = 0
    ocr_failed_keyframe_ids: list[str] = []

    for video in videos:
        existing = await mongo.get_video(video.video_id)
        status = existing.get("ingest_status") if existing is not None else None
        if status in _TERMINAL_STATUSES:
            skipped.append(video.video_id)
            continue

        try:
            if status == STATUS_EMBEDDED:
                ocr_summary = await resume_ocr_for_video(
                    video,
                    existing,
                    mongo=mongo,
                    ocr_client=ocr_client,
                    es=es,
                    ocr_token_budget=tokens_remaining,
                    ocr_config=ocr_config,
                )
            else:
                ocr_summary = await ingest_video(
                    video,
                    shot_detector=shot_detector,
                    encoder=encoder,
                    mongo=mongo,
                    milvus=milvus,
                    image_dir=image_dir,
                    ocr_client=ocr_client,
                    es=es,
                    ocr_token_budget=tokens_remaining,
                    ocr_config=ocr_config,
                )
            if ocr_summary is not None:
                tokens_remaining -= ocr_summary.tokens_used
                ocr_gemini_calls += ocr_summary.gemini_calls
                ocr_after_text_filter += ocr_summary.after_text_filter
                ocr_after_dedupe += ocr_summary.after_dedupe
                ocr_failed_keyframe_ids.extend(ocr_summary.failed_keyframe_ids)
            completed.append(video.video_id)
        except Exception as exc:
            # Deliberately broad: this is the one per-video error boundary in
            # a multi-hour batch run, and neither BEiT3Encoder nor MilvusStore
            # normalizes their failures down to OSError/ValueError (e.g. a
            # CUDA OOM surfaces as torch.cuda.OutOfMemoryError, a RuntimeError
            # subclass; pymilvus raises its own MilvusException hierarchy) —
            # narrowing this to specific types silently let those crash the
            # whole run instead of failing just the one video (CLAUDE.md §4:
            # "video lỗi -> error list riêng, pipeline chạy tiếp"). Every
            # branch below logs with video_id and records failure state,
            # nothing here is silently swallowed.
            logger.error("ingest failed for video_id=%s: %s", video.video_id, exc)
            await mongo.upsert_video(_pending_video_document(video, STATUS_FAILED))
            failed.append(video.video_id)

    return IngestSummary(
        completed_video_ids=completed,
        skipped_video_ids=skipped,
        failed_video_ids=failed,
        ocr_tokens_used=ocr_token_budget - tokens_remaining,
        ocr_gemini_calls=ocr_gemini_calls,
        ocr_after_text_filter=ocr_after_text_filter,
        ocr_after_dedupe=ocr_after_dedupe,
        ocr_failed_keyframe_ids=ocr_failed_keyframe_ids,
    )
