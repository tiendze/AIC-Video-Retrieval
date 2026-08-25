"""Imports a Kaggle-produced embed_out artifact into Milvus + MongoDB —
giai đoạn 2 of `.claude/plans/kaggle-offline-embed.plan.md`.

Kaggle already did the GPU work (BEiT-3 encode over 873 videos / 221,903
keyframes) and shipped `data/embed_out/vectors/<video_id>.{npy,jsonl}`: an
(n_kf, embedding_dim) L2-normalized matrix plus row-aligned
`{keyframe_id, video_id, frame_idx}` metadata, sorted by frame_idx ascending
within each video (plan §3 — "hợp đồng sắp xếp" lives in the file, not in
code run twice on two machines). Everything else CLAUDE.md's invariants
require — `shot_id`, `global_index`, `s_v`/`e_v`, `timestamp`, `fps`,
`author`/`batch`/`title`/... — is derived here from local-only sources:
`data/shots.json` (real shot boundaries, resolves plan §9.1's "shot_id mất
nghĩa" open item) and `DatasetAdapter.iter_videos()` (video metadata + the
raw .mp4 path needed to read `fps`).

`scripts/import_embed_out.py` is the CLI that wires this module to a live
Milvus/MongoDB and the parity check in `pipelines/verify_embed_parity.py`.
"""

from __future__ import annotations

import bisect
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.stores.milvus_client import KeyframeVector
from app.stores.mongo_client import KeyframeDocument, VideoDocument
from datasets.base import VideoRef
from pipelines.video_io import frame_idx_to_timestamp

STATUS_COMPLETE = "complete"


@dataclass(frozen=True)
class ShotRange:
    """One shot's absolute frame boundaries, from `data/shots.json`."""

    shot_id: int
    start_frame: int
    end_frame: int


@dataclass(frozen=True)
class ImportedKeyframe:
    """One keyframe row with every field Milvus/MongoDB need, joined from
    embed_out + shots.json + fps."""

    keyframe_id: str
    video_id: str
    shot_id: str
    frame_idx: int
    global_index: int
    timestamp: float
    embedding: np.ndarray
    author: str
    batch: str


def batch_of(video_id: str) -> str:
    return video_id.split("_V", 1)[0]


def is_video_already_imported(video_doc: dict | None) -> bool:
    """`True` when `video_doc` (a raw MongoDB `videos` find result, or `None`
    if no such video exists yet) already reached `STATUS_COMPLETE` — the
    caller's cue to skip re-importing it. Without this check, re-running the
    CLI over an already-imported video without `--reset` silently desyncs
    `global_index`/`s_v`/`e_v` between Milvus (upserts to the new value) and
    MongoDB `keyframes` (duplicate-key inserts are skipped, so the OLD value
    survives there) — CLAUDE.md T1: a re-run must not corrupt prior state."""
    return video_doc is not None and video_doc.get("ingest_status") == STATUS_COMPLETE


def discover_video_ids(vectors_dir: Path) -> list[str]:
    """Sorted `video_id` list from `vectors/*.npy` — this sort order is what
    decides `global_index` assignment order across the whole import run
    (plan §3)."""
    npy_ids = {p.stem for p in vectors_dir.glob("*.npy")}
    jsonl_ids = {p.stem for p in vectors_dir.glob("*.jsonl")}
    missing_jsonl = npy_ids - jsonl_ids
    missing_npy = jsonl_ids - npy_ids
    if missing_jsonl or missing_npy:
        raise ValueError(
            f"{vectors_dir}: mismatched .npy/.jsonl pairs — "
            f"missing .jsonl for {sorted(missing_jsonl)}, missing .npy for {sorted(missing_npy)}"
        )
    return sorted(npy_ids)


def align_metadata_to_video_ids(
    video_ids: Sequence[str], metadata_by_id: Mapping[str, dict]
) -> dict[str, dict]:
    """Aligns organiser metadata to local video ids.

    The current AIC metadata dump has a duplicate first record: `L21_V001`
    and `L21_V002` both describe the same YouTube video. The keyframe/vector
    artifacts are ordered correctly, so the duplicate shifts the metadata
    content by one row for the rest of the corpus. When that signature is
    present, map `video_ids[i]` to `metadata_ids[i + 1]` after the first row;
    otherwise keep the ordinary filename-based mapping.
    """
    sorted_video_ids = sorted(video_ids)
    sorted_metadata_ids = sorted(metadata_by_id)
    if len(sorted_video_ids) != len(sorted_metadata_ids) or not sorted_video_ids:
        return {
            video_id: metadata_by_id[video_id]
            for video_id in sorted_video_ids
            if video_id in metadata_by_id
        }

    first = metadata_by_id.get(sorted_metadata_ids[0])
    second = metadata_by_id.get(sorted_metadata_ids[1]) if len(sorted_metadata_ids) > 1 else None
    has_duplicate_first_record = (
        first is not None
        and second is not None
        and first.get("watch_url")
        and first.get("watch_url") == second.get("watch_url")
        and first.get("title") == second.get("title")
    )
    if not has_duplicate_first_record:
        return {
            video_id: metadata_by_id[video_id]
            for video_id in sorted_video_ids
            if video_id in metadata_by_id
        }

    aligned: dict[str, dict] = {}
    for index, video_id in enumerate(sorted_video_ids):
        if index == 0:
            metadata_id = sorted_metadata_ids[0]
        elif index + 1 < len(sorted_metadata_ids):
            metadata_id = sorted_metadata_ids[index + 1]
        else:
            metadata_id = video_id
        if metadata_id in metadata_by_id:
            aligned[video_id] = metadata_by_id[metadata_id]
    return aligned


def load_video_vectors(vectors_dir: Path, video_id: str) -> tuple[np.ndarray, list[dict]]:
    """Loads one video's (n_kf, dim) embedding matrix plus its row-aligned
    jsonl metadata. A row-count mismatch means a corrupted or partially
    written artifact — never silently truncate to the shorter side."""
    vectors = np.load(vectors_dir / f"{video_id}.npy")
    lines = (vectors_dir / f"{video_id}.jsonl").read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line]
    if len(rows) != vectors.shape[0]:
        raise ValueError(f"{video_id}: {vectors.shape[0]} vector rows but {len(rows)} jsonl lines")
    return vectors, rows


def load_shot_index(shots_json_path: Path) -> dict[str, list[ShotRange]]:
    """Reads `data/shots.json`'s real shot boundaries, keyed by `video_id`
    (`video_name` minus its `.mp4` suffix). Ranges come out sorted by
    `start_frame` for `resolve_shot_id`'s binary search."""
    with shots_json_path.open(encoding="utf-8") as f:
        data = json.load(f)
    index: dict[str, list[ShotRange]] = {}
    for video in data["videos"]:
        video_id = Path(video["video_name"]).stem
        ranges = sorted(
            (
                ShotRange(
                    shot_id=shot["shot_id"],
                    start_frame=shot["start_frame"],
                    end_frame=shot["end_frame"],
                )
                for shot in video["shots"]
            ),
            key=lambda r: r.start_frame,
        )
        index[video_id] = ranges
    return index


def resolve_shot_id(shot_ranges: Sequence[ShotRange], video_id: str, frame_idx: int) -> str:
    """Finds the shot whose `[start_frame, end_frame]` contains `frame_idx`
    via binary search (ranges are contiguous and sorted by `start_frame`).

    Raises rather than falling back to a placeholder on a miss: an
    out-of-range `frame_idx` means the embed_out extraction and the shot
    detection it's being joined against disagree, which is a real data
    problem, not something to paper over (CLAUDE.md §7)."""
    starts = [r.start_frame for r in shot_ranges]
    i = bisect.bisect_right(starts, frame_idx) - 1
    if i < 0 or not (shot_ranges[i].start_frame <= frame_idx <= shot_ranges[i].end_frame):
        raise ValueError(
            f"{video_id}: frame_idx={frame_idx} falls outside every shot range in data/shots.json"
        )
    return f"{video_id}_shot{shot_ranges[i].shot_id}"


def build_imported_keyframes(
    video_id: str,
    vectors: np.ndarray,
    jsonl_rows: Sequence[dict],
    shot_ranges: Sequence[ShotRange],
    fps: float,
    global_index_start: int,
    author: str,
) -> list[ImportedKeyframe]:
    """Pairs each embedding row with its jsonl metadata (same row order —
    the embed_out sorting contract) and assigns a contiguous `global_index`
    starting at `global_index_start`, in row order (== frame_idx ascending,
    so DANTE's `tau < t` ordering is preserved)."""
    batch = batch_of(video_id)
    records = []
    previous_frame_idx: int | None = None
    for i, row in enumerate(jsonl_rows):
        if row["video_id"] != video_id:
            raise ValueError(
                f"{video_id}: jsonl row {i} has video_id={row['video_id']!r}, expected {video_id!r}"
            )
        frame_idx = row["frame_idx"]
        if previous_frame_idx is not None and frame_idx <= previous_frame_idx:
            raise ValueError(
                f"{video_id}: frame_idx not ascending at jsonl row {i} "
                f"({frame_idx} after {previous_frame_idx}) — embed_out's row-order contract "
                "(plan §3) is violated, global_index would desync from frame_idx order"
            )
        previous_frame_idx = frame_idx
        records.append(
            ImportedKeyframe(
                keyframe_id=row["keyframe_id"],
                video_id=video_id,
                shot_id=resolve_shot_id(shot_ranges, video_id, frame_idx),
                frame_idx=frame_idx,
                global_index=global_index_start + i,
                timestamp=frame_idx_to_timestamp(frame_idx, fps),
                embedding=vectors[i],
                author=author,
                batch=batch,
            )
        )
    return records


def to_milvus_vector(keyframe: ImportedKeyframe) -> KeyframeVector:
    return KeyframeVector(
        keyframe_id=keyframe.keyframe_id,
        embedding=keyframe.embedding,
        video_id=keyframe.video_id,
        global_index=keyframe.global_index,
        frame_idx=keyframe.frame_idx,
        timestamp=keyframe.timestamp,
        author=keyframe.author,
        batch=keyframe.batch,
    )


def to_keyframe_document(keyframe: ImportedKeyframe, image_dir: Path) -> KeyframeDocument:
    """`image_path`/`thumbnail_path` point at the same file: the Kaggle
    keyframe set under `data/keyframe/` has no separate resized thumbnail,
    unlike `pipelines/ingest.py`'s `save_keyframe_images` which generates
    one for the TransNetV2 branch."""
    path = str(
        image_dir
        / keyframe.batch
        / keyframe.video_id
        / f"{keyframe.video_id}_{keyframe.frame_idx}.jpg"
    )
    return KeyframeDocument(
        keyframe_id=keyframe.keyframe_id,
        video_id=keyframe.video_id,
        shot_id=keyframe.shot_id,
        frame_idx=keyframe.frame_idx,
        timestamp=keyframe.timestamp,
        global_index=keyframe.global_index,
        image_path=path,
        thumbnail_path=path,
    )


@dataclass(frozen=True)
class VectorsOnlyKeyframe:
    """One keyframe row for `--encoder`-flagged (non-`beit3`) imports:
    metadata (`global_index`/`frame_idx`/`timestamp`) comes ENTIRELY from
    MongoDB, already written there by the primary BEiT-3 import — only
    `embedding` comes from the candidate encoder's own embed_out artifact.
    Never used to build a `KeyframeDocument`/`VideoDocument`; this row only
    ever becomes a `KeyframeVector` (Milvus), see
    `vectors_only_row_to_milvus_vector` below."""

    keyframe_id: str
    video_id: str
    global_index: int
    frame_idx: int
    timestamp: float
    embedding: np.ndarray
    author: str
    batch: str


def build_vectors_only_rows(
    video_id: str,
    vectors: np.ndarray,
    jsonl_rows: Sequence[dict],
    mongo_keyframe_docs: Sequence[dict],
    author: str,
) -> list[VectorsOnlyKeyframe]:
    """Joins a candidate encoder's embed_out vectors to the metadata MongoDB
    ALREADY has for this video (written by the primary `--encoder beit3`
    import) — vectors-only mode never re-derives `global_index`/`frame_idx`/
    `timestamp`, it reads them straight from Mongo so a second encoder's
    collection can never desync those values from what DANTE/search already
    rely on (encoder-swap plan §"Quyết định chiến lược #2").

    Raises ValueError if the artifact's `keyframe_id` set for this video does
    not EXACTLY match Mongo's — a partial/extra/missing set here means the
    candidate artifact came from a different keyframe source than what is
    already ingested, and importing it anyway would silently create a
    collection with holes or with `keyframe_id`s search can never resolve
    back to metadata (CLAUDE.md §3: "hỏng âm thầm").
    """
    mongo_by_id = {doc["keyframe_id"]: doc for doc in mongo_keyframe_docs}
    artifact_ids = {row["keyframe_id"] for row in jsonl_rows}
    mongo_ids = set(mongo_by_id)

    missing_in_mongo = artifact_ids - mongo_ids
    missing_in_artifact = mongo_ids - artifact_ids
    if missing_in_mongo or missing_in_artifact:
        raise ValueError(
            f"{video_id}: keyframe_id set mismatch between embed_out artifact and MongoDB -- "
            f"{len(missing_in_mongo)} in artifact but not Mongo "
            f"(e.g. {sorted(missing_in_mongo)[:5]}), "
            f"{len(missing_in_artifact)} in Mongo but not artifact "
            f"(e.g. {sorted(missing_in_artifact)[:5]})"
        )

    batch = batch_of(video_id)
    rows = []
    for i, row in enumerate(jsonl_rows):
        doc = mongo_by_id[row["keyframe_id"]]
        rows.append(
            VectorsOnlyKeyframe(
                keyframe_id=row["keyframe_id"],
                video_id=video_id,
                global_index=doc["global_index"],
                frame_idx=doc["frame_idx"],
                timestamp=doc["timestamp"],
                embedding=vectors[i],
                author=author,
                batch=batch,
            )
        )
    return rows


def vectors_only_row_to_milvus_vector(row: VectorsOnlyKeyframe) -> KeyframeVector:
    return KeyframeVector(
        keyframe_id=row.keyframe_id,
        embedding=row.embedding,
        video_id=row.video_id,
        global_index=row.global_index,
        frame_idx=row.frame_idx,
        timestamp=row.timestamp,
        author=row.author,
        batch=row.batch,
    )


def build_video_document(
    video: VideoRef, keyframes: Sequence[ImportedKeyframe], fps: float
) -> VideoDocument:
    global_indices = [k.global_index for k in keyframes]
    return VideoDocument(
        video_id=video.video_id,
        s_v=min(global_indices),
        e_v=max(global_indices),
        duration=int(video.duration_sec),
        fps=fps,
        num_keyframes=len(keyframes),
        ingest_status=STATUS_COMPLETE,
        author=video.category or "",
        batch=batch_of(video.video_id),
        title=video.title or "",
        publish_date=video.publish_date or "",
        watch_url=video.watch_url or "",
        channel_id=video.channel_id or "",
    )
