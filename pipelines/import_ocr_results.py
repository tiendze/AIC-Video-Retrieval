"""Imports the team's already-run Gemini OCR output (`data/ocr_results.jsonl`)
into Elasticsearch — same TEXT-branch destination as `pipelines/ocr.py`, but
skipping the live Gemini call since the text was already extracted offline.

The file is one record per keyframe from the same Kaggle embed_out set
`pipelines/import_embed_out.py` imports (873 videos / 221,903 keyframes):
`{"folder", "video_id", "frame_id", "ocr_text", "raw_text"}`. `frame_id` here
is the same absolute frame index as MongoDB's `frame_idx` (CLAUDE.md T2), so
the join to `keyframe_id` goes through already-imported MongoDB keyframe
docs, not a locally reconstructed `f"{video_id}_f{frame_idx}"` string --
that keeps this module agnostic to whichever keyframe_id convention actually
produced those docs, and naturally reports (never silently drops) any
frame_id OCR has that Mongo doesn't yet, e.g. because that video hasn't been
imported via `scripts/import_embed_out.py` yet.

`scripts/import_ocr_results.py` is the CLI that wires this to a live
MongoDB + Elasticsearch.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from app.stores.es_client import OcrDocument


def load_ocr_results(path: Path) -> dict[str, list[dict[str, Any]]]:
    """Reads the OCR jsonl, grouping records by `video_id` in file order --
    lets the caller process one video at a time without holding a second
    per-video pass over the whole file."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            grouped[record["video_id"]].append(record)
    return dict(grouped)


def build_keyframe_index(mongo_keyframe_docs: Sequence[dict[str, Any]]) -> dict[int, tuple[str, float]]:
    """`frame_idx -> (keyframe_id, timestamp)` for one video's already-
    imported MongoDB keyframe docs (`MongoStore.get_keyframes_for_video`)."""
    return {doc["frame_idx"]: (doc["keyframe_id"], doc["timestamp"]) for doc in mongo_keyframe_docs}


def build_ocr_documents_for_video(
    video_id: str,
    records: Sequence[dict[str, Any]],
    keyframe_index: dict[int, tuple[str, float]],
) -> tuple[list[OcrDocument], list[int]]:
    """Joins this video's OCR records to `keyframe_index` by `frame_id`.
    Returns `(matched_documents, unmatched_frame_ids)` -- unmatched ids are
    the caller's evidence to log/report, never to swallow (CLAUDE.md §7)."""
    documents: list[OcrDocument] = []
    unmatched: list[int] = []
    for record in records:
        frame_id = record["frame_id"]
        match = keyframe_index.get(frame_id)
        if match is None:
            unmatched.append(frame_id)
            continue
        keyframe_id, timestamp = match
        documents.append(
            OcrDocument(
                keyframe_id=keyframe_id,
                ocr_text=record["ocr_text"],
                video_id=video_id,
                timestamp=timestamp,
            )
        )
    return documents, unmatched
