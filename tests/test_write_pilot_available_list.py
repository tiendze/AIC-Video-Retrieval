"""RED reproducer for pipelines/write_pilot_available_list.py (2026-08-10
follow-up to plan §P0.9) -- regenerates configs/eval/pilot_available_v1.txt
from MongoDB's live ingest_status instead of trusting the file's stale
prior contents (only 29/873 video_ids: the Kaggle-offline-embed import path
writes ingest_status straight to MongoDB and never touches this file).
"""

from __future__ import annotations

import hashlib

from pipelines.write_pilot_available_list import (
    AVAILABLE_STATUSES,
    available_video_ids,
    render_list_file,
    write_pilot_available_file,
)


def test_available_statuses_matches_embedded_ocr_done_and_complete():
    assert AVAILABLE_STATUSES == {"embedded", "ocr_done", "complete"}


def test_available_video_ids_excludes_pending_decoding_and_failed():
    docs = [
        {"video_id": "a", "ingest_status": "pending"},
        {"video_id": "b", "ingest_status": "decoding"},
        {"video_id": "c", "ingest_status": "failed"},
        {"video_id": "d", "ingest_status": "embedded"},
        {"video_id": "e", "ingest_status": "ocr_done"},
        {"video_id": "f", "ingest_status": "complete"},
    ]

    assert available_video_ids(docs) == ["d", "e", "f"]


def test_available_video_ids_deduplicates_and_sorts():
    docs = [
        {"video_id": "L21_V002", "ingest_status": "complete"},
        {"video_id": "L21_V001", "ingest_status": "complete"},
        {"video_id": "L21_V001", "ingest_status": "complete"},  # duplicate doc
    ]

    assert available_video_ids(docs) == ["L21_V001", "L21_V002"]


def test_available_video_ids_treats_missing_status_as_unavailable():
    docs = [{"video_id": "a"}]

    assert available_video_ids(docs) == []


def test_render_list_file_joins_with_trailing_newline():
    assert render_list_file(["a", "b"]) == "a\nb\n"


def test_render_list_file_returns_empty_string_for_no_ids():
    assert render_list_file([]) == ""


def test_write_pilot_available_file_writes_content_and_sha256_sidecar(tmp_path):
    out_path = tmp_path / "pilot_available_v1.txt"

    digest = write_pilot_available_file(["L21_V001", "L21_V002"], out_path)

    content = out_path.read_text(encoding="utf-8")
    assert content == "L21_V001\nL21_V002\n"
    assert digest == hashlib.sha256(content.encode("utf-8")).hexdigest()
    sidecar = out_path.with_name(out_path.name + ".sha256").read_text(encoding="utf-8")
    assert sidecar == digest + "\n"
