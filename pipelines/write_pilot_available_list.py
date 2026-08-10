"""Regenerates `configs/eval/pilot_available_v1.txt` (+ `.sha256`) from
MongoDB's live `videos.ingest_status`, not from the file's own prior
contents (2026-08-10 follow-up to plan §P0.9).

Why this exists: `scripts/run_ingest.py`'s own `_write_pilot_available_list`
only ever wrote entries for videos IT processed in that invocation -- its
original local `--batches L21` pilot run, 29 `video_id`s. The
Kaggle-offline-embed pipeline (`scripts/import_embed_out.py` +
`scripts/import_ocr_results.py`, `.claude/plans/kaggle-offline-embed.plan.md`)
ingested the other ~844 videos of the 873-video corpus through a completely
different code path that writes `ingest_status` straight to MongoDB and
never touches this file -- so `pilot_available_v1.txt` has been stuck at 29
entries while the real corpus moved on. `eval/run.py` (P0.9) treats this
file as the authoritative denominator (CLAUDE.md §8: "số video_id duy nhất
thực tế đánh giá được"); a stale file silently undercounts every metric
report by ~97%.

`AVAILABLE_STATUSES` matches `scripts/run_ingest.py`'s own definition of
"available" ("video_ids that reached at least 'embedded'") -- enough for
the `semantic` channel; per-query `ocr` gaps stay visible in
`eval/run.py`'s `per_query.csv` rather than being hidden by exclusion from
the denominator.

`scripts/write_pilot_available_list.py` is the CLI that wires this to a
live MongoDB.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pipelines.ingest import STATUS_COMPLETE, STATUS_EMBEDDED, STATUS_OCR_DONE

AVAILABLE_STATUSES = frozenset({STATUS_EMBEDDED, STATUS_OCR_DONE, STATUS_COMPLETE})


def available_video_ids(video_docs: Sequence[dict[str, Any]]) -> list[str]:
    """Sorted, de-duplicated `video_id`s whose `ingest_status` is at least
    `embedded` -- mirrors `AIC2026Adapter`'s own dedup-by-id discipline
    (CLAUDE.md §3 "khử trùng video_id khi liệt kê video")."""
    return sorted(
        {doc["video_id"] for doc in video_docs if doc.get("ingest_status") in AVAILABLE_STATUSES}
    )


def render_list_file(video_ids: Sequence[str]) -> str:
    return "\n".join(video_ids) + ("\n" if video_ids else "")


def write_pilot_available_file(video_ids: Sequence[str], out_path: Path) -> str:
    """Writes `out_path` and its `.sha256` sidecar. Returns the digest.
    Replaces the file's content outright (no merge with what was there
    before) -- MongoDB's `ingest_status` is now the single source of truth
    for "available", so a stale prior file has nothing worth preserving."""
    content = render_list_file(video_ids)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    out_path.with_name(out_path.name + ".sha256").write_text(digest + "\n", encoding="utf-8")
    return digest
