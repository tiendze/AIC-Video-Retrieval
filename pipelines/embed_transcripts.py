"""Batch-encodes ASR transcript segments and writes vectors to Milvus (P0.11b).

Thin glue between `Bgem3Encoder` and `TranscriptMilvusStore` — mirrors
`pipelines/embed.py`'s shape exactly, swapping `encode_image` for
`encode_text` and keyframe fields for the segment's frame *window*.
Callers read segments back from `AsrEsStore.scroll_all()` (P0.11a's already-
indexed transcripts) rather than re-running Whisper.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from app.stores.milvus_client import TranscriptMilvusStore, TranscriptVector


@dataclass(frozen=True)
class TranscriptToEmbed:
    segment_id: str
    text: str
    video_id: str
    start_frame_idx: int
    end_frame_idx: int
    start_time: float
    end_time: float


def embed_and_write_transcripts(
    segments: Sequence[TranscriptToEmbed],
    encoder,
    milvus: TranscriptMilvusStore,
) -> None:
    """Encodes every segment's text in one batch, then upserts all rows in one call."""
    if not segments:
        return
    vectors = encoder.encode_text([s.text for s in segments])
    rows = [
        TranscriptVector(
            segment_id=s.segment_id,
            embedding=vectors[i],
            video_id=s.video_id,
            start_frame_idx=s.start_frame_idx,
            end_frame_idx=s.end_frame_idx,
            start_time=s.start_time,
            end_time=s.end_time,
        )
        for i, s in enumerate(segments)
    ]
    milvus.upsert(rows)
