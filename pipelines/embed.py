"""Batch-encodes keyframe images and writes vectors to Milvus (P0.7).

Thin glue between `MultimodalEncoder` (P0.5) and `MilvusStore` (P0.6) — the
encoder already L2-normalizes and the store already keys collections by
`model_id`, so this module only has to keep `frame_idx` traveling with every
row (CLAUDE.md §3: never dropped, never recomputed later).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from PIL import Image

from app.encoders.base import MultimodalEncoder
from app.stores.milvus_client import KeyframeVector, MilvusStore


@dataclass(frozen=True)
class KeyframeToEmbed:
    keyframe_id: str
    image: Image.Image
    video_id: str
    global_index: int
    frame_idx: int
    timestamp: float
    author: str
    batch: str


def embed_and_write(
    keyframes: Sequence[KeyframeToEmbed],
    encoder: MultimodalEncoder,
    milvus: MilvusStore,
) -> None:
    """Encodes every image in one batch, then upserts all rows in one call."""
    if not keyframes:
        return
    vectors = encoder.encode_image([k.image for k in keyframes])
    rows = [
        KeyframeVector(
            keyframe_id=k.keyframe_id,
            embedding=vectors[i],
            video_id=k.video_id,
            global_index=k.global_index,
            frame_idx=k.frame_idx,
            timestamp=k.timestamp,
            author=k.author,
            batch=k.batch,
        )
        for i, k in enumerate(keyframes)
    ]
    milvus.upsert(rows)
