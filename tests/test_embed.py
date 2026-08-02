"""RED reproducer for pipelines/embed.py (P0.7).

Uses fake encoder/store test doubles (same shape as MultimodalEncoder /
MilvusStore) — no real model or database needed to verify the wiring.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from pipelines.embed import KeyframeToEmbed, embed_and_write


class _FakeEncoder:
    embedding_dim = 4
    model_id = "fake_encoder"

    def __init__(self):
        self.encode_image_calls: list[list[Image.Image]] = []

    def encode_image(self, images):
        self.encode_image_calls.append(list(images))
        # deterministic per-call-index vector so tests can check row mapping
        return np.array([[i, i, i, i] for i in range(len(images))], dtype=np.float32)

    def encode_text(self, texts):
        raise NotImplementedError


class _FakeMilvusStore:
    def __init__(self):
        self.upsert_calls = []

    def upsert(self, vectors):
        self.upsert_calls.append(list(vectors))


def _keyframe(i: int) -> KeyframeToEmbed:
    return KeyframeToEmbed(
        keyframe_id=f"video1_{i}",
        image=Image.new("RGB", (8, 8)),
        video_id="video1",
        global_index=i,
        frame_idx=i * 10,
        timestamp=i * 0.4,
        author="channel1",
        batch="L21",
    )


def test_embed_and_write_encodes_all_images_in_one_batch():
    encoder = _FakeEncoder()
    milvus = _FakeMilvusStore()
    keyframes = [_keyframe(0), _keyframe(1), _keyframe(2)]

    embed_and_write(keyframes, encoder, milvus)

    assert len(encoder.encode_image_calls) == 1
    assert len(encoder.encode_image_calls[0]) == 3


def test_embed_and_write_upserts_one_row_per_keyframe_with_matching_vector():
    encoder = _FakeEncoder()
    milvus = _FakeMilvusStore()
    keyframes = [_keyframe(0), _keyframe(1)]

    embed_and_write(keyframes, encoder, milvus)

    assert len(milvus.upsert_calls) == 1
    rows = milvus.upsert_calls[0]
    assert len(rows) == 2
    for i, row in enumerate(rows):
        assert row.keyframe_id == keyframes[i].keyframe_id
        assert row.frame_idx == keyframes[i].frame_idx
        assert row.global_index == keyframes[i].global_index
        np.testing.assert_array_equal(row.embedding, np.array([i, i, i, i], dtype=np.float32))


def test_embed_and_write_with_empty_input_does_not_call_encoder_or_store():
    encoder = _FakeEncoder()
    milvus = _FakeMilvusStore()

    embed_and_write([], encoder, milvus)

    assert encoder.encode_image_calls == []
    assert milvus.upsert_calls == []
