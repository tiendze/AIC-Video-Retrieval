"""RED reproducer for pipelines/embed_transcripts.py (P0.11b).

Uses fake encoder/store test doubles (same shape as Bgem3Encoder /
TranscriptMilvusStore) — no real model or database needed to verify the
wiring. Mirrors tests/test_embed.py's convention.
"""

from __future__ import annotations

import numpy as np

from pipelines.embed_transcripts import TranscriptToEmbed, embed_and_write_transcripts


class _FakeEncoder:
    embedding_dim = 4
    model_id = "fake_bge_m3"

    def __init__(self):
        self.encode_text_calls: list[list[str]] = []

    def encode_text(self, texts):
        self.encode_text_calls.append(list(texts))
        return np.array([[i, i, i, i] for i in range(len(texts))], dtype=np.float32)


class _FakeTranscriptMilvusStore:
    def __init__(self):
        self.upsert_calls = []

    def upsert(self, vectors):
        self.upsert_calls.append(list(vectors))


def _segment(i: int) -> TranscriptToEmbed:
    return TranscriptToEmbed(
        segment_id=f"video1_seg{i}",
        text=f"noi dung cau {i}",
        video_id="video1",
        start_frame_idx=i * 10,
        end_frame_idx=i * 10 + 5,
        start_time=i * 2.0,
        end_time=i * 2.0 + 1.5,
    )


def test_embed_and_write_transcripts_encodes_all_text_in_one_batch():
    encoder = _FakeEncoder()
    milvus = _FakeTranscriptMilvusStore()
    segments = [_segment(0), _segment(1), _segment(2)]

    embed_and_write_transcripts(segments, encoder, milvus)

    assert len(encoder.encode_text_calls) == 1
    assert encoder.encode_text_calls[0] == [s.text for s in segments]


def test_embed_and_write_transcripts_upserts_one_row_per_segment_with_matching_vector():
    encoder = _FakeEncoder()
    milvus = _FakeTranscriptMilvusStore()
    segments = [_segment(0), _segment(1)]

    embed_and_write_transcripts(segments, encoder, milvus)

    assert len(milvus.upsert_calls) == 1
    rows = milvus.upsert_calls[0]
    assert len(rows) == 2
    for i, row in enumerate(rows):
        assert row.segment_id == segments[i].segment_id
        assert row.video_id == segments[i].video_id
        assert row.start_frame_idx == segments[i].start_frame_idx
        assert row.end_frame_idx == segments[i].end_frame_idx
        assert row.start_time == segments[i].start_time
        assert row.end_time == segments[i].end_time
        np.testing.assert_array_equal(row.embedding, np.array([i, i, i, i], dtype=np.float32))


def test_embed_and_write_transcripts_with_empty_input_does_not_call_encoder_or_store():
    encoder = _FakeEncoder()
    milvus = _FakeTranscriptMilvusStore()

    embed_and_write_transcripts([], encoder, milvus)

    assert encoder.encode_text_calls == []
    assert milvus.upsert_calls == []
