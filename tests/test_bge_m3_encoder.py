"""Real-model smoke test for Bgem3Encoder (P0.11b) -- downloads BAAI/bge-m3
from HuggingFace Hub on first run (~2GB). Skipped by default
(`addopts = "-m 'not integration and not model'"`), mirrors
tests/test_beit3_encoder.py's `pytest.mark.model` convention.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sentence_transformers")

from app.encoders.bge_m3 import Bgem3Encoder  # noqa: E402

pytestmark = pytest.mark.model


@pytest.fixture(scope="module")
def encoder() -> Bgem3Encoder:
    return Bgem3Encoder(device="cpu", batch_size=4)


def test_encode_text_returns_l2_normalized_float32(encoder: Bgem3Encoder):
    vectors = encoder.encode_text(["xin chào", "đồng bằng sông Cửu Long"])

    assert vectors.shape == (2, 1024)
    assert vectors.dtype == np.float32
    norms = (vectors**2).sum(axis=1) ** 0.5
    assert norms == pytest.approx([1.0, 1.0], abs=1e-4)


def test_encode_text_is_deterministic(encoder: Bgem3Encoder):
    first = encoder.encode_text(["tin tức chính"])
    second = encoder.encode_text(["tin tức chính"])

    assert (first == second).all()


def test_encode_text_empty_list_returns_empty_array(encoder: Bgem3Encoder):
    vectors = encoder.encode_text([])

    assert vectors.shape == (0, 1024)
