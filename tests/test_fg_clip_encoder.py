"""FGClipEncoder model-contract tests.

Requires the local FG-CLIP checkpoint and optional dependencies:

    pip install -e ".[fg-clip]"
    pytest -m model tests/test_fg_clip_encoder.py -v
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("einops")

pytestmark = pytest.mark.model

_MODEL_PATH = Path(os.environ.get("FGCLIP_MODEL_PATH", "models/fgclip_base"))
_EXPECTED_DIM = 512

if not _MODEL_PATH.exists():
    pytest.skip(
        "FGCLIP_MODEL_PATH not set and models/fgclip_base is missing",
        allow_module_level=True,
    )


@pytest.fixture(scope="module")
def encoder():
    from app.encoders.fg_clip import FGClipEncoder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return FGClipEncoder(model_path=str(_MODEL_PATH), device=device, batch_size=2)


@pytest.fixture()
def sample_images() -> list[Image.Image]:
    return [Image.new("RGB", (96, 64), color=(40 + i * 50, 100, 180)) for i in range(2)]


@pytest.fixture()
def sample_texts() -> list[str]:
    return ["a pink blouse", "a black skirt"]


def test_encoder_returns_float32_array_of_shape_n_by_dim(encoder, sample_images, sample_texts):
    image_vecs = encoder.encode_image(sample_images)
    text_vecs = encoder.encode_text(sample_texts)

    assert image_vecs.shape == (len(sample_images), encoder.embedding_dim)
    assert image_vecs.dtype == np.float32
    assert text_vecs.shape == (len(sample_texts), encoder.embedding_dim)
    assert text_vecs.dtype == np.float32


def test_encoder_output_is_l2_normalized(encoder, sample_images, sample_texts):
    image_vecs = encoder.encode_image(sample_images)
    text_vecs = encoder.encode_text(sample_texts)

    np.testing.assert_allclose(np.linalg.norm(image_vecs, axis=1), 1.0, atol=1e-4)
    np.testing.assert_allclose(np.linalg.norm(text_vecs, axis=1), 1.0, atol=1e-4)


def test_encode_text_and_image_share_embedding_dim(encoder):
    assert encoder.embedding_dim == _EXPECTED_DIM


def test_dense_image_and_text_token_embeddings_are_normalized(encoder, sample_images):
    patches = encoder.encode_image_patches(sample_images)
    tokens = encoder.encode_text_tokens(["red seasoning packet"])

    assert patches.shape[0] == len(sample_images)
    assert patches.shape[1] > 0
    assert patches.shape[2] == encoder.embedding_dim
    assert patches.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(patches, axis=2), 1.0, atol=1e-4)

    assert len(tokens) == 1
    assert tokens[0].embeddings.shape[0] == len(tokens[0].tokens)
    assert tokens[0].embeddings.shape[0] > 0
    assert tokens[0].embeddings.shape[1] == encoder.embedding_dim
    assert tokens[0].embeddings.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(tokens[0].embeddings, axis=1), 1.0, atol=1e-4)


def test_text_is_padded_to_fgclip_short_caption_length(encoder, sample_texts):
    token_ids = encoder.debug_tokenize(sample_texts)

    assert token_ids.shape[1] == 77


def test_max_text_tokens_matches_short_caption_length(encoder):
    assert encoder.max_text_tokens == 77


def test_count_text_tokens_reports_real_length_not_padded_length(encoder, sample_texts):
    counts = encoder.count_text_tokens(sample_texts)

    assert len(counts) == len(sample_texts)
    assert all(count > 0 for count in counts)
    # Unlike debug_tokenize()/encode_text(), this must NOT be padded to 77 --
    # short sample texts stay short, which is what lets
    # app/services/trake_service.py:_encode_events compare a real length
    # against max_text_tokens instead of always seeing 77.
    assert all(count < 77 for count in counts)


def test_count_text_tokens_flags_a_text_over_the_cap(encoder):
    long_text = " ".join(["a distinct visual detail"] * 40)

    counts = encoder.count_text_tokens([long_text])

    assert counts[0] > encoder.max_text_tokens
