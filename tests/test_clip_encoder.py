"""RED reproducer for P1.4 — ClipEncoder must satisfy the MultimodalEncoder
contract (same mirror as tests/test_beit3_encoder.py).

Requires open_clip (see pyproject.toml `baseline-clip` extra) and downloads
the ViT-B/32 OpenAI checkpoint on first run (network required, then cached).
None of that exists on a bare host, so this whole module is skipped unless
open_clip is importable:

    pip install -e '.[baseline-clip]'
    pip install --no-deps open-clip-torch==3.3.0   # see pyproject.toml comment
    pytest -m model tests/test_clip_encoder.py -v
"""

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("torch")
pytest.importorskip("open_clip")

pytestmark = pytest.mark.model


@pytest.fixture(scope="module")
def encoder():
    import torch

    from app.encoders.clip_baseline import ClipEncoder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return ClipEncoder(device=device, batch_size=4)


@pytest.fixture()
def sample_images() -> list[Image.Image]:
    return [Image.new("RGB", (96, 64), color=(i * 40, 100, 200 - i * 30)) for i in range(3)]


@pytest.fixture()
def sample_texts() -> list[str]:
    return ["a man is speaking", "an evening news broadcast", "a fire truck"]


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

    image_norms = np.linalg.norm(image_vecs, axis=1)
    text_norms = np.linalg.norm(text_vecs, axis=1)

    np.testing.assert_allclose(image_norms, 1.0, atol=1e-4)
    np.testing.assert_allclose(text_norms, 1.0, atol=1e-4)


def test_encode_text_and_image_share_embedding_dim(encoder):
    assert encoder.embedding_dim == 512


def test_same_input_gives_same_vector(encoder, sample_images, sample_texts):
    first_image_vecs = encoder.encode_image(sample_images)
    second_image_vecs = encoder.encode_image(sample_images)
    np.testing.assert_allclose(first_image_vecs, second_image_vecs, atol=1e-5)

    first_text_vecs = encoder.encode_text(sample_texts)
    second_text_vecs = encoder.encode_text(sample_texts)
    np.testing.assert_allclose(first_text_vecs, second_text_vecs, atol=1e-5)


def test_encode_image_with_empty_input_returns_empty_array_not_error(encoder):
    vecs = encoder.encode_image([])

    assert vecs.shape == (0, encoder.embedding_dim)
    assert vecs.dtype == np.float32


def test_encode_text_with_empty_input_returns_empty_array_not_error(encoder):
    vecs = encoder.encode_text([])

    assert vecs.shape == (0, encoder.embedding_dim)
    assert vecs.dtype == np.float32
