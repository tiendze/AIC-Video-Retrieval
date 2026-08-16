"""RED reproducer for the encoder swap (plan §T3) -- PeCoreEncoder must
satisfy the MultimodalEncoder contract (same mirror as
tests/test_clip_encoder.py / tests/test_siglip2_encoder.py).

Requires open_clip + timm>=1.0.19 (see pyproject.toml `encoder-candidates`
extra, PE-Core addition 2026-08-14) and downloads the bigG-14-448 checkpoint
on first run (network required, then cached) -- this is the heaviest
candidate in the encoder-swap plan (~1.8B vision params), so this module
SKIPS entirely unless open_clip is importable:

    pip install -e '.[encoder-candidates]'
    pytest -m model tests/test_pe_core_encoder.py -v
"""

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("torch")
pytest.importorskip("open_clip")

pytestmark = pytest.mark.model

_EXPECTED_DIM = 1280  # verified 2026-08-14 from open_clip_config.json model_cfg.embed_dim


@pytest.fixture(scope="module")
def encoder():
    import torch

    from app.encoders.pe_core import PeCoreEncoder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return PeCoreEncoder(device=device, batch_size=4)


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
    assert encoder.embedding_dim == _EXPECTED_DIM


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


def test_batching_does_not_change_vectors(encoder, sample_texts):
    """batch_size is a throughput knob, never a correctness one -- mirrors
    test_siglip2_encoder.py's equivalent, catches padding/attention-mask
    bugs that only surface once a batch holds strings of differing length."""
    batched = encoder.encode_text(sample_texts)
    one_at_a_time = np.concatenate([encoder.encode_text([text]) for text in sample_texts], axis=0)

    np.testing.assert_allclose(batched, one_at_a_time, atol=1e-4)


def test_visual_tower_is_timm_backed(encoder):
    """Sanity check for the whole reason this candidate needs timm>=1.0.19 in
    the first place (see app/encoders/pe_core.py module docstring): if
    open_clip ever silently fell back to a non-timm stub, embedding_dim could
    still match by coincidence while the actual vision tower were wrong."""
    assert "timm" in type(encoder._model.visual).__module__.lower() or hasattr(
        encoder._model.visual, "trunk"
    )
