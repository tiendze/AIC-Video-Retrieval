"""RED reproducer for P0.5 — BEiT3Encoder must satisfy the MultimodalEncoder contract.

Requires torch/timm/torchscale/transformers (see pyproject.toml `beit3` extra) and a
downloaded checkpoint (scripts/setup_beit3.sh) — none of that exists on a bare host,
so this whole module is skipped unless both are present. Run inside the GPU container:

    BEIT3_CHECKPOINT_PATH=models/beit3/beit3_base_itc_patch16_224.pth \
    BEIT3_SPM_PATH=models/beit3/beit3.spm \
    pytest -m model tests/test_beit3_encoder.py -v
"""

import os

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.model

_CHECKPOINT_PATH = os.environ.get("BEIT3_CHECKPOINT_PATH")
_SPM_PATH = os.environ.get("BEIT3_SPM_PATH")

if not _CHECKPOINT_PATH or not _SPM_PATH:
    pytest.skip(
        "BEIT3_CHECKPOINT_PATH / BEIT3_SPM_PATH not set — run scripts/setup_beit3.sh first",
        allow_module_level=True,
    )


@pytest.fixture(scope="module")
def encoder():
    from app.encoders.beit3 import BEiT3Encoder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return BEiT3Encoder(
        checkpoint_path=_CHECKPOINT_PATH,
        sentencepiece_model_path=_SPM_PATH,
        device=device,
        batch_size=4,
    )


@pytest.fixture()
def sample_images() -> list[Image.Image]:
    return [Image.new("RGB", (96, 64), color=(i * 40, 100, 200 - i * 30)) for i in range(3)]


@pytest.fixture()
def sample_texts() -> list[str]:
    return ["một người đàn ông đang nói chuyện", "bản tin thời sự buổi tối", "xe cứu hỏa"]


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
    assert encoder.embedding_dim == 768


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


def test_encode_image_patches_returns_normalized_local_patch_embeddings(encoder, sample_images):
    patches = encoder.encode_image_patches(sample_images)

    assert patches.shape[0] == len(sample_images)
    assert patches.shape[1] > 0
    assert patches.shape[2] == encoder.embedding_dim
    assert patches.dtype == np.float32
    assert np.isfinite(patches).all()
    np.testing.assert_allclose(np.linalg.norm(patches, axis=2), 1.0, atol=1e-4)


def test_encode_text_tokens_excludes_special_and_padding_tokens(encoder):
    encoded = encoder.encode_text_tokens(["red seasoning packet"])

    assert len(encoded) == 1
    assert encoded[0].embeddings.shape[0] == len(encoded[0].tokens)
    assert encoded[0].embeddings.shape[0] > 0
    assert encoded[0].embeddings.shape[1] == encoder.embedding_dim
    assert encoded[0].embeddings.dtype == np.float32
    assert np.isfinite(encoded[0].embeddings).all()
    np.testing.assert_allclose(np.linalg.norm(encoded[0].embeddings, axis=1), 1.0, atol=1e-4)
