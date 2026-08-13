"""RED reproducer for the encoder swap (plan §T2) — SigLip2Encoder must
satisfy the MultimodalEncoder contract (same mirror as
tests/test_clip_encoder.py / tests/test_beit3_encoder.py).

Requires transformers >= 4.49 (SigLIP 2 landed there; the repo's `beit3`
extra pins 4.44.2, so this module SKIPS on the default env) and downloads
~1.6GB of so400m weights on first run (network required, then cached):

    pip install -e '.[encoder-candidates]'
    pytest -m model tests/test_siglip2_encoder.py -v

The last test here is the one that catches the failure mode the docs warn
about loudest: SigLIP 2 was trained with `padding="max_length"`,
`max_length=64`, so a processor call using the default padding produces a
silently WRONG text vector — no exception, just worse retrieval. Same class
of trap as CLIP's QuickGELU mismatch (configs/model/clip.yaml:7-14).
"""

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

if tuple(int(part) for part in transformers.__version__.split(".")[:2]) < (4, 49):
    pytest.skip(
        f"SigLIP 2 needs transformers >= 4.49, found {transformers.__version__}",
        allow_module_level=True,
    )

pytestmark = pytest.mark.model

_EXPECTED_DIM = 1152  # verified 2026-08-13 from the checkpoint's config.json


@pytest.fixture(scope="module")
def encoder():
    import torch

    from app.encoders.siglip2 import SigLip2Encoder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SigLip2Encoder(device=device, batch_size=4)


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
    """batch_size is a throughput knob, never a correctness one -- a text
    encoded alone must land on the same vector as when it rides in a batch.
    Catches padding/attention-mask bugs that only surface once a batch holds
    strings of differing length."""
    batched = encoder.encode_text(sample_texts)
    one_at_a_time = np.concatenate([encoder.encode_text([text]) for text in sample_texts], axis=0)

    np.testing.assert_allclose(batched, one_at_a_time, atol=1e-4)


def test_text_is_padded_to_max_length_the_model_was_trained_with(encoder, sample_texts):
    """SigLIP 2 was trained with padding="max_length", max_length=64
    (huggingface.co/docs/transformers/main/en/model_doc/siglip2: "IMPORTANT:
    we pass padding=max_length and max_length=64 since the model was trained
    with this"). Default/dynamic padding still runs and still returns a
    normalized vector -- it is just the WRONG vector, which no other test
    here would notice. This asserts the encoder pins the trained setting.
    """
    token_ids = encoder.debug_tokenize(sample_texts)

    assert token_ids.shape[1] == 64, (
        "text must be padded to exactly 64 tokens; dynamic padding produces "
        "silently wrong embeddings with SigLIP 2"
    )
