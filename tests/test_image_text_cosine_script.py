from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from scripts.image_text_cosine import (
    _build_result,
    _cosine_similarity,
    _encode_pair,
    _resolve_image_path,
)


class _FakeEncoder:
    model_id = "fake-model"
    embedding_dim = 3

    def __init__(self) -> None:
        self.images = []
        self.texts = []

    def encode_image(self, images):
        self.images.append(list(images))
        return np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

    def encode_text(self, texts):
        self.texts.append(list(texts))
        return np.array([[0.5, 0.5, 0.70710677]], dtype=np.float32)


def test_cosine_similarity_uses_dot_product_for_normalized_vectors() -> None:
    image_vector = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    text_vector = np.array([0.5, 0.5, 0.70710677], dtype=np.float32)

    assert _cosine_similarity(image_vector, text_vector) == pytest.approx(0.5)


def test_cosine_similarity_raises_on_dimension_mismatch() -> None:
    with pytest.raises(ValueError, match="vector dimensions do not match"):
        _cosine_similarity(np.array([1.0, 0.0]), np.array([1.0, 0.0, 0.0]))


def test_resolve_image_path_accepts_direct_relative_path(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("scripts.image_text_cosine._REPO_ROOT", tmp_path)
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"image-bytes")

    assert _resolve_image_path("frame.jpg") == image_path


def test_resolve_image_path_finds_filename_under_image_root(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("scripts.image_text_cosine._REPO_ROOT", tmp_path)
    image_root = tmp_path / "keyframes"
    nested = image_root / "L21_V001"
    nested.mkdir(parents=True)
    image_path = nested / "000001.jpg"
    image_path.write_bytes(b"image-bytes")

    assert _resolve_image_path("000001.jpg", "keyframes") == image_path


def test_resolve_image_path_prefers_relative_path_under_image_root(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("scripts.image_text_cosine._REPO_ROOT", tmp_path)
    for dirname in ("L21_V001", "L21_V002"):
        directory = tmp_path / "keyframes" / dirname
        directory.mkdir(parents=True)
        (directory / "000001.jpg").write_bytes(b"image-bytes")

    assert _resolve_image_path("L21_V001/000001.jpg", "keyframes") == (
        tmp_path / "keyframes" / "L21_V001" / "000001.jpg"
    )


def test_resolve_image_path_rejects_duplicate_filename_matches(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("scripts.image_text_cosine._REPO_ROOT", tmp_path)
    for dirname in ("a", "b"):
        directory = tmp_path / "keyframes" / dirname
        directory.mkdir(parents=True)
        (directory / "frame.jpg").write_bytes(b"image-bytes")

    with pytest.raises(ValueError, match="matched multiple files"):
        _resolve_image_path("frame.jpg", "keyframes")


def test_encode_pair_sends_rgb_image_and_text_to_encoder(tmp_path) -> None:
    image_path = tmp_path / "frame.png"
    Image.new("RGBA", (2, 2), color=(255, 0, 0, 255)).save(image_path)
    encoder = _FakeEncoder()

    image_vector, text_vector = _encode_pair(encoder, image_path, "red object")

    assert encoder.images[0][0].mode == "RGB"
    assert encoder.texts == [["red object"]]
    np.testing.assert_array_equal(image_vector, np.array([1.0, 0.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(text_vector, np.array([0.5, 0.5, 0.70710677], dtype=np.float32))


def test_build_result_includes_encoder_metadata(tmp_path) -> None:
    encoder = _FakeEncoder()

    result = _build_result(
        image_path=tmp_path / "frame.jpg",
        text="red object",
        encoder=encoder,
        config_path=tmp_path / "beit3.yaml",
        cosine=0.5,
    )

    assert result == {
        "image": str(tmp_path / "frame.jpg"),
        "text": "red object",
        "cosine_similarity": 0.5,
        "model_id": "fake-model",
        "embedding_dim": 3,
        "config": str(tmp_path / "beit3.yaml"),
    }
