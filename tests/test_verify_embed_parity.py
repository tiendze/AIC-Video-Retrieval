"""RED reproducer for pipelines/verify_embed_parity.py.

All pure-function tests -- no GPU, no real BEiT-3 checkpoint needed. A fake
`encode_text` callable stands in for BEiT3Encoder, same convention as
tests/test_embed_transcripts.py's _FakeEncoder.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from pipelines.verify_embed_parity import (
    EmbedParityError,
    ParityProbeResult,
    ParityReport,
    cosine_per_row,
    load_manifest,
    load_probe,
    run_parity_probe,
    sha256_file,
    verify_checksums,
    verify_hf_identity,
    verify_manifest_fields,
    verify_weight_sha256,
)


def _manifest(**overrides) -> dict:
    base = {
        "model_id": "beit3_base_itc_patch16_224",
        "embedding_dim": 768,
        "checkpoint_sha256": "checkpoint-hash",
        "spm_sha256": "spm-hash",
    }
    base.update(overrides)
    return base


def test_load_manifest_reads_json_file(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"model_id": "x"}), encoding="utf-8")

    result = load_manifest(manifest_path)

    assert result == {"model_id": "x"}


def test_sha256_file_matches_known_digest(tmp_path):
    path = tmp_path / "sample.bin"
    path.write_bytes(b"hello world")

    assert sha256_file(path) == hashlib.sha256(b"hello world").hexdigest()


def test_verify_checksums_passes_when_both_files_match(tmp_path):
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"checkpoint-bytes")
    spm = tmp_path / "tokenizer.spm"
    spm.write_bytes(b"spm-bytes")
    manifest = _manifest(
        checkpoint_sha256=hashlib.sha256(b"checkpoint-bytes").hexdigest(),
        spm_sha256=hashlib.sha256(b"spm-bytes").hexdigest(),
    )

    verify_checksums(manifest, checkpoint, spm)  # must not raise


def test_verify_checksums_raises_on_checkpoint_mismatch(tmp_path):
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"wrong-bytes")
    spm = tmp_path / "tokenizer.spm"
    spm.write_bytes(b"spm-bytes")
    manifest = _manifest(checkpoint_sha256="expected-hash", spm_sha256="irrelevant")

    with pytest.raises(EmbedParityError, match="checkpoint_sha256 mismatch"):
        verify_checksums(manifest, checkpoint, spm)


def test_verify_checksums_raises_on_spm_mismatch(tmp_path):
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"checkpoint-bytes")
    spm = tmp_path / "tokenizer.spm"
    spm.write_bytes(b"wrong-bytes")
    manifest = _manifest(
        checkpoint_sha256=hashlib.sha256(b"checkpoint-bytes").hexdigest(),
        spm_sha256="expected-hash",
    )

    with pytest.raises(EmbedParityError, match="spm_sha256 mismatch"):
        verify_checksums(manifest, checkpoint, spm)


def test_verify_manifest_fields_passes_when_model_id_and_dim_match():
    manifest = _manifest(model_id="beit3_base_itc_patch16_224", embedding_dim=768)

    verify_manifest_fields(manifest, model_id="beit3_base_itc_patch16_224", embedding_dim=768)


def test_verify_manifest_fields_raises_on_model_id_mismatch():
    manifest = _manifest(model_id="beit3_base_itc_patch16_224")

    with pytest.raises(EmbedParityError, match="model_id mismatch"):
        verify_manifest_fields(manifest, model_id="siglip_base", embedding_dim=768)


def test_verify_manifest_fields_raises_on_embedding_dim_mismatch():
    manifest = _manifest(embedding_dim=768)

    with pytest.raises(EmbedParityError, match="embedding_dim mismatch"):
        verify_manifest_fields(manifest, model_id="beit3_base_itc_patch16_224", embedding_dim=512)


def test_load_probe_reads_texts_and_vectors(tmp_path):
    probe_dir = tmp_path / "probe"
    probe_dir.mkdir()
    (probe_dir / "probe_texts.json").write_text(json.dumps(["a", "b"]), encoding="utf-8")
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    np.save(probe_dir / "probe_text_vectors.npy", vectors)

    texts, loaded_vectors = load_probe(probe_dir)

    assert texts == ["a", "b"]
    np.testing.assert_array_equal(loaded_vectors, vectors)


def test_load_probe_raises_on_count_mismatch(tmp_path):
    probe_dir = tmp_path / "probe"
    probe_dir.mkdir()
    (probe_dir / "probe_texts.json").write_text(json.dumps(["a", "b", "c"]), encoding="utf-8")
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    np.save(probe_dir / "probe_text_vectors.npy", vectors)

    with pytest.raises(EmbedParityError, match="probe text/vector count mismatch"):
        load_probe(probe_dir)


def test_cosine_per_row_is_one_for_identical_vectors():
    a = np.array([[1.0, 2.0, 3.0], [0.5, 0.5, 0.5]], dtype=np.float32)

    result = cosine_per_row(a, a)

    np.testing.assert_allclose(result, [1.0, 1.0], atol=1e-6)


def test_cosine_per_row_is_zero_for_orthogonal_vectors():
    a = np.array([[1.0, 0.0]], dtype=np.float32)
    b = np.array([[0.0, 1.0]], dtype=np.float32)

    result = cosine_per_row(a, b)

    np.testing.assert_allclose(result, [0.0], atol=1e-6)


def test_run_parity_probe_returns_one_result_per_text_with_matching_cosine():
    texts = ["mot", "hai"]
    remote_vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    def fake_encode_text(batch):
        assert list(batch) == texts
        return np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    results = run_parity_probe(texts, remote_vectors, fake_encode_text)

    assert results == (
        ParityProbeResult(text="mot", cosine=pytest.approx(1.0)),
        ParityProbeResult(text="hai", cosine=pytest.approx(1.0)),
    )


def test_run_parity_probe_detects_low_cosine_on_mismatched_encoder():
    texts = ["mot"]
    remote_vectors = np.array([[1.0, 0.0]], dtype=np.float32)

    def fake_encode_text(batch):
        return np.array([[0.0, 1.0]], dtype=np.float32)

    results = run_parity_probe(texts, remote_vectors, fake_encode_text)

    assert results[0].cosine == pytest.approx(0.0)


def test_parity_report_passed_true_when_all_cosines_above_threshold():
    report = ParityReport(
        probe_results=(
            ParityProbeResult(text="a", cosine=0.99995),
            ParityProbeResult(text="b", cosine=0.9999),
        ),
        threshold=0.9999,
    )

    assert report.passed is True
    assert report.min_cosine == pytest.approx(0.9999)


def test_parity_report_passed_false_when_any_cosine_below_threshold():
    report = ParityReport(
        probe_results=(
            ParityProbeResult(text="a", cosine=0.99995),
            ParityProbeResult(text="b", cosine=0.5),
        ),
        threshold=0.9999,
    )

    assert report.passed is False
    assert report.min_cosine == pytest.approx(0.5)


def _candidate_manifest(**overrides) -> dict:
    base = {
        "manifest_version": 2,
        "encoder_kind": "siglip2",
        "model_id": "siglip2_so400m_p14_384",
        "embedding_dim": 1152,
        "hf_repo_id": "google/siglip2-so400m-patch14-384",
        "hf_revision": "e8e487298228002f3d8a82e0cd5c8ea9c567f57f",
        "weight_files_sha256": {"model.safetensors": "weight-hash"},
    }
    base.update(overrides)
    return base


def test_verify_hf_identity_passes_when_repo_and_revision_match():
    manifest = _candidate_manifest()

    verify_hf_identity(
        manifest,
        hf_repo_id="google/siglip2-so400m-patch14-384",
        hf_revision="e8e487298228002f3d8a82e0cd5c8ea9c567f57f",
    )  # must not raise


def test_verify_hf_identity_raises_on_repo_id_mismatch():
    manifest = _candidate_manifest(hf_repo_id="google/siglip2-so400m-patch14-384")

    with pytest.raises(EmbedParityError, match="hf_repo_id mismatch"):
        verify_hf_identity(
            manifest, hf_repo_id="google/siglip2-base-patch16-224", hf_revision="anything"
        )


def test_verify_hf_identity_raises_on_revision_mismatch():
    manifest = _candidate_manifest(hf_revision="e8e487298228002f3d8a82e0cd5c8ea9c567f57f")

    with pytest.raises(EmbedParityError, match="hf_revision mismatch"):
        verify_hf_identity(
            manifest,
            hf_repo_id="google/siglip2-so400m-patch14-384",
            hf_revision="0000000000000000000000000000000000000",
        )


def test_verify_weight_sha256_passes_when_all_files_match():
    manifest = _candidate_manifest(
        weight_files_sha256={"model.safetensors": "abc123", "extra.bin": "def456"}
    )

    verify_weight_sha256(
        manifest, compute_sha256=lambda filename: {"model.safetensors": "abc123", "extra.bin": "def456"}[filename]
    )  # must not raise


def test_verify_weight_sha256_raises_on_mismatch():
    manifest = _candidate_manifest(weight_files_sha256={"model.safetensors": "expected-hash"})

    with pytest.raises(EmbedParityError, match="weight_files_sha256 mismatch for model.safetensors"):
        verify_weight_sha256(manifest, compute_sha256=lambda filename: "different-hash")


def test_verify_weight_sha256_raises_when_manifest_missing_field():
    manifest = _candidate_manifest()
    del manifest["weight_files_sha256"]

    with pytest.raises(EmbedParityError, match="missing 'weight_files_sha256'"):
        verify_weight_sha256(manifest, compute_sha256=lambda filename: "irrelevant")
