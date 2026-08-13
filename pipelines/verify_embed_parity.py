"""Verifies a Kaggle-produced embed_out artifact before it is imported.

Local has no images to re-encode and compare against (kaggle-offline-embed
plan §3), so the only way to catch "checkpoint on Kaggle != checkpoint local
will use for query-time encoding" is: (1) hash-compare the checkpoint/tokenizer
files, and (2) re-encode the manifest's fixed probe texts locally and check
they land almost exactly on the vectors Kaggle computed for them. Either check
failing must block import outright (CLAUDE.md §3 "một encoder duy nhất") —
never warn and continue.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class EmbedParityError(Exception):
    """A manifest/checksum/probe check failed -- import must not proceed."""


@dataclass(frozen=True)
class ParityProbeResult:
    text: str
    cosine: float


@dataclass(frozen=True)
class ParityReport:
    probe_results: tuple[ParityProbeResult, ...]
    threshold: float

    @property
    def min_cosine(self) -> float:
        return min(result.cosine for result in self.probe_results)

    @property
    def passed(self) -> bool:
        return self.min_cosine >= self.threshold


def load_manifest(manifest_path: Path) -> dict:
    with manifest_path.open(encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checksums(manifest: dict, checkpoint_path: Path, spm_path: Path) -> None:
    """Raises EmbedParityError if the local checkpoint/tokenizer bytes differ from
    what produced this artifact on Kaggle."""
    actual_checkpoint = sha256_file(checkpoint_path)
    if actual_checkpoint != manifest["checkpoint_sha256"]:
        raise EmbedParityError(
            f"checkpoint_sha256 mismatch: manifest={manifest['checkpoint_sha256']} "
            f"local({checkpoint_path})={actual_checkpoint}"
        )
    actual_spm = sha256_file(spm_path)
    if actual_spm != manifest["spm_sha256"]:
        raise EmbedParityError(
            f"spm_sha256 mismatch: manifest={manifest['spm_sha256']} "
            f"local({spm_path})={actual_spm}"
        )


def verify_manifest_fields(manifest: dict, model_id: str, embedding_dim: int) -> None:
    """Raises EmbedParityError if the artifact was produced by a different encoder
    identity than the one about to run the probe."""
    if manifest["model_id"] != model_id:
        raise EmbedParityError(
            f"model_id mismatch: manifest={manifest['model_id']} local={model_id}"
        )
    if manifest["embedding_dim"] != embedding_dim:
        raise EmbedParityError(
            f"embedding_dim mismatch: manifest={manifest['embedding_dim']} local={embedding_dim}"
        )


def load_probe(probe_dir: Path) -> tuple[list[str], np.ndarray]:
    with (probe_dir / "probe_texts.json").open(encoding="utf-8") as f:
        texts = json.load(f)
    vectors = np.load(probe_dir / "probe_text_vectors.npy")
    if len(texts) != vectors.shape[0]:
        raise EmbedParityError(
            f"probe text/vector count mismatch: {len(texts)} texts, {vectors.shape[0]} vectors"
        )
    return texts, vectors


def cosine_per_row(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between two (N, dim) arrays."""
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.sum(a_norm * b_norm, axis=1)


def verify_hf_identity(manifest: dict, hf_repo_id: str, hf_revision: str) -> None:
    """Raises EmbedParityError if the artifact's manifest doesn't name the SAME
    HuggingFace repo/revision this encoder config is pinned to. This is the
    HF-hub-based encoders' equivalent of verify_checksums() -- there is no
    local checkpoint FILE to hash (weights load via `from_pretrained`), so
    identity starts with (repo_id, commit revision); verify_weight_sha256()
    below covers the case the revision matches but the underlying bytes
    somehow don't."""
    if manifest.get("hf_repo_id") != hf_repo_id:
        raise EmbedParityError(
            f"hf_repo_id mismatch: manifest={manifest.get('hf_repo_id')!r} local={hf_repo_id!r}"
        )
    if manifest.get("hf_revision") != hf_revision:
        raise EmbedParityError(
            f"hf_revision mismatch: manifest={manifest.get('hf_revision')!r} local={hf_revision!r} "
            "-- Kaggle and local are pinned to different commits of the same model"
        )


def verify_weight_sha256(manifest: dict, compute_sha256: Callable[[str], str]) -> None:
    """Raises EmbedParityError if any weight file's hash in
    `manifest['weight_files_sha256']` doesn't match what `compute_sha256`
    (filename -> sha256 hex digest) reports right now. `compute_sha256` is
    injected -- same DI convention as `run_parity_probe`'s `encode_text` --
    so this stays a pure function in tests; `hf_hub_weight_sha256` below is
    the real (network-touching) implementation used outside tests."""
    manifest_hashes = manifest.get("weight_files_sha256")
    if not manifest_hashes:
        raise EmbedParityError("manifest missing 'weight_files_sha256' -- cannot verify weight identity")
    for filename, expected_sha256 in manifest_hashes.items():
        actual_sha256 = compute_sha256(filename)
        if actual_sha256 != expected_sha256:
            raise EmbedParityError(
                f"weight_files_sha256 mismatch for {filename}: "
                f"manifest={expected_sha256} local={actual_sha256}"
            )


def hf_hub_weight_sha256(hf_repo_id: str, hf_revision: str) -> Callable[[str], str]:
    """Returns a `compute_sha256(filename)` callable bound to one
    (repo, revision) pair, for `verify_weight_sha256`. Downloads (or reuses
    the local cache of) the file from the HF hub -- the exact bytes
    `AutoModel.from_pretrained(hf_repo_id, revision=hf_revision)` would load
    locally, so a match here really does mean "same weights Kaggle used".

    Imports `huggingface_hub` lazily: this module is ALSO used for BEiT-3
    verification (`verify_checksums`), which must never require installing
    it just to run a routine beit3 import."""

    def _compute(filename: str) -> str:
        from huggingface_hub import hf_hub_download

        local_path = Path(hf_hub_download(repo_id=hf_repo_id, filename=filename, revision=hf_revision))
        return sha256_file(local_path)

    return _compute


def run_parity_probe(
    texts: Sequence[str],
    remote_vectors: np.ndarray,
    encode_text: Callable[[Sequence[str]], np.ndarray],
) -> tuple[ParityProbeResult, ...]:
    """Encodes `texts` locally via `encode_text` and compares each row to `remote_vectors`."""
    local_vectors = encode_text(list(texts))
    cosines = cosine_per_row(local_vectors, remote_vectors)
    return tuple(
        ParityProbeResult(text=text, cosine=float(cosine))
        for text, cosine in zip(texts, cosines, strict=True)
    )
