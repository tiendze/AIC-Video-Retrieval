"""Bgem3Encoder — text-to-text embedding for the ASR transcript channel (P0.11b).

Deliberately a *third* encoder, not a reuse of `BEiT3Encoder`: `beit3_base_itc`
is trained image-text contrastive, not text-text, and its alignment was
learned on English captions — Vietnamese-only text pairs land off-distribution
even though `XLMRobertaTokenizer` is multilingual at the token level
(references.md, plan P0.11b). This does not violate the CLAUDE.md §3 "một
encoder duy nhất" invariant: that invariant is about ingest and query *within
the same channel* using the same model, not about banning a second encoder
for a structurally different channel — but `model_id` still must be embedded
in the Milvus collection name (`transcripts_{model_id}`).

Text-only: this class does not implement `MultimodalEncoder` (no
`encode_image`) — bge-m3 has no image capability, so satisfying that
Protocol would mean a stub method that always raises, which is worse than
just not claiming the interface.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

_EMBEDDING_DIM = 1024


class Bgem3Encoder:
    """Wraps `sentence_transformers.SentenceTransformer("BAAI/bge-m3")`
    (verified 2026-08-02, references.md §5.4). Only the dense embedding is
    used here — bge-m3 also supports sparse/ColBERT-style multi-vector
    output via `FlagEmbedding`, out of scope for P0.11b."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = "cuda",
        batch_size: int = 32,
        model_id: str = "bge_m3",
    ) -> None:
        from sentence_transformers import SentenceTransformer  # lazy: heavy, GPU-bound

        self._model = SentenceTransformer(model_name, device=device)
        self._batch_size = batch_size
        self._model_id = model_id

    @property
    def embedding_dim(self) -> int:
        return _EMBEDDING_DIM

    @property
    def model_id(self) -> str:
        return self._model_id

    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        """Return shape (N, embedding_dim), dtype float32, L2-normalized.

        L2-normalizes explicitly rather than trusting the library default
        (CLAUDE.md §3: normalization happens at the encoder, not scattered
        at call sites) — `sentence-transformers`' own normalization default
        is not something this project's tests have verified against a real
        checkpoint."""
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        vectors = self._model.encode(
            list(texts), batch_size=self._batch_size, convert_to_numpy=True
        ).astype(np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # guard: an all-zero embedding must not divide by zero
        return vectors / norms
