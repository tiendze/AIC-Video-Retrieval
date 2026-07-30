"""MultimodalEncoder Protocol — the risk-R1 firewall around BEiT-3.

Every caller in the pipeline talks to this interface only, never to a concrete
encoder implementation directly. Swapping BEiT3Encoder for SigLIPEncoder (fallback
if BEiT-3 setup blows the 1-day timebox) must not require touching call sites.

Contract every implementation must satisfy (enforced by tests, see
tests/test_beit3_encoder.py):
- encode_image / encode_text return np.ndarray, shape (N, embedding_dim), dtype float32
- rows are L2-normalized inside the encoder — callers never normalize themselves
- same input -> same output vector (deterministic inference)
"""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np
from PIL import Image


class MultimodalEncoder(Protocol):
    """Maps images and text into the same L2-normalized embedding space."""

    @property
    def embedding_dim(self) -> int: ...

    @property
    def model_id(self) -> str:
        """Identifier embedded in the Milvus collection name.

        Changing the underlying model/checkpoint must change model_id, which in
        turn changes the collection name — this makes mixing embeddings from two
        different models impossible at the design level, not just by convention.
        """
        ...

    def encode_image(self, images: Sequence[Image.Image]) -> np.ndarray:
        """Return shape (N, embedding_dim), dtype float32, L2-normalized."""
        ...

    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        """Return shape (N, embedding_dim), dtype float32, L2-normalized."""
        ...
