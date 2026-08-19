"""MultimodalEncoder Protocol for vision-language retrieval encoders.

Every caller in the pipeline talks to this interface only, never to a concrete
encoder implementation directly. Swapping the active semantic encoder must not
require touching store/search call sites.

Contract every implementation must satisfy (enforced by tests, see
tests/test_*_encoder.py):
- encode_image / encode_text return np.ndarray, shape (N, embedding_dim), dtype float32
- rows are L2-normalized inside the encoder — callers never normalize themselves
- same input -> same output vector (deterministic inference)
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

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
