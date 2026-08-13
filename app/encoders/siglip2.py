"""SigLip2Encoder -- MultimodalEncoder backed by google/siglip2-so400m-patch14-384,
candidate A1 of the encoder-swap plan
(.claude/plans/encoder-swap-siglip2-pecore.plan.md §T2). Multilingual text
tower (Gemma tokenizer, 109 languages) -- the point of this candidate is
testing whether Vietnamese queries can skip the VI->EN translation step that
BEiT3Encoder needs (docs/research-encoder-vietnamese.md §6.3).

Used zero-shot, same as BEiT3Encoder/ClipEncoder (no dataset-specific
fine-tuning -- PRD §1.4 has no ground truth to fine-tune against).

L2-normalization happens here explicitly (CLAUDE.md §3: "L2-normalize mọi
embedding ngay tại encoder") -- transformers' get_image_features/
get_text_features do not normalize on their own.

CRITICAL preprocessing detail, verified 2026-08-13 against
https://huggingface.co/docs/transformers/main/en/model_doc/siglip2 :
"IMPORTANT: we pass padding=max_length and max_length=64 since the model was
trained with this". The default/dynamic padding a naive AutoProcessor call
would use still runs without error and still returns a normalized vector --
it is just the WRONG vector. Same failure class as CLIP's QuickGELU mismatch
(configs/model/clip.yaml). encode_text hard-codes padding="max_length",
max_length=64, truncation=True; these are not exposed as free parameters.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

_MAX_TEXT_TOKENS = 64


class SigLip2Encoder:
    """MultimodalEncoder implementation wrapping transformers' Siglip2Model."""

    def __init__(
        self,
        hf_repo_id: str = "google/siglip2-so400m-patch14-384",
        hf_revision: str = "e8e487298228002f3d8a82e0cd5c8ea9c567f57f",
        device: str = "cuda",
        batch_size: int = 16,
        model_id: str = "siglip2_so400m_p14_384",
        preprocess_workers: int = 8,
    ) -> None:
        self._device = torch.device(device)
        self._batch_size = batch_size
        self._model_id = model_id
        self._model = (
            AutoModel.from_pretrained(hf_repo_id, revision=hf_revision).to(self._device).eval()
        )
        self._processor = AutoProcessor.from_pretrained(hf_repo_id, revision=hf_revision)
        self._embedding_dim = self._model.config.text_config.projection_size
        self._preprocess_executor = ThreadPoolExecutor(max_workers=preprocess_workers)

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def model_id(self) -> str:
        return self._model_id

    def encode_image(self, images: Sequence[Image.Image]) -> np.ndarray:
        if not images:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        vectors = [self._encode_image_batch(batch) for batch in self._batched(images)]
        return np.concatenate(vectors, axis=0).astype(np.float32)

    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        vectors = [self._encode_text_batch(batch) for batch in self._batched(texts)]
        return np.concatenate(vectors, axis=0).astype(np.float32)

    def debug_tokenize(self, texts: Sequence[str]) -> torch.Tensor:
        """Exposes the tokenized `input_ids` this encoder actually feeds the
        model -- test-only escape hatch (tests/test_siglip2_encoder.py) to
        assert padding="max_length" is really in effect, since a wrong
        padding mode fails silently otherwise (see module docstring)."""
        return self._tokenize(texts)["input_ids"]

    def _batched(self, items: Sequence) -> list:
        return [items[i : i + self._batch_size] for i in range(0, len(items), self._batch_size)]

    def _encode_image_batch(self, images: Sequence[Image.Image]) -> np.ndarray:
        pixel_values = torch.stack(
            list(self._preprocess_executor.map(self._preprocess_one, images))
        ).to(self._device)
        with torch.no_grad(), self._autocast():
            features = self._model.get_image_features(pixel_values=pixel_values)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.float().cpu().numpy()

    def _preprocess_one(self, image: Image.Image) -> torch.Tensor:
        # The FULL transform (convert + resize/normalize/tensor-build) must run
        # inside the thread pool, not just .convert("RGB") -- calling the whole
        # processor once on a pre-batched list runs single-threaded, defeating
        # preprocess_workers entirely (measured 2026-08-14: batch_size 16->64,
        # a 4x jump, only bought +7% throughput with both CPU and GPU sitting
        # idle -- the tell that the bottleneck wasn't in either of them, it was
        # this single-threaded call). Verified bit-identical to a batched call
        # on the same images (max abs diff 0.0) before making this change --
        # the processor is a stateless per-image transform, no cross-image
        # batch statistics, so per-image vs batched calls cannot legally differ.
        return self._processor(images=[image.convert("RGB")], return_tensors="pt")["pixel_values"][0]

    def _encode_text_batch(self, texts: Sequence[str]) -> np.ndarray:
        inputs = self._tokenize(texts).to(self._device)
        with torch.no_grad(), self._autocast():
            features = self._model.get_text_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.float().cpu().numpy()

    def _tokenize(self, texts: Sequence[str]):
        # padding="max_length" / max_length=64 are NOT free parameters -- see
        # module docstring. Do not change these without re-reading the docs.
        return self._processor(
            text=list(texts),
            padding="max_length",
            max_length=_MAX_TEXT_TOKENS,
            truncation=True,
            return_tensors="pt",
        )

    def _autocast(self):
        return torch.autocast(
            device_type=self._device.type, dtype=torch.float16, enabled=self._device.type == "cuda"
        )
