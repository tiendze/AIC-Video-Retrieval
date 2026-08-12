"""ClipEncoder -- MultimodalEncoder backed by open_clip's CLIP ViT-B/32
(OpenAI weights), the P1.4 mandatory baseline (PRD §6.3): "CLIP ViT-B/32,
baseline multimodal phổ biến ... phải chạy bằng encoder của mình trên
keyframe của mình" -- BTC's own `clip-features-32/` is keyed to BTC's
keyframes, not ours, so it cannot substitute (references.md §2.5).

Used zero-shot, same as BEiT3Encoder (no dataset-specific fine-tuning --
PRD §1.4 has no ground truth to fine-tune against).

open_clip's `create_model_and_transforms` already returns a model whose
`encode_image`/`encode_text` are NOT L2-normalized -- normalization happens
here, explicitly, matching the encoder-contract invariant (CLAUDE.md §3:
"L2-normalize mọi embedding ngay tại encoder, không rải rác ở nơi gọi").

Deliberately does NOT call `torch.cuda.empty_cache()` per batch the way
BEiT3Encoder does: that pattern exists there because BEiT-3 runs inside the
live FastAPI process sharing the 8GB VRAM budget with other in-process state
(CLAUDE.md §1.4). This encoder's two callers -- the standalone
`scripts/embed_clip_baseline.py` batch job, and `eval/run.py`'s `clip`
channel which runs strictly AFTER BEiT-3's own channels finish, never
concurrently -- have nothing else contending for VRAM during a batch loop.
`empty_cache()` returns cached memory to the CUDA driver for OTHER
processes/contexts to use; within one loop it only adds a synchronization +
free/realloc round-trip per batch, which measurably hurts throughput on a
small model like ViT-B/32 where that overhead can rival the batch's own
compute time. PyTorch's caching allocator already reuses freed activation
memory across batches in the SAME process without this call.

Image preprocessing (`_preprocess_one`: PIL decode + convert + resize/crop
+ normalize-to-tensor) runs on a `ThreadPoolExecutor`, not the sequential
list comprehension a naive port of BEiT3Encoder would use. Measured
2026-08-12 on the actual embed job: batch_size 64->256 (4x) only bought
+27% throughput (47 -> 59.5 keyframes/s) while VRAM stayed flat at ~1-1.5GB
-- both signs point at PIL image decode as the real bottleneck, not GPU
compute, since a GPU-bound workload would scale much closer to linearly
with batch size. Threading (not multiprocessing) works here because
Pillow's C-level decode/resize releases the GIL for the bulk of its work,
so multiple images preprocess concurrently on separate CPU cores while the
GPU stays busy on the previous batch's forward pass -- no image data
crosses a process boundary, so there's no pickling cost either.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import open_clip
import torch
from PIL import Image

_EMBEDDING_DIM = 512  # ViT-B/32's projection dim -- fixed by the architecture.


class ClipEncoder:
    """MultimodalEncoder implementation wrapping open_clip's CLIP ViT-B/32."""

    def __init__(
        self,
        model_name: str = "ViT-B-32-quickgelu",
        pretrained_tag: str = "openai",
        device: str = "cuda",
        batch_size: int = 64,
        model_id: str = "clip_vit_b32_openai",
        preprocess_workers: int = 8,
    ) -> None:
        self._device = torch.device(device)
        self._batch_size = batch_size
        self._model_id = model_id
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained_tag
        )
        self._model = model.to(self._device).eval()
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer(model_name)
        self._preprocess_executor = ThreadPoolExecutor(max_workers=preprocess_workers)

    @property
    def embedding_dim(self) -> int:
        return _EMBEDDING_DIM

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

    def _batched(self, items: Sequence) -> list:
        return [items[i : i + self._batch_size] for i in range(0, len(items), self._batch_size)]

    def _encode_image_batch(self, images: Sequence[Image.Image]) -> np.ndarray:
        preprocessed = list(self._preprocess_executor.map(self._preprocess_one, images))
        pixel_values = torch.stack(preprocessed).to(self._device)
        with torch.no_grad(), self._autocast():
            features = self._model.encode_image(pixel_values)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.float().cpu().numpy()

    def _preprocess_one(self, image: Image.Image) -> torch.Tensor:
        return self._preprocess(image.convert("RGB"))

    def _encode_text_batch(self, texts: Sequence[str]) -> np.ndarray:
        tokens = self._tokenizer(list(texts)).to(self._device)
        with torch.no_grad(), self._autocast():
            features = self._model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.float().cpu().numpy()

    def _autocast(self):
        return torch.autocast(
            device_type=self._device.type, dtype=torch.float16, enabled=self._device.type == "cuda"
        )
