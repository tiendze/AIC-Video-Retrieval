"""PeCoreEncoder -- MultimodalEncoder backed by timm/PE-Core-bigG-14-448 (Meta
Perception Encoder, remapped for OpenCLIP), candidate A2 of the encoder-swap
plan (.claude/plans/encoder-swap-siglip2-pecore.plan.md §T3). This is the
vision tower MERVIN (arXiv 2605.16120) used to reach 79/88 on the AIC
preliminary round. Text tower is a standard CLIP BPE tokenizer, English-only
(open_clip_config.json: text_cfg.context_length=72, vocab_size=49408,
custom_text=true) -- NOT the multilingual Gemma tokenizer SigLip2Encoder
uses, so Vietnamese queries MUST still go through query_translator; there is
no "pecore_vi" channel the way "siglip2_vi" exists for A1
(docs/research-encoder-vietnamese.md §6.3).

License verified 2026-08-14 via
https://huggingface.co/api/models/timm/PE-Core-bigG-14-448 : apache-2.0, no
FAIR-NC/research-only restriction -- closes
docs/research-encoder-vietnamese.md §9 Q1 (the "Meta might be non-commercial"
worry does not apply to this checkpoint).

embedding_dim=1280, image_size=448 verified 2026-08-14 by fetching
https://huggingface.co/timm/PE-Core-bigG-14-448/raw/main/open_clip_config.json
directly: model_cfg.embed_dim=1280, vision_cfg.timm_model_name=
"vit_pe_core_gigantic_patch14_448" -- confirms the vision tower is
timm-backed (hence pyproject.toml's `timm>=1.0.19` pin in the
encoder-candidates extra; plain open_clip architectures like ClipEncoder's
ViT-B/32 never touch timm at all).

L2-normalization happens here explicitly (CLAUDE.md §3: "L2-normalize mọi
embedding ngay tại encoder") -- open_clip's encode_image/encode_text are not
normalized on their own, same as ClipEncoder.

Warning: hf_revision is NOT enforceable at load time for this encoder,
unlike SigLip2Encoder's `AutoModel.from_pretrained(revision=...)`. Verified
2026-08-14 against open_clip==3.3.0 source (src/open_clip/factory.py,
src/open_clip/pretrained.py): create_model_and_transforms("hf-hub:<repo>")
calls download_pretrained_from_hf(model_id, cache_dir=cache_dir) with NO
revision argument, even though download_pretrained_from_hf() itself accepts
one -- the factory's hf-hub branch just never forwards it. There is no
public `revision=` kwarg on create_model_and_transforms. Consequences:
  1. This class only takes hf_repo_id, not hf_revision -- accepting a
     hf_revision parameter that silently does nothing would itself be the
     kind of "hỏng âm thầm" CLAUDE.md §3 warns about.
  2. The parity gate for this candidate (encoder-swap plan §T4, not yet
     implemented for pe_core) must rely on weight_files_sha256 + image/text
     probe comparison to catch drift, not on a load-time pin --
     notebooks/kaggle_embed_pecore.ipynb records whatever commit "main"
     actually resolved to at encode time into manifest.json as provenance,
     not as an enforced pin.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import open_clip
import torch
from PIL import Image

_EMBEDDING_DIM = 1280  # verified 2026-08-14 from open_clip_config.json model_cfg.embed_dim


class PeCoreEncoder:
    """MultimodalEncoder implementation wrapping open_clip's PE-Core-bigG-14-448."""

    def __init__(
        self,
        hf_repo_id: str = "timm/PE-Core-bigG-14-448",
        device: str = "cuda",
        batch_size: int = 8,
        model_id: str = "pe_core_bigg14_448",
        preprocess_workers: int = 8,
    ) -> None:
        self._device = torch.device(device)
        self._batch_size = batch_size
        self._model_id = model_id
        open_clip_model_name = f"hf-hub:{hf_repo_id}"
        model, _, preprocess = open_clip.create_model_and_transforms(open_clip_model_name)
        self._model = model.to(self._device).eval()
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer(open_clip_model_name)
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
