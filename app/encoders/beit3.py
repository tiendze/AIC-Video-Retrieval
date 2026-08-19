"""BEiT3Encoder — MultimodalEncoder backed by the beit3_base_itc checkpoint.

Used zero-shot (no dataset-specific fine-tuning): the AIC 2025 dataset has no
ground truth to fine-tune against (PRD §1.4), and the baseline paper does not
fine-tune BEiT-3 either. L2-normalization happens inside BEiT3ForRetrieval.forward
(F.normalize on vision_cls/language_cls) — this class does not re-normalize.

Preprocessing mirrors microsoft/unilm/beit3/datasets.py eval path exactly:
- image: Resize((224, 224), BICUBIC) -> ToTensor -> Normalize(mean=0.5, std=0.5)
- text: XLMRobertaTokenizer -> [bos] + ids[:max_len-2] + [eos], padded to max_len

See docs/references.md §3.2 for setup context and PRD risk R1 (timeboxed 1 day;
falls back to a SigLIPEncoder behind the same Protocol if this is not done in time).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from timm.models import create_model
from torchvision import transforms
from transformers import XLMRobertaTokenizer

# noqa: F401 — importing this registers the timm model used by create_model() below
from app.encoders.vendor.beit3_upstream import modeling_retrieval  # noqa: F401
from app.encoders.vendor.beit3_upstream.utils import load_state_dict

_TIMM_MODEL_NAME = "beit3_base_patch16_224_retrieval"
_EMBEDDING_DIM = 768
_IMAGE_SIZE = 224
_MAX_TEXT_TOKENS = 64
_IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
_IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)


@dataclass(frozen=True)
class LocalTextEmbeddings:
    text: str
    tokens: tuple[str, ...]
    embeddings: np.ndarray


class BEiT3Encoder:
    """MultimodalEncoder implementation wrapping BEiT3ForRetrieval (beit3_base_itc)."""

    def __init__(
        self,
        checkpoint_path: str,
        sentencepiece_model_path: str,
        device: str = "cuda",
        batch_size: int = 16,
        model_id: str = "beit3_base_itc_patch16_224",
    ) -> None:
        self._device = torch.device(device)
        self._batch_size = batch_size
        self._model_id = model_id
        self._tokenizer = XLMRobertaTokenizer(sentencepiece_model_path)
        self._transform = transforms.Compose(
            [
                transforms.Resize(
                    (_IMAGE_SIZE, _IMAGE_SIZE), interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=_IMAGENET_INCEPTION_MEAN, std=_IMAGENET_INCEPTION_STD),
            ]
        )
        self._model = create_model(_TIMM_MODEL_NAME)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        load_state_dict(self._model, checkpoint.get("model", checkpoint))
        self._model.to(self._device).eval()

    @property
    def embedding_dim(self) -> int:
        return _EMBEDDING_DIM

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def preprocess_id(self) -> str:
        return f"beit3_resize{_IMAGE_SIZE}_bicubic_norm0.5_patch16_text{_MAX_TEXT_TOKENS}"

    def encode_image(self, images: Sequence[Image.Image]) -> np.ndarray:
        if not images:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        vectors = [self._encode_image_batch(batch) for batch in self._batched(images)]
        return np.concatenate(vectors, axis=0).astype(np.float32)

    def encode_image_patches(self, images: Sequence[Image.Image]) -> np.ndarray:
        """Return projected, L2-normalized local patch embeddings.

        This reuses the BEiT3 image forward path used for global retrieval,
        but takes encoder tokens after CLS and applies the retrieval
        `vision_head` to each token. No image patch files are created.
        """
        if not images:
            return np.empty((0, 0, self.embedding_dim), dtype=np.float32)
        vectors = [self._encode_image_patch_batch(batch) for batch in self._batched(images)]
        return np.concatenate(vectors, axis=0).astype(np.float32)

    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        vectors = [self._encode_text_batch(batch) for batch in self._batched(texts)]
        return np.concatenate(vectors, axis=0).astype(np.float32)

    def encode_text_tokens(self, texts: Sequence[str]) -> list[LocalTextEmbeddings]:
        """Return projected local token embeddings, excluding BOS/EOS/padding."""
        if not texts:
            return []
        outputs: list[LocalTextEmbeddings] = []
        for batch in self._batched(texts):
            outputs.extend(self._encode_text_token_batch(batch))
        return outputs

    def _batched(self, items: Sequence) -> list:
        return [items[i : i + self._batch_size] for i in range(0, len(items), self._batch_size)]

    def _encode_image_batch(self, images: Sequence[Image.Image]) -> np.ndarray:
        pixel_values = torch.stack([self._transform(image.convert("RGB")) for image in images])
        pixel_values = pixel_values.to(self._device)
        with torch.no_grad(), self._autocast():
            vision_cls, _ = self._model(image=pixel_values, only_infer=True)
        self._empty_cache()
        return vision_cls.float().cpu().numpy()

    def _encode_image_patch_batch(self, images: Sequence[Image.Image]) -> np.ndarray:
        pixel_values = torch.stack([self._transform(image.convert("RGB")) for image in images])
        pixel_values = pixel_values.to(self._device)
        with torch.no_grad(), self._autocast():
            outputs = self._model.beit3(
                textual_tokens=None,
                visual_tokens=pixel_values,
                text_padding_position=None,
            )
            patch_tokens = outputs["encoder_out"][:, 1:, :]
            patch_embeddings = self._model.vision_head(patch_tokens)
            patch_embeddings = F.normalize(patch_embeddings, dim=-1)
        self._empty_cache()
        return patch_embeddings.float().cpu().numpy()

    def _encode_text_batch(self, texts: Sequence[str]) -> np.ndarray:
        encoded = [self._tokenize_with_tokens(text) for text in texts]
        token_ids = torch.tensor(
            [item[0] for item in encoded], dtype=torch.long, device=self._device
        )
        padding_mask = torch.tensor(
            [item[1] for item in encoded], dtype=torch.long, device=self._device
        )
        with torch.no_grad(), self._autocast():
            _, language_cls = self._model(
                text_description=token_ids, padding_mask=padding_mask, only_infer=True
            )
        self._empty_cache()
        return language_cls.float().cpu().numpy()

    def _encode_text_token_batch(self, texts: Sequence[str]) -> list[LocalTextEmbeddings]:
        encoded = [self._tokenize_with_tokens(text) for text in texts]
        token_ids = torch.tensor(
            [item[0] for item in encoded], dtype=torch.long, device=self._device
        )
        padding_mask = torch.tensor(
            [item[1] for item in encoded], dtype=torch.long, device=self._device
        )
        with torch.no_grad(), self._autocast():
            outputs = self._model.beit3(
                textual_tokens=token_ids,
                visual_tokens=None,
                text_padding_position=padding_mask,
            )
            token_states = outputs["encoder_out"]
            projected = self._model.language_head(token_states)
            projected = F.normalize(projected, dim=-1)

        results: list[LocalTextEmbeddings] = []
        for row_index, text in enumerate(texts):
            valid_tokens = encoded[row_index][2]
            # Position 0 is BOS/CLS and position len(valid_tokens)+1 is EOS.
            local = projected[row_index, 1 : 1 + len(valid_tokens), :]
            results.append(
                LocalTextEmbeddings(
                    text=text,
                    tokens=tuple(valid_tokens),
                    embeddings=local.float().cpu().numpy().astype(np.float32),
                )
            )
        self._empty_cache()
        return results

    def _tokenize(self, text: str) -> tuple[list[int], list[int]]:
        ids, padding_mask, _ = self._tokenize_with_tokens(text)
        return ids, padding_mask

    def _tokenize_with_tokens(self, text: str) -> tuple[list[int], list[int], list[str]]:
        tokens = self._tokenizer.tokenize(text)
        ids = self._tokenizer.convert_tokens_to_ids(tokens)
        if len(ids) == 0:
            raise ValueError("text must contain at least one token")
        ids = ids[: _MAX_TEXT_TOKENS - 2]
        tokens = tokens[: _MAX_TEXT_TOKENS - 2]
        ids = [self._tokenizer.bos_token_id, *ids, self._tokenizer.eos_token_id]
        num_tokens = len(ids)
        padding_mask = [0] * num_tokens + [1] * (_MAX_TEXT_TOKENS - num_tokens)
        ids = ids + [self._tokenizer.pad_token_id] * (_MAX_TEXT_TOKENS - num_tokens)
        return ids, padding_mask, tokens

    def _autocast(self):
        return torch.autocast(
            device_type=self._device.type, dtype=torch.float16, enabled=self._device.type == "cuda"
        )

    def _empty_cache(self) -> None:
        if self._device.type == "cuda":
            torch.cuda.empty_cache()
