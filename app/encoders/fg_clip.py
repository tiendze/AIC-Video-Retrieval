"""FGClipEncoder backed by the local FG-CLIP Base ViT-B/16 checkpoint.

The repository uses FG-CLIP through Hugging Face's remote-code entrypoint in
`models/fgclip_base`. The wrapper keeps the same MultimodalEncoder contract as
the other encoders: all public vectors are float32 NumPy arrays and every row is
L2-normalized before it leaves the encoder.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

_DEFAULT_MODEL_PATH = "models/fgclip_base"
_DEFAULT_MODEL_ID = "fgclip_base"
_DEFAULT_IMAGE_SIZE = 224
_DEFAULT_MAX_TEXT_TOKENS = 77
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_HF_MODULES_CACHE = _REPO_ROOT / ".cache" / "huggingface" / "modules"
_DEPENDENCY_HINT = (
    "FG-CLIP requires torch, torchvision, transformers, pillow, safetensors, and einops. "
    'Install with: pip install -e ".[fg-clip]"'
)


@dataclass(frozen=True)
class LocalTextEmbeddings:
    text: str
    tokens: tuple[str, ...]
    embeddings: np.ndarray


class FGClipEncoder:
    """MultimodalEncoder implementation wrapping FG-CLIP Base ViT-B/16."""

    def __init__(
        self,
        model_path: str = _DEFAULT_MODEL_PATH,
        device: str = "cuda",
        batch_size: int = 16,
        model_id: str = _DEFAULT_MODEL_ID,
        image_size: int = _DEFAULT_IMAGE_SIZE,
        max_text_tokens: int = _DEFAULT_MAX_TEXT_TOKENS,
        embedding_dim: int | None = None,
    ) -> None:
        _configure_hf_modules_cache()
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on optional extra.
            raise ImportError(_DEPENDENCY_HINT) from exc

        self._torch = torch
        self._device = torch.device(device)
        self._batch_size = batch_size
        self._model_id = model_id
        self._image_size = image_size
        self._max_text_tokens = max_text_tokens
        self._model_path = str(model_path)
        self._checkpoint_id = _display_repo_path(Path(model_path))
        self._variant = "FG-CLIP Base ViT-B/16"

        try:
            self._model = (
                AutoModelForCausalLM.from_pretrained(self._model_path, trust_remote_code=True)
                .to(self._device)
                .eval()
            )
        except ImportError as exc:  # pragma: no cover - depends on optional extra.
            raise ImportError(_DEPENDENCY_HINT) from exc
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_path,
            clean_up_tokenization_spaces=False,
        )
        self._image_processor = AutoImageProcessor.from_pretrained(self._model_path)

        inferred_dim = int(getattr(self._model.config, "projection_dim", 0) or 0)
        if inferred_dim <= 0:
            raise ValueError("FG-CLIP checkpoint did not expose config.projection_dim")
        if embedding_dim is not None and int(embedding_dim) != inferred_dim:
            raise ValueError(
                f"FG-CLIP embedding_dim mismatch: config={embedding_dim} checkpoint={inferred_dim}"
            )
        self._embedding_dim = inferred_dim

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def encoder_kind(self) -> str:
        return "fg_clip"

    @property
    def checkpoint_id(self) -> str:
        return self._checkpoint_id

    @property
    def variant(self) -> str:
        return self._variant

    @property
    def preprocess_id(self) -> str:
        return f"fgclip_resize{self._image_size}_processor_patch16_text{self._max_text_tokens}"

    @property
    def max_text_tokens(self) -> int:
        """The tokenizer's `max_length` (77 for the short/`walk_short_pos`
        path this encoder always uses -- see the model card's "Long
        captions" note for the 248-token alternative, which this class does
        not wire up). `_tokenize` always calls with `truncation=True`, so any
        text over this length is silently cut; callers that need to warn
        about that should compare against this, not hardcode 77."""
        return self._max_text_tokens

    def encode_image(self, images: Sequence[Image.Image]) -> np.ndarray:
        if not images:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        vectors = [self._encode_image_batch(batch) for batch in self._batched(images)]
        return np.concatenate(vectors, axis=0).astype(np.float32)

    def encode_image_patches(self, images: Sequence[Image.Image]) -> np.ndarray:
        if not images:
            return np.empty((0, 0, self.embedding_dim), dtype=np.float32)
        vectors = [self._encode_image_patch_batch(batch) for batch in self._batched(images)]
        return np.concatenate(vectors, axis=0).astype(np.float32)

    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        vectors = [self._encode_text_batch(batch) for batch in self._batched(texts)]
        return np.concatenate(vectors, axis=0).astype(np.float32)

    def encode_local_phrases(self, phrases: Sequence[str]) -> np.ndarray:
        """Encode phrase-level visual concepts for 8x8 patch MaxSim scoring."""
        return self.encode_text(phrases)

    def count_text_tokens(self, texts: Sequence[str]) -> list[int]:
        """Real token count per text, WITHOUT the padding/truncation
        `_tokenize` applies for the encode path -- `_tokenize` always pads to
        `max_length` and truncates silently, so it cannot tell a caller
        whether a text was actually cut. Used to warn (not fix) callers whose
        text may exceed `max_text_tokens`, e.g. a long TRAKE event
        description (`app/services/trake_service.py`)."""
        if not texts:
            return []
        encoded = self._tokenizer(list(texts), truncation=False, padding=False)
        return [len(ids) for ids in encoded["input_ids"]]

    def encode_text_tokens(self, texts: Sequence[str]) -> list[LocalTextEmbeddings]:
        if not texts:
            return []
        outputs: list[LocalTextEmbeddings] = []
        for batch in self._batched(texts):
            outputs.extend(self._encode_text_token_batch(batch))
        return outputs

    def debug_tokenize(self, texts: Sequence[str]) -> Any:
        """Expose token ids for model tests without coupling callers to HF internals."""
        return self._tokenize(texts)["input_ids"]

    def _batched(self, items: Sequence) -> list:
        return [items[i : i + self._batch_size] for i in range(0, len(items), self._batch_size)]

    def _encode_image_batch(self, images: Sequence[Image.Image]) -> np.ndarray:
        pixel_values = self._preprocess_images(images)
        with self._torch.no_grad(), self._autocast():
            features = self._model.get_image_features(pixel_values=pixel_values)
            features = self._normalize_tensor(features)
        self._empty_cache()
        return features.float().cpu().numpy()

    def _encode_image_patch_batch(self, images: Sequence[Image.Image]) -> np.ndarray:
        pixel_values = self._preprocess_images(images)
        with self._torch.no_grad(), self._autocast():
            features = self._model.get_image_dense_features(pixel_values=pixel_values)
            features = self._normalize_tensor(features)
        self._empty_cache()
        return features.float().cpu().numpy()

    def _preprocess_images(self, images: Sequence[Image.Image]):
        prepared = [
            image.convert("RGB").resize(
                (self._image_size, self._image_size),
                Image.Resampling.BICUBIC,
            )
            for image in images
        ]
        pixel_values = self._image_processor.preprocess(prepared, return_tensors="pt")[
            "pixel_values"
        ]
        return pixel_values.to(self._device)

    def _encode_text_batch(self, texts: Sequence[str]) -> np.ndarray:
        tokenized = self._tokenize(texts)
        input_ids = tokenized["input_ids"].to(self._device)
        with self._torch.no_grad(), self._autocast():
            features = self._model.get_text_features(input_ids=input_ids, walk_short_pos=True)
            features = self._normalize_tensor(features)
        self._empty_cache()
        return features.float().cpu().numpy()

    def _encode_text_token_batch(self, texts: Sequence[str]) -> list[LocalTextEmbeddings]:
        tokenized = self._tokenize(texts, return_special_tokens_mask=True)
        input_ids = tokenized["input_ids"].to(self._device)
        attention_mask = tokenized["attention_mask"].cpu().numpy().astype(bool)
        special_mask = tokenized["special_tokens_mask"].cpu().numpy().astype(bool)

        with self._torch.no_grad(), self._autocast():
            outputs = self._model.text_model(
                input_ids=input_ids,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
                walk_short_pos=True,
            )
            projected = self._model.text_projection(outputs.last_hidden_state)
            projected = self._normalize_tensor(projected)

        results: list[LocalTextEmbeddings] = []
        projected_cpu = projected.float().cpu().numpy().astype(np.float32)
        input_ids_cpu = input_ids.cpu()
        for row_index, text in enumerate(texts):
            valid_positions = attention_mask[row_index] & ~special_mask[row_index]
            valid_ids = input_ids_cpu[row_index, valid_positions].tolist()
            tokens = tuple(
                self._clean_token(token)
                for token in self._tokenizer.convert_ids_to_tokens(valid_ids)
            )
            results.append(
                LocalTextEmbeddings(
                    text=text,
                    tokens=tokens,
                    embeddings=projected_cpu[row_index, valid_positions, :],
                )
            )
        self._empty_cache()
        return results

    def _tokenize(self, texts: Sequence[str], *, return_special_tokens_mask: bool = False):
        return self._tokenizer(
            list(texts),
            max_length=self._max_text_tokens,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_special_tokens_mask=return_special_tokens_mask,
        )

    @staticmethod
    def _clean_token(token: str) -> str:
        return token.replace("</w>", "")

    def _normalize_tensor(self, tensor):
        return tensor / tensor.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)

    def _autocast(self):
        return self._torch.autocast(
            device_type=self._device.type,
            dtype=self._torch.float16,
            enabled=self._device.type == "cuda",
        )

    def _empty_cache(self) -> None:
        if self._device.type == "cuda":
            self._torch.cuda.empty_cache()


def _configure_hf_modules_cache() -> None:
    cache_path = os.environ.setdefault("HF_MODULES_CACHE", str(_DEFAULT_HF_MODULES_CACHE))
    try:
        import transformers.dynamic_module_utils as dynamic_module_utils
        import transformers.utils.hub as transformers_hub
    except ImportError:
        return

    dynamic_module_utils.HF_MODULES_CACHE = cache_path
    transformers_hub.HF_MODULES_CACHE = cache_path


def _display_repo_path(path: Path) -> str:
    try:
        display = path.resolve().relative_to(_REPO_ROOT.resolve())
    except ValueError:
        display = path
    return str(display).replace("\\", "/")
