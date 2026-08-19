"""Attribute-object binding reranking helpers for KIS Stage 2.

ABE uses the active vision-language encoder in two ways that the normal global
retrieval path does not expose: projected image patch tokens and projected
local text tokens. The implementation is intentionally isolated here so the
Text/OCR rerank math remains unchanged and Milvus continues to store only
global image vectors.
"""

from __future__ import annotations

import math
import os
import re
import tempfile
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app.services.kis_scoring import (
    AbePair,
    AbePairScore,
    AbePatchHeatmap,
    AbePatchHit,
    KisConfig,
    KisQuery,
    RankedScore,
    rank_percentiles,
)

_WHITESPACE_RE = re.compile(r"\s+")
_DEFAULT_CREATED_BY = "scripts/build_abe_concept_bank.py"
_L2_ATOL = 2e-3


def _normalize_metadata_identifier(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_")


def _normalize_metadata_path(value: Any) -> str:
    return str(value).strip().replace("\\", "/").lower()


def _metadata_truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


class AbeError(Exception):
    """Raised for expected ABE setup/runtime failures."""


@dataclass(frozen=True)
class AbePairAssignment:
    attribute: str
    object: str
    positive_attributes: tuple[str, ...]
    negative_attributes: tuple[str, ...]


@dataclass(frozen=True)
class AbeNearestConcept:
    concept: str
    index: int
    similarity: float


@dataclass(frozen=True)
class LocalAlignmentResult:
    score: float
    top_patches: list[AbePatchHit]


@dataclass(frozen=True)
class AbeQueryComponents:
    pairs: list[AbePairScore]
    base_local: float
    refined_local: float
    binding_delta: float
    local_score: float
    global_abe: float
    raw_score: float
    patch_heatmap: AbePatchHeatmap | None


@dataclass(frozen=True)
class AbeRankedQuery:
    components_by_id: dict[str, AbeQueryComponents]
    rankings: dict[str, RankedScore]


@dataclass(frozen=True)
class _TextTokenEmbeddings:
    tokens: tuple[str, ...]
    embeddings: np.ndarray


@dataclass(frozen=True)
class _RefinedPairEmbeddings:
    attribute: str
    object: str
    positive_attributes: tuple[str, ...]
    negative_attributes: tuple[str, ...]
    attribute_embedding: np.ndarray
    object_embedding: np.ndarray


@dataclass(frozen=True)
class _PreparedAbeQuery:
    assignments: list[AbePairAssignment]
    pair_scores: list[AbePairScore]
    group_text: str
    global_embedding: np.ndarray
    base_embeddings: np.ndarray
    base_labels: list[str]
    refined_embeddings: np.ndarray
    refined_labels: list[str]


def normalize_abe_text(text: str) -> str:
    normalized = _WHITESPACE_RE.sub(" ", text.strip().lower())
    return normalized


def normalize_abe_pairs(pairs: Sequence[AbePair]) -> tuple[AbePair, ...]:
    normalized: list[AbePair] = []
    for pair in pairs:
        attribute = normalize_abe_text(pair.attribute)
        object_text = normalize_abe_text(pair.object)
        if attribute and object_text:
            normalized.append(AbePair(attribute=attribute, object=object_text))
    return tuple(normalized)


def assign_abe_pair_attributes(pairs: Sequence[AbePair]) -> list[AbePairAssignment]:
    normalized = normalize_abe_pairs(pairs)
    assignments: list[AbePairAssignment] = []
    for index, pair in enumerate(normalized):
        negatives = tuple(
            other.attribute
            for other_index, other in enumerate(normalized)
            if other_index != index and other.attribute
        )
        assignments.append(
            AbePairAssignment(
                attribute=pair.attribute,
                object=pair.object,
                positive_attributes=(pair.attribute,),
                negative_attributes=negatives,
            )
        )
    return assignments


def load_concept_strings(path: Path) -> tuple[list[str], int]:
    raw_concepts = path.read_text(encoding="utf-8").splitlines()
    return deduplicate_concepts(raw_concepts)


def deduplicate_concepts(concepts: Sequence[str]) -> tuple[list[str], int]:
    seen: set[str] = set()
    result: list[str] = []
    duplicates_removed = 0
    for concept in concepts:
        normalized = normalize_abe_text(concept)
        if not normalized:
            continue
        if normalized in seen:
            duplicates_removed += 1
            continue
        seen.add(normalized)
        result.append(normalized)
    return result, duplicates_removed


def l2_normalize(array: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    norms = np.linalg.norm(values, axis=axis, keepdims=True)
    return values / np.maximum(norms, eps)


def validate_l2_normalized(embeddings: np.ndarray, *, atol: float = _L2_ATOL) -> None:
    if embeddings.size == 0:
        raise ValueError("embeddings must not be empty")
    if not np.isfinite(embeddings).all():
        raise ValueError("embeddings contain NaN or Inf")
    norms = np.linalg.norm(embeddings.astype(np.float32), axis=-1)
    if not np.allclose(norms, 1.0, atol=atol):
        raise ValueError("embeddings are not L2-normalized")


def build_concept_bank_payload(
    concepts: Sequence[str],
    encoder: Any,
    *,
    checkpoint: str,
    model_id: str,
    batch_size: int,
    dtype: str = "float16",
    encoder_kind: str | None = None,
    variant: str | None = None,
    source_vocab: str | None = None,
    created_by: str = _DEFAULT_CREATED_BY,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    if dtype not in {"float16", "float32"}:
        raise ValueError("dtype must be float16 or float32")
    unique_concepts, duplicates_removed = deduplicate_concepts(concepts)
    if not unique_concepts:
        raise ValueError("concept list is empty after normalization")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    encoded_batches: list[np.ndarray] = []
    total = len(unique_concepts)
    for start in range(0, total, batch_size):
        batch = unique_concepts[start : start + batch_size]
        batch_vectors = np.asarray(encoder.encode_text(batch), dtype=np.float32)
        encoded_batches.append(batch_vectors)
        if progress is not None:
            progress(min(start + len(batch), total), total)

    embeddings = np.concatenate(encoded_batches, axis=0)
    if embeddings.shape != (len(unique_concepts), int(encoder.embedding_dim)):
        raise ValueError(
            "encoded concept shape mismatch: "
            f"got {embeddings.shape}, expected {(len(unique_concepts), encoder.embedding_dim)}"
        )
    embeddings = l2_normalize(embeddings)
    validate_l2_normalized(embeddings)
    output_dtype = np.float16 if dtype == "float16" else np.float32
    saved_embeddings = embeddings.astype(output_dtype)
    validate_l2_normalized(saved_embeddings)

    return {
        "concepts": unique_concepts,
        "embeddings": saved_embeddings,
        "metadata": {
            "encoder": encoder_kind or type(encoder).__name__,
            "variant": variant,
            "checkpoint": checkpoint,
            "model_id": model_id,
            "embedding_dim": int(encoder.embedding_dim),
            "dtype": dtype,
            "normalization": "l2",
            "normalized": True,
            "source_vocab": source_vocab,
            "created_by": created_by,
            "duplicates_removed": duplicates_removed,
            "num_concepts": len(unique_concepts),
        },
    }


def save_concept_bank(payload: Mapping[str, Any], output_path: Path) -> None:
    torch = _import_torch()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=str(output_path.parent)
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        payload_to_save = dict(payload)
        embeddings = payload_to_save["embeddings"]
        if not hasattr(embeddings, "detach"):
            embeddings = torch.from_numpy(np.asarray(embeddings))
        payload_to_save["embeddings"] = embeddings
        torch.save(payload_to_save, tmp_path)
        os.replace(tmp_path, output_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _import_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional extra.
        raise AbeError("ABE concept-bank serialization requires torch") from exc
    return torch


class AbeConceptBank:
    def __init__(
        self,
        concepts: Sequence[str],
        embeddings: np.ndarray,
        metadata: Mapping[str, Any],
    ) -> None:
        if len(concepts) != embeddings.shape[0]:
            raise ValueError("concept count does not match embedding rows")
        self.concepts = list(concepts)
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.metadata = dict(metadata)
        validate_l2_normalized(self.embeddings)

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        expected_model_id: str | None = None,
        expected_embedding_dim: int | None = None,
        expected_encoder_kind: str | None = None,
        expected_checkpoint: str | None = None,
    ) -> AbeConceptBank:
        embeddings = payload["embeddings"]
        if hasattr(embeddings, "detach"):
            embeddings = embeddings.detach().cpu().numpy()
        bank = cls(
            concepts=list(payload["concepts"]),
            embeddings=np.asarray(embeddings),
            metadata=dict(payload.get("metadata", {})),
        )
        bank.validate_compatibility(
            expected_model_id=expected_model_id,
            expected_embedding_dim=expected_embedding_dim,
            expected_encoder_kind=expected_encoder_kind,
            expected_checkpoint=expected_checkpoint,
        )
        return bank

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        expected_model_id: str | None = None,
        expected_embedding_dim: int | None = None,
        expected_encoder_kind: str | None = None,
        expected_checkpoint: str | None = None,
    ) -> AbeConceptBank:
        torch = _import_torch()
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return cls.from_payload(
            payload,
            expected_model_id=expected_model_id,
            expected_embedding_dim=expected_embedding_dim,
            expected_encoder_kind=expected_encoder_kind,
            expected_checkpoint=expected_checkpoint,
        )

    def validate_compatibility(
        self,
        *,
        expected_model_id: str | None,
        expected_embedding_dim: int | None,
        expected_encoder_kind: str | None = None,
        expected_checkpoint: str | None = None,
    ) -> None:
        if (
            expected_embedding_dim is not None
            and self.embeddings.shape[1] != expected_embedding_dim
        ):
            raise ValueError(
                "ABE concept-bank embedding_dim mismatch: "
                f"bank={self.embeddings.shape[1]} active={expected_embedding_dim}"
            )
        metadata_dim = self.metadata.get("embedding_dim")
        if metadata_dim is not None and expected_embedding_dim is not None:
            if int(metadata_dim) != expected_embedding_dim:
                raise ValueError(
                    "ABE concept-bank metadata embedding_dim mismatch: "
                    f"bank={metadata_dim} active={expected_embedding_dim}"
                )

        bank_encoder = self.metadata.get("encoder")
        if expected_encoder_kind is not None:
            if bank_encoder is None:
                warnings.warn(
                    "ABE concept bank has no encoder metadata; verify it was built with "
                    f"{expected_encoder_kind!r}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            elif _normalize_metadata_identifier(bank_encoder) != _normalize_metadata_identifier(
                expected_encoder_kind
            ):
                raise ValueError(
                    "ABE concept-bank encoder mismatch: "
                    f"bank={bank_encoder!r} active={expected_encoder_kind!r}"
                )

        normalization = self.metadata.get("normalization")
        if normalization is not None and str(normalization).lower() != "l2":
            raise ValueError(
                "ABE concept-bank normalization mismatch: "
                f"bank={normalization!r} active='l2'"
            )
        normalized = self.metadata.get("normalized")
        if normalized is not None and not _metadata_truthy(normalized):
            raise ValueError("ABE concept-bank metadata says embeddings are not normalized")
        dtype = self.metadata.get("dtype")
        if dtype is not None and str(dtype) not in {"float16", "float32"}:
            raise ValueError(f"ABE concept-bank dtype is unsupported: {dtype!r}")

        if expected_model_id is None:
            return
        bank_model_id = self.metadata.get("model_id")
        if bank_model_id is None:
            warnings.warn(
                "ABE concept bank has no model_id metadata; verify it was built with the "
                f"active encoder {expected_model_id!r}",
                RuntimeWarning,
                stacklevel=2,
            )
        elif str(bank_model_id) != expected_model_id:
            raise ValueError(
                "ABE concept-bank model_id mismatch: "
                f"bank={bank_model_id!r} active={expected_model_id!r}"
            )

        if expected_checkpoint is None:
            return
        bank_checkpoint = self.metadata.get("checkpoint")
        if bank_checkpoint is None:
            warnings.warn(
                "ABE concept bank has no checkpoint metadata; verify it was built with "
                f"{expected_checkpoint!r}",
                RuntimeWarning,
                stacklevel=2,
            )
            return
        if _normalize_metadata_path(bank_checkpoint) != _normalize_metadata_path(
            expected_checkpoint
        ):
            raise ValueError(
                "ABE concept-bank checkpoint mismatch: "
                f"bank={bank_checkpoint!r} active={expected_checkpoint!r}"
            )


class AbeConceptLookup:
    def __init__(self, concept_bank: AbeConceptBank, encoder: Any, top_p: int) -> None:
        if top_p < 1:
            raise ValueError("top_p must be positive")
        self._concept_bank = concept_bank
        self._encoder = encoder
        self._top_p = top_p
        self._cache: dict[str, list[AbeNearestConcept]] = {}

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    def nearest_concepts(self, object_text: str) -> list[AbeNearestConcept]:
        normalized = normalize_abe_text(object_text)
        if not normalized:
            raise ValueError("object_text must not be blank")
        cached = self._cache.get(normalized)
        if cached is not None:
            return cached

        object_embedding = np.asarray(self._encoder.encode_text([normalized])[0], dtype=np.float32)
        object_embedding = l2_normalize(object_embedding)
        similarities = self._concept_bank.embeddings @ object_embedding
        top_p = min(self._top_p, similarities.shape[0])
        if top_p == similarities.shape[0]:
            top_indices = np.argsort(-similarities)
        else:
            partitioned = np.argpartition(-similarities, top_p - 1)[:top_p]
            top_indices = partitioned[np.argsort(-similarities[partitioned])]
        result = [
            AbeNearestConcept(
                concept=self._concept_bank.concepts[int(index)],
                index=int(index),
                similarity=float(similarities[int(index)]),
            )
            for index in top_indices[:top_p]
        ]
        self._cache[normalized] = result
        return result


class AbePatchCache:
    def __init__(self, dtype: str = "float16") -> None:
        if dtype not in {"float16", "float32"}:
            raise ValueError("dtype must be float16 or float32")
        self._dtype = np.float16 if dtype == "float16" else np.float32
        self._signature: tuple[str, str, tuple[str, ...]] | None = None
        self._patches_by_id: dict[str, np.ndarray] = {}

    @property
    def cache_size(self) -> int:
        return len(self._patches_by_id)

    def get_or_encode(
        self,
        *,
        candidate_ids: Sequence[str],
        image_paths_by_id: Mapping[str, Path],
        encoder: Any,
        batch_size: int,
    ) -> dict[str, np.ndarray]:
        if not hasattr(encoder, "encode_image_patches"):
            raise AbeError("ABE requires an encoder with encode_image_patches support")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        signature = (
            str(getattr(encoder, "model_id", "")),
            str(getattr(encoder, "preprocess_id", "")),
            tuple(candidate_ids),
        )
        if self._signature == signature and all(
            keyframe_id in self._patches_by_id for keyframe_id in candidate_ids
        ):
            return {keyframe_id: self._patches_by_id[keyframe_id] for keyframe_id in candidate_ids}

        self._signature = signature
        self._patches_by_id = {}
        missing_paths = [
            keyframe_id for keyframe_id in candidate_ids if keyframe_id not in image_paths_by_id
        ]
        if missing_paths:
            raise AbeError(
                "ABE requires image paths for every candidate; "
                f"missing ids include {missing_paths[:5]}"
            )
        for start in range(0, len(candidate_ids), batch_size):
            batch_ids = list(candidate_ids[start : start + batch_size])
            images = [_load_image(image_paths_by_id[keyframe_id]) for keyframe_id in batch_ids]
            batch_patches = np.asarray(encoder.encode_image_patches(images), dtype=np.float32)
            _validate_patch_batch(batch_patches, len(batch_ids), int(encoder.embedding_dim))
            for keyframe_id, patches in zip(batch_ids, batch_patches, strict=True):
                self._patches_by_id[keyframe_id] = patches.astype(self._dtype)

        return {keyframe_id: self._patches_by_id[keyframe_id] for keyframe_id in candidate_ids}


def _load_image(path: Path) -> Image.Image:
    if not path.exists():
        raise AbeError(f"ABE source image missing: {path}")
    with Image.open(path) as image:
        return image.convert("RGB")


def _validate_patch_batch(patches: np.ndarray, expected_batch: int, embedding_dim: int) -> None:
    if patches.ndim != 3:
        raise ValueError(f"patch embeddings must be rank-3, got shape {patches.shape}")
    if patches.shape[0] != expected_batch:
        raise ValueError(
            f"patch batch size mismatch: got {patches.shape[0]}, expected {expected_batch}"
        )
    if patches.shape[1] < 1:
        raise ValueError("patch embeddings must contain at least one patch")
    if patches.shape[2] != embedding_dim:
        raise ValueError(
            f"patch embedding_dim mismatch: got {patches.shape[2]}, expected {embedding_dim}"
        )
    validate_l2_normalized(patches)


def local_alignment_score(
    text_embeddings: np.ndarray,
    patch_embeddings: np.ndarray,
    *,
    patch_top_k: int,
    labels: Sequence[str] | None = None,
) -> LocalAlignmentResult:
    text = np.asarray(text_embeddings, dtype=np.float32)
    patches = np.asarray(patch_embeddings, dtype=np.float32)
    if text.ndim != 2 or patches.ndim != 2:
        raise ValueError("local alignment expects text MxD and patches PxD")
    if text.shape[0] == 0:
        raise ValueError("local alignment requires at least one text embedding")
    if patches.shape[0] == 0:
        raise ValueError("local alignment requires at least one image patch")
    if text.shape[1] != patches.shape[1]:
        raise ValueError(
            f"local alignment dim mismatch: text={text.shape[1]} patches={patches.shape[1]}"
        )
    if patch_top_k < 1:
        raise ValueError("patch_top_k must be positive")

    similarities = text @ patches.T
    top_k = min(patch_top_k, patches.shape[0])
    partitioned = np.argpartition(-similarities, top_k - 1, axis=1)[:, :top_k]
    top_scores = np.take_along_axis(similarities, partitioned, axis=1)
    order = np.argsort(-top_scores, axis=1)
    top_indices = np.take_along_axis(partitioned, order, axis=1)
    sorted_scores = np.take_along_axis(top_scores, order, axis=1)

    per_token_scores = sorted_scores.mean(axis=1)
    top_patches: list[AbePatchHit] = []
    for token_index in range(text.shape[0]):
        label = labels[token_index] if labels is not None and token_index < len(labels) else ""
        for score, patch_index in zip(
            sorted_scores[token_index], top_indices[token_index], strict=True
        ):
            top_patches.append(
                AbePatchHit(label=label, patch_index=int(patch_index), score=float(score))
            )

    return LocalAlignmentResult(score=float(per_token_scores.mean()), top_patches=top_patches)


def infer_patch_grid(patch_count: int) -> tuple[int, int]:
    if patch_count < 1:
        raise ValueError("patch_count must be positive")
    root = int(math.sqrt(patch_count))
    if root * root == patch_count:
        return root, root
    width = math.ceil(math.sqrt(patch_count))
    return width, math.ceil(patch_count / width)


def abe_raw_score(
    *,
    base_local: float,
    refined_local: float,
    global_abe: float,
    global_weight: float,
) -> tuple[float, float, float]:
    if not 0 <= global_weight <= 1:
        raise ValueError("global_weight must be in [0, 1]")
    binding_delta = abs(refined_local - base_local)
    local_score = refined_local + binding_delta
    raw_score = (1.0 - global_weight) * local_score + global_weight * global_abe
    return float(binding_delta), float(local_score), float(raw_score)


class AbeScorer:
    def __init__(
        self,
        *,
        encoder: Any,
        concept_bank: AbeConceptBank,
        config: KisConfig,
    ) -> None:
        if not hasattr(encoder, "encode_text_tokens"):
            raise AbeError("ABE requires an encoder with encode_text_tokens support")
        self._encoder = encoder
        self._concept_bank = concept_bank
        self._config = config
        self._concept_lookup = AbeConceptLookup(
            concept_bank=concept_bank,
            encoder=encoder,
            top_p=config.abe_concept_top_p,
        )
        self._global_text_cache: dict[str, np.ndarray] = {}
        self._token_text_cache: dict[str, _TextTokenEmbeddings] = {}
        self._refined_cache: dict[tuple[str, str, tuple[str, ...]], _RefinedPairEmbeddings] = {}

    def score_queries(
        self,
        *,
        candidate_ids: Sequence[str],
        queries: Sequence[KisQuery],
        image_paths_by_id: Mapping[str, Path],
        global_image_vectors_by_id: Mapping[str, np.ndarray],
        patch_cache: AbePatchCache,
    ) -> dict[int, AbeRankedQuery]:
        if not candidate_ids:
            raise ValueError("candidate_ids must not be empty")
        patches_by_id = patch_cache.get_or_encode(
            candidate_ids=candidate_ids,
            image_paths_by_id=image_paths_by_id,
            encoder=self._encoder,
            batch_size=self._config.abe_patch_batch_size,
        )

        ranked_queries: dict[int, AbeRankedQuery] = {}
        for query in queries:
            prepared = self._prepare_query(query)
            components_by_id: dict[str, AbeQueryComponents] = {}
            raw_scores_by_id: dict[str, float] = {}
            for keyframe_id in candidate_ids:
                image_patches = np.asarray(patches_by_id[keyframe_id], dtype=np.float32)
                base = local_alignment_score(
                    prepared.base_embeddings,
                    image_patches,
                    patch_top_k=self._config.abe_patch_top_k,
                    labels=prepared.base_labels,
                )
                refined = local_alignment_score(
                    prepared.refined_embeddings,
                    image_patches,
                    patch_top_k=self._config.abe_patch_top_k,
                    labels=prepared.refined_labels,
                )
                image_vector = global_image_vectors_by_id.get(keyframe_id)
                global_score = (
                    _cosine_similarity(
                        prepared.global_embedding,
                        np.asarray(image_vector, dtype=np.float32),
                    )
                    if image_vector is not None
                    else -1.0
                )
                binding_delta, local_score, raw_score = abe_raw_score(
                    base_local=base.score,
                    refined_local=refined.score,
                    global_abe=global_score,
                    global_weight=self._config.abe_global_weight,
                )
                patch_count = image_patches.shape[0]
                grid_width, grid_height = infer_patch_grid(patch_count)
                components = AbeQueryComponents(
                    pairs=prepared.pair_scores,
                    base_local=base.score,
                    refined_local=refined.score,
                    binding_delta=binding_delta,
                    local_score=local_score,
                    global_abe=global_score,
                    raw_score=raw_score,
                    patch_heatmap=AbePatchHeatmap(
                        patch_count=patch_count,
                        grid_width=grid_width,
                        grid_height=grid_height,
                        base_top_patches=base.top_patches,
                        refined_top_patches=refined.top_patches,
                    ),
                )
                components_by_id[keyframe_id] = components
                raw_scores_by_id[keyframe_id] = raw_score

            ranked_queries[query.query_index] = AbeRankedQuery(
                components_by_id=components_by_id,
                rankings=rank_percentiles(raw_scores_by_id, candidate_ids),
            )
        return ranked_queries

    def _prepare_query(self, query: KisQuery) -> _PreparedAbeQuery:
        assignments = assign_abe_pair_attributes(query.pairs)
        if not assignments:
            raise ValueError("ABE query requires at least one valid pair")
        group_text = ", ".join(
            f"{assignment.attribute} {assignment.object}" for assignment in assignments
        )
        global_embedding = self._encode_global_text(group_text)
        base_tokens = self._encode_text_tokens(group_text)
        refined_pairs = [self._refined_pair_embeddings(assignment) for assignment in assignments]
        refined_embeddings = np.asarray(
            [
                embedding
                for pair in refined_pairs
                for embedding in (pair.attribute_embedding, pair.object_embedding)
            ],
            dtype=np.float32,
        )
        refined_labels = [
            label
            for pair in refined_pairs
            for label in (f"{pair.attribute} -> {pair.object}", pair.object)
        ]
        pair_scores = [
            AbePairScore(
                attribute=assignment.attribute,
                object=assignment.object,
                positive_attributes=list(assignment.positive_attributes),
                negative_attributes=list(assignment.negative_attributes),
            )
            for assignment in assignments
        ]
        return _PreparedAbeQuery(
            assignments=assignments,
            pair_scores=pair_scores,
            group_text=group_text,
            global_embedding=global_embedding,
            base_embeddings=base_tokens.embeddings,
            base_labels=list(base_tokens.tokens),
            refined_embeddings=refined_embeddings,
            refined_labels=refined_labels,
        )

    def _refined_pair_embeddings(self, assignment: AbePairAssignment) -> _RefinedPairEmbeddings:
        key = (
            normalize_abe_text(assignment.attribute),
            normalize_abe_text(assignment.object),
            tuple(sorted(normalize_abe_text(attr) for attr in assignment.negative_attributes)),
        )
        cached = self._refined_cache.get(key)
        if cached is not None:
            return cached

        attribute_embedding = self._encode_global_text(assignment.attribute)
        object_embedding = self._encode_global_text(assignment.object)
        nearest = self._concept_lookup.nearest_concepts(assignment.object)
        concept_indices = [concept.index for concept in nearest]
        concept_embeddings = self._concept_bank.embeddings[concept_indices]
        positive_phrases = [f"{assignment.attribute} {concept.concept}" for concept in nearest]
        positive_embeddings = self._encode_global_text_batch(positive_phrases)
        b_plus = (positive_embeddings - concept_embeddings).mean(axis=0)

        if assignment.negative_attributes:
            negative_phrases = [
                f"{negative} {concept.concept}"
                for negative in assignment.negative_attributes
                for concept in nearest
            ]
            negative_embeddings = self._encode_global_text_batch(negative_phrases)
            repeated_concepts = np.tile(
                concept_embeddings, (len(assignment.negative_attributes), 1)
            )
            b_minus = (negative_embeddings - repeated_concepts).mean(axis=0)
        else:
            b_minus = np.zeros_like(object_embedding)

        refined_object = l2_normalize(object_embedding + b_plus - b_minus)
        refined_attribute = l2_normalize(attribute_embedding + object_embedding)
        result = _RefinedPairEmbeddings(
            attribute=assignment.attribute,
            object=assignment.object,
            positive_attributes=assignment.positive_attributes,
            negative_attributes=assignment.negative_attributes,
            attribute_embedding=refined_attribute,
            object_embedding=refined_object,
        )
        self._refined_cache[key] = result
        return result

    def _encode_global_text(self, text: str) -> np.ndarray:
        normalized = normalize_abe_text(text)
        cached = self._global_text_cache.get(normalized)
        if cached is not None:
            return cached
        embedding = np.asarray(self._encoder.encode_text([normalized])[0], dtype=np.float32)
        embedding = l2_normalize(embedding)
        self._global_text_cache[normalized] = embedding
        return embedding

    def _encode_global_text_batch(self, texts: Sequence[str]) -> np.ndarray:
        normalized_texts = [
            normalized for text in texts if (normalized := normalize_abe_text(text))
        ]
        uncached = [text for text in normalized_texts if text not in self._global_text_cache]
        if uncached:
            embeddings = np.asarray(self._encoder.encode_text(uncached), dtype=np.float32)
            embeddings = l2_normalize(embeddings)
            for text, embedding in zip(uncached, embeddings, strict=True):
                self._global_text_cache[text] = embedding
        return np.asarray(
            [self._global_text_cache[text] for text in normalized_texts], dtype=np.float32
        )

    def _encode_text_tokens(self, text: str) -> _TextTokenEmbeddings:
        normalized = normalize_abe_text(text)
        cached = self._token_text_cache.get(normalized)
        if cached is not None:
            return cached
        encoded = self._encoder.encode_text_tokens([normalized])[0]
        embeddings = np.asarray(encoded.embeddings, dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[0] == 0:
            raise ValueError("ABE text token encoding returned no valid tokens")
        validate_l2_normalized(embeddings)
        result = _TextTokenEmbeddings(tokens=tuple(encoded.tokens), embeddings=embeddings)
        self._token_text_cache[normalized] = result
        return result


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    if left.ndim != 1 or right.ndim != 1 or right.size == 0:
        return -1.0
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        return -1.0
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= 0 or right_norm <= 0:
        return -1.0
    return float(np.dot(left, right) / (left_norm * right_norm))
