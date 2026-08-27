"""KIS FG-CLIP phrase-to-patch local retrieval.

The online KIS path uses precomputed 8x8 FG-CLIP dense image features from
disk. It never decodes images and never extracts patch vectors at request time.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from collections import OrderedDict, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.services.kis_scoring import KisConfig

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class LocalEmbeddingRef:
    keyframe_id: str
    video_id: str
    frame_idx: int


@dataclass(frozen=True)
class LocalEmbeddingBatch:
    candidate_ids: tuple[str, ...]
    features: np.ndarray
    missing_candidate_ids: tuple[str, ...]


@dataclass(frozen=True)
class LocalPhraseGroup:
    query_index: int
    phrases: tuple[str, ...]


@dataclass(frozen=True)
class LocalCandidateScore:
    local_score: float | None
    local_query_scores: tuple[float, ...]
    local_embedding_available: bool


@dataclass(frozen=True)
class LocalRerankOutput:
    scores_by_id: dict[str, LocalCandidateScore]
    phrase_encode_ms: float
    embedding_io_ms: float
    scoring_ms: float
    num_local_candidates: int
    num_missing_local_embeddings: int


@dataclass
class VideoLocalFiles:
    embeddings: np.ndarray
    row_by_frame_idx: dict[int, int]


class KISLocalEmbeddingStore:
    """Disk-backed lookup for precomputed per-keyframe local FG-CLIP features."""

    def __init__(
        self,
        *,
        dense_root: str | Path,
        grid_size: int,
        num_patches: int,
        embedding_dim: int,
        mmap_cache_size: int,
    ) -> None:
        root = Path(dense_root)
        self._root = root if root.is_absolute() else _REPO_ROOT / root
        self._grid_size = grid_size
        self._num_patches = num_patches
        self._embedding_dim = embedding_dim
        self._mmap_cache_size = mmap_cache_size
        self._video_dirs = self._read_manifest()
        self._metadata = self._read_metadata()
        self._cache: OrderedDict[str, VideoLocalFiles] = OrderedDict()
        self._warned_missing_videos: set[str] = set()
        self._warned_bad_videos: set[str] = set()

    @classmethod
    def from_config(cls, config: KisConfig) -> KISLocalEmbeddingStore:
        return cls(
            dense_root=config.local_dense_root,
            grid_size=config.local_grid_size,
            num_patches=config.local_num_patches,
            embedding_dim=config.local_embedding_dim,
            mmap_cache_size=config.local_mmap_cache_size,
        )

    @property
    def metadata(self) -> Mapping[str, Any]:
        return self._metadata

    @property
    def cache_video_ids(self) -> tuple[str, ...]:
        return tuple(self._cache.keys())

    def get(self, video_id: str, frame_idx: int) -> np.ndarray | None:
        files = self._open_video(video_id)
        if files is None:
            return None
        row = files.row_by_frame_idx.get(int(frame_idx))
        if row is None:
            return None
        return np.asarray(files.embeddings[row], dtype=np.float32)

    def load_video(self, video_id: str) -> VideoLocalFiles | None:
        """The whole video's dense array (mmap) + its frame_idx->row map.

        For TRAKE's B2 bucket rerank (`app/services/trake_service.py`), which
        scores every candidate frame of a video against every event at once --
        reading `N_frames` rows sequentially out of one open mmap is both
        cheaper and simpler than KIS's per-keyframe random-access `load_many`,
        and the video is going to be read almost in full either way (measured
        2026-08-27: MaxSim's winning frame lands as deep as rank 273-308 in
        the video's own global-cosine order, so a pre-filtered pool would
        just miss it).
        """
        return self._open_video(video_id)

    def load_many(self, refs: Sequence[LocalEmbeddingRef]) -> LocalEmbeddingBatch:
        if not refs:
            empty_embeddings = np.empty(
                (0, self._num_patches, self._embedding_dim), dtype=np.float32
            )
            return LocalEmbeddingBatch((), empty_embeddings, ())

        by_video: dict[str, list[LocalEmbeddingRef]] = defaultdict(list)
        for ref in refs:
            by_video[ref.video_id].append(ref)

        features_by_id: dict[str, np.ndarray] = {}
        missing: list[str] = []
        for video_id, video_refs in by_video.items():
            files = self._open_video(video_id)
            if files is None:
                missing.extend(ref.keyframe_id for ref in video_refs)
                continue
            for ref in video_refs:
                row = files.row_by_frame_idx.get(int(ref.frame_idx))
                if row is None:
                    missing.append(ref.keyframe_id)
                    continue
                features_by_id[ref.keyframe_id] = np.asarray(
                    files.embeddings[row], dtype=np.float32
                )

        ordered_ids = tuple(ref.keyframe_id for ref in refs if ref.keyframe_id in features_by_id)
        if ordered_ids:
            features = np.stack([features_by_id[keyframe_id] for keyframe_id in ordered_ids])
        else:
            features = np.empty((0, self._num_patches, self._embedding_dim), dtype=np.float32)
        return LocalEmbeddingBatch(
            candidate_ids=ordered_ids,
            features=features.astype(np.float32, copy=False),
            missing_candidate_ids=tuple(missing),
        )

    def _read_metadata(self) -> dict[str, Any]:
        path = self._root / "metadata.json"
        if not path.exists():
            return {}
        try:
            with path.open(encoding="utf-8") as f:
                return dict(json.load(f))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("KIS local retrieval: could not read metadata.json at %s: %s", path, exc)
            return {}

    def _read_manifest(self) -> dict[str, Path]:
        path = self._root / "manifest.csv"
        if not path.exists():
            return {}
        video_dirs: dict[str, Path] = {}
        try:
            with path.open(encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    video_id = _first_present(row, "video_id", "video", "id")
                    if not video_id:
                        continue
                    video_dirs[video_id] = self._manifest_video_dir(row, video_id)
        except OSError as exc:
            logger.warning("KIS local retrieval: could not read manifest.csv at %s: %s", path, exc)
        return video_dirs

    def _manifest_video_dir(self, row: Mapping[str, str | None], video_id: str) -> Path:
        for key in (
            "video_dir",
            "embedding_dir",
            "relative_dir",
            "dir",
            "path",
            "embeddings_path",
        ):
            value = (row.get(key) or "").strip()
            if not value:
                continue
            path = Path(value)
            if key == "embeddings_path":
                path = path.parent
            return path if path.is_absolute() else self._root / path
        batch = (row.get("batch") or row.get("collection") or "").strip()
        return self._root / batch / video_id if batch else self._inferred_video_dir(video_id)

    def _open_video(self, video_id: str) -> VideoLocalFiles | None:
        cached = self._cache.get(video_id)
        if cached is not None:
            self._cache.move_to_end(video_id)
            return cached

        video_dir = self._video_dirs.get(video_id, self._inferred_video_dir(video_id))
        embeddings_path = video_dir / "embeddings.npy"
        frame_idx_path = video_dir / "frame_idx.npy"
        if not embeddings_path.exists() or not frame_idx_path.exists():
            if video_id not in self._warned_missing_videos:
                logger.warning(
                    "KIS local retrieval: local embeddings missing for video_id=%s under %s",
                    video_id,
                    video_dir,
                )
                self._warned_missing_videos.add(video_id)
            return None

        try:
            embeddings = np.load(embeddings_path, mmap_mode="r")
            frame_indices = np.asarray(np.load(frame_idx_path), dtype=np.int64)
            self._validate_video_arrays(video_id, embeddings, frame_indices)
        except (OSError, ValueError) as exc:
            if video_id not in self._warned_bad_videos:
                logger.warning(
                    "KIS local retrieval: local embedding files are invalid for video_id=%s: %s",
                    video_id,
                    exc,
                )
                self._warned_bad_videos.add(video_id)
            return None

        files = VideoLocalFiles(
            embeddings=embeddings,
            row_by_frame_idx={int(frame_idx): row for row, frame_idx in enumerate(frame_indices)},
        )
        self._cache[video_id] = files
        self._cache.move_to_end(video_id)
        while len(self._cache) > self._mmap_cache_size:
            self._cache.popitem(last=False)
        return files

    def _validate_video_arrays(
        self, video_id: str, embeddings: np.ndarray, frame_indices: np.ndarray
    ) -> None:
        expected_tail = (self._num_patches, self._embedding_dim)
        if embeddings.ndim != 3 or tuple(embeddings.shape[1:]) != expected_tail:
            raise ValueError(
                f"{video_id} embeddings shape {embeddings.shape} != [N, {expected_tail[0]}, "
                f"{expected_tail[1]}]"
            )
        if frame_indices.ndim != 1:
            raise ValueError(f"{video_id} frame_idx.npy must be rank-1")
        if embeddings.shape[0] != frame_indices.shape[0]:
            raise ValueError(
                f"{video_id} embeddings rows {embeddings.shape[0]} != frame_idx rows "
                f"{frame_indices.shape[0]}"
            )

    def _inferred_video_dir(self, video_id: str) -> Path:
        batch = video_id.split("_", 1)[0]
        return self._root / batch / video_id


class LocalPhraseReranker:
    """Encodes manual phrase groups once and reranks candidates with MaxSim."""

    def __init__(self, *, encoder: Any, store: KISLocalEmbeddingStore, config: KisConfig) -> None:
        self._encoder = encoder
        self._store = store
        self._config = config

    def rerank(
        self,
        candidates: Sequence[LocalEmbeddingRef],
        phrase_groups: Sequence[LocalPhraseGroup],
    ) -> LocalRerankOutput:
        if not phrase_groups:
            return LocalRerankOutput({}, 0.0, 0.0, 0.0, 0, 0)

        phrase_start = time.perf_counter()
        phrase_embeddings_by_group = self._encode_phrase_groups(phrase_groups)
        phrase_encode_ms = (time.perf_counter() - phrase_start) * 1000

        scores_by_id: dict[str, LocalCandidateScore] = {}
        missing_ids: list[str] = []
        embedding_io_ms = 0.0
        scoring_ms = 0.0
        for start in range(0, len(candidates), self._config.local_score_batch_size):
            batch_refs = candidates[start : start + self._config.local_score_batch_size]
            io_start = time.perf_counter()
            batch = self._store.load_many(batch_refs)
            embedding_io_ms += (time.perf_counter() - io_start) * 1000
            missing_ids.extend(batch.missing_candidate_ids)
            if not batch.candidate_ids:
                continue

            scoring_start = time.perf_counter()
            query_scores = []
            for group in phrase_groups:
                phrase_embeddings = phrase_embeddings_by_group[group.query_index]
                query_scores.append(maxsim_query_scores(batch.features, phrase_embeddings))
            per_query = np.stack(query_scores, axis=1).astype(np.float32, copy=False)
            local_scores = per_query.mean(axis=1)
            scoring_ms += (time.perf_counter() - scoring_start) * 1000

            for row, keyframe_id in enumerate(batch.candidate_ids):
                scores_by_id[keyframe_id] = LocalCandidateScore(
                    local_score=float(local_scores[row]),
                    local_query_scores=tuple(float(score) for score in per_query[row]),
                    local_embedding_available=True,
                )

        for candidate in candidates:
            if candidate.keyframe_id not in scores_by_id:
                scores_by_id[candidate.keyframe_id] = LocalCandidateScore(
                    local_score=None,
                    local_query_scores=(),
                    local_embedding_available=False,
                )

        return LocalRerankOutput(
            scores_by_id=scores_by_id,
            phrase_encode_ms=phrase_encode_ms,
            embedding_io_ms=embedding_io_ms,
            scoring_ms=scoring_ms,
            num_local_candidates=sum(
                1 for score in scores_by_id.values() if score.local_embedding_available
            ),
            num_missing_local_embeddings=len(set(missing_ids)),
        )

    def _encode_phrase_groups(
        self, phrase_groups: Sequence[LocalPhraseGroup]
    ) -> dict[int, np.ndarray]:
        unique_phrases = list(
            dict.fromkeys(phrase for group in phrase_groups for phrase in group.phrases)
        )
        if hasattr(self._encoder, "encode_local_phrases"):
            encoded = self._encoder.encode_local_phrases(unique_phrases)
        else:
            encoded = self._encoder.encode_text(unique_phrases)
        embeddings_by_phrase = {
            phrase: embedding
            for phrase, embedding in zip(
                unique_phrases, _l2_normalize(np.asarray(encoded, dtype=np.float32)), strict=True
            )
        }
        return {
            group.query_index: np.asarray(
                [embeddings_by_phrase[phrase] for phrase in group.phrases], dtype=np.float32
            )
            for group in phrase_groups
        }


def normalize_local_phrase_groups(
    text_queries: Sequence[str],
    local_phrases_by_query: Sequence[Sequence[str]] | None,
) -> tuple[LocalPhraseGroup, ...]:
    groups: list[LocalPhraseGroup] = []
    for query_index, _ in enumerate(text_queries):
        raw_phrases = (
            local_phrases_by_query[query_index]
            if local_phrases_by_query is not None and query_index < len(local_phrases_by_query)
            else ()
        )
        phrases = tuple(_clean_phrase_list(raw_phrases))
        if phrases:
            groups.append(LocalPhraseGroup(query_index=query_index, phrases=phrases))
    return tuple(groups)


def maxsim_query_scores(patches: np.ndarray, phrase_embeddings: np.ndarray) -> np.ndarray:
    """Vectorized `[B, 64, D] x [M, D] -> [B]` phrase MaxSim mean."""

    patch_array = np.asarray(patches, dtype=np.float32)
    phrase_array = np.asarray(phrase_embeddings, dtype=np.float32)
    if patch_array.ndim != 3:
        raise ValueError(f"patches must be [B, P, D], got {patch_array.shape}")
    if phrase_array.ndim != 2 or phrase_array.shape[0] == 0:
        raise ValueError(f"phrase_embeddings must be non-empty [M, D], got {phrase_array.shape}")
    if patch_array.shape[2] != phrase_array.shape[1]:
        raise ValueError(
            f"embedding dimension mismatch: patches={patch_array.shape[2]} "
            f"phrases={phrase_array.shape[1]}"
        )

    try:
        import torch
    except ImportError:
        similarities = np.einsum("bpd,md->bmp", patch_array, phrase_array, optimize=True)
        return similarities.max(axis=2).mean(axis=1).astype(np.float32, copy=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        patch_tensor = torch.as_tensor(patch_array, dtype=torch.float32, device=device)
        phrase_tensor = torch.as_tensor(phrase_array, dtype=torch.float32, device=device)
        similarities = torch.einsum("bpd,md->bmp", patch_tensor, phrase_tensor)
        scores = similarities.max(dim=2).values.mean(dim=1)
    return scores.detach().cpu().numpy().astype(np.float32, copy=False)


def _clean_phrase_list(phrases: Sequence[str]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for phrase in phrases:
        stripped = str(phrase).strip()
        if not stripped:
            continue
        key = stripped.casefold()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(stripped)
    return cleaned


def _l2_normalize(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return (values / np.maximum(norms, 1e-12)).astype(np.float32, copy=False)


def _first_present(row: Mapping[str, str | None], *keys: str) -> str:
    for key in keys:
        value = (row.get(key) or "").strip()
        if value:
            return value
    return ""
