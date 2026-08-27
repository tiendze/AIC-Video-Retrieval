from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.services.kis_local_retrieval import (
    KISLocalEmbeddingStore,
    LocalEmbeddingRef,
    LocalPhraseGroup,
    LocalPhraseReranker,
    maxsim_query_scores,
    normalize_local_phrase_groups,
)
from app.services.kis_scoring import KisConfig


class _PhraseEncoder:
    embedding_dim = 3
    model_id = "fake-encoder"
    encoder_kind = "fg_clip"

    def __init__(self, embeddings_by_phrase: dict[str, np.ndarray]) -> None:
        self._embeddings_by_phrase = embeddings_by_phrase
        self.calls: list[list[str]] = []

    def encode_local_phrases(self, phrases: list[str]) -> np.ndarray:
        self.calls.append(list(phrases))
        return np.asarray([self._embeddings_by_phrase[phrase] for phrase in phrases])


def _write_video(
    root: Path,
    video_id: str,
    frame_indices: list[int],
    embeddings: np.ndarray,
) -> Path:
    video_dir = root / video_id.split("_", 1)[0] / video_id
    video_dir.mkdir(parents=True)
    np.save(video_dir / "embeddings.npy", embeddings.astype(np.float16))
    np.save(video_dir / "frame_idx.npy", np.asarray(frame_indices, dtype=np.int64))
    (video_dir / "filenames.txt").write_text("\n".join(f"{idx}.jpg" for idx in frame_indices))
    (video_dir / "done.json").write_text('{"done": true}', encoding="utf-8")
    return video_dir


def _store(root: Path, cache_size: int = 2) -> KISLocalEmbeddingStore:
    return KISLocalEmbeddingStore(
        dense_root=root,
        grid_size=2,
        num_patches=4,
        embedding_dim=3,
        mmap_cache_size=cache_size,
    )


def test_local_embedding_store_maps_video_frame_to_correct_row(tmp_path: Path) -> None:
    embeddings = np.arange(2 * 4 * 3, dtype=np.float32).reshape(2, 4, 3)
    _write_video(tmp_path, "L21_V001", [100, 200], embeddings)
    store = _store(tmp_path)

    result = store.get("L21_V001", 200)

    assert result is not None
    assert result.shape == (4, 3)
    np.testing.assert_allclose(result, embeddings[1], atol=1e-3)


def test_local_embedding_store_returns_none_for_missing_frame(tmp_path: Path) -> None:
    embeddings = np.ones((1, 4, 3), dtype=np.float32)
    _write_video(tmp_path, "L21_V001", [100], embeddings)
    store = _store(tmp_path)

    assert store.get("L21_V001", 999) is None


def test_local_embedding_store_bounds_mmap_cache(tmp_path: Path) -> None:
    embeddings = np.ones((1, 4, 3), dtype=np.float32)
    _write_video(tmp_path, "L21_V001", [1], embeddings)
    _write_video(tmp_path, "L22_V002", [1], embeddings)
    _write_video(tmp_path, "L23_V003", [1], embeddings)
    store = _store(tmp_path, cache_size=2)

    assert store.get("L21_V001", 1) is not None
    assert store.get("L22_V002", 1) is not None
    assert store.get("L23_V003", 1) is not None

    assert store.cache_video_ids == ("L22_V002", "L23_V003")


def test_maxsim_math_uses_best_patch_then_mean_phrase_score() -> None:
    patches = np.asarray(
        [
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
            ]
        ],
        dtype=np.float32,
    )
    phrases = np.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)

    scores = maxsim_query_scores(patches, phrases)

    assert scores.tolist() == pytest.approx([1.0])


def test_multi_query_average_ignores_empty_phrase_groups(tmp_path: Path) -> None:
    embeddings = np.asarray(
        [
            [
                [1.0, 0.0, 0.0],
                [0.8, 0.6, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ],
        dtype=np.float32,
    )
    _write_video(tmp_path, "L21_V001", [10], embeddings)
    groups = normalize_local_phrase_groups(
        ["q1", "q2", "q3"],
        [["phrase-a"], [], ["phrase-b", "phrase-b", "  "]],
    )
    reranker = LocalPhraseReranker(
        encoder=_PhraseEncoder(
            {
                "phrase-a": np.array([1.0, 0.0, 0.0], dtype=np.float32),
                "phrase-b": np.array([0.0, 0.6, 0.8], dtype=np.float32),
            }
        ),
        store=_store(tmp_path),
        config=KisConfig(
            local_score_batch_size=8,
            local_grid_size=2,
            local_num_patches=4,
            local_embedding_dim=3,
        ),
    )

    result = reranker.rerank(
        [LocalEmbeddingRef("A", "L21_V001", 10)],
        groups,
    )

    score = result.scores_by_id["A"]
    assert [group.query_index for group in groups] == [0, 2]
    assert score.local_query_scores == pytest.approx((1.0, 0.8))
    assert score.local_score == pytest.approx(0.9)


def test_missing_local_embedding_does_not_fail_request(tmp_path: Path) -> None:
    reranker = LocalPhraseReranker(
        encoder=_PhraseEncoder({"phrase": np.array([1.0, 0.0, 0.0], dtype=np.float32)}),
        store=_store(tmp_path),
        config=KisConfig(
            local_score_batch_size=8,
            local_grid_size=2,
            local_num_patches=4,
            local_embedding_dim=3,
        ),
    )

    result = reranker.rerank(
        [LocalEmbeddingRef("missing", "L99_V999", 10)],
        [LocalPhraseGroup(query_index=0, phrases=("phrase",))],
    )

    score = result.scores_by_id["missing"]
    assert score.local_score is None
    assert score.local_embedding_available is False
    assert result.num_missing_local_embeddings == 1
