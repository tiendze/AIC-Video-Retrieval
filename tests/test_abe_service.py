from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.services.abe import (
    AbeConceptBank,
    AbeConceptLookup,
    AbePatchCache,
    AbeScorer,
    abe_raw_score,
    assign_abe_pair_attributes,
    build_concept_bank_payload,
    local_alignment_score,
    save_concept_bank,
)
from app.services.kis_scoring import AbePair, KisConfig, KisQuery, QueryType


@dataclass(frozen=True)
class _LocalText:
    text: str
    tokens: tuple[str, ...]
    embeddings: np.ndarray


class _FakeAbeEncoder:
    embedding_dim = 6
    model_id = "fake-fgclip"
    preprocess_id = "fake-preprocess"

    def __init__(self) -> None:
        self.encode_text_calls: list[list[str]] = []
        self.encode_image_patches_calls = 0

    def encode_text(self, texts):
        self.encode_text_calls.append(list(texts))
        return np.asarray([self._text_vector(text) for text in texts], dtype=np.float32)

    def encode_text_tokens(self, texts):
        return [
            _LocalText(
                text=text,
                tokens=tuple(self._tokens(text)),
                embeddings=np.asarray(
                    [self._text_vector(token) for token in self._tokens(text)], dtype=np.float32
                ),
            )
            for text in texts
        ]

    def encode_image_patches(self, images):
        self.encode_image_patches_calls += 1
        batches = []
        for image in images:
            red, green, blue = image.getpixel((0, 0))
            if red >= green and red >= blue:
                patches = [
                    self._text_vector("red"),
                    self._text_vector("packet"),
                    self._text_vector("white"),
                    self._text_vector("uniform"),
                ]
            else:
                patches = [
                    self._text_vector("white"),
                    self._text_vector("packet"),
                    self._text_vector("red"),
                    self._text_vector("uniform"),
                ]
            batches.append(np.asarray(patches, dtype=np.float32))
        return np.asarray(batches, dtype=np.float32)

    def _tokens(self, text: str) -> list[str]:
        return [token.strip(",") for token in text.lower().split() if token.strip(",")]

    def _text_vector(self, text: str) -> np.ndarray:
        text = text.lower().strip()
        exact = {
            "red": np.array([1, 0, 0, 0, 0, 0], dtype=np.float32),
            "white": np.array([0, 1, 0, 0, 0, 0], dtype=np.float32),
            "packet": np.array([0, 0, 1, 0, 0, 0], dtype=np.float32),
            "uniform": np.array([0, 0, 0, 1, 0, 0], dtype=np.float32),
            "helmet": np.array([0, 0, 0, 0, 1, 0], dtype=np.float32),
            "shirt": np.array([0, 0, 0, 0, 0, 1], dtype=np.float32),
            "red packet, white uniform": np.array([1, 0, 1, 0, 0, 0], dtype=np.float32),
        }
        if text in exact:
            return _normalize(exact[text])
        tokens = self._tokens(text)
        if not tokens:
            return np.zeros(self.embedding_dim, dtype=np.float32)
        vector = np.sum([self._text_vector(token) for token in tokens], axis=0)
        return _normalize(vector)


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)


def _concept_bank(encoder: _FakeAbeEncoder) -> AbeConceptBank:
    concepts = ["packet", "uniform", "helmet", "shirt"]
    embeddings = encoder.encode_text(concepts)
    return AbeConceptBank(
        concepts=concepts,
        embeddings=embeddings,
        metadata={"model_id": encoder.model_id, "embedding_dim": encoder.embedding_dim},
    )


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    Image.new("RGB", (8, 8), color=color).save(path, format="JPEG")


def test_concept_preprocessing_deduplicates_encodes_and_builds_metadata() -> None:
    encoder = _FakeAbeEncoder()

    payload = build_concept_bank_payload(
        [" Shirt ", "shirt", "", "packet"],
        encoder,
        checkpoint="fake.pth",
        model_id=encoder.model_id,
        batch_size=2,
        encoder_kind="fg_clip",
        variant="FG-CLIP Base ViT-B/16",
        source_vocab="data/abe/splice_concepts.txt",
    )

    assert payload["concepts"] == ["shirt", "packet"]
    assert payload["embeddings"].shape == (2, encoder.embedding_dim)
    assert payload["embeddings"].dtype == np.float16
    assert np.isfinite(payload["embeddings"]).all()
    np.testing.assert_allclose(
        np.linalg.norm(payload["embeddings"].astype(np.float32), axis=1), 1.0, atol=2e-3
    )
    assert payload["metadata"]["checkpoint"] == "fake.pth"
    assert payload["metadata"]["encoder"] == "fg_clip"
    assert payload["metadata"]["variant"] == "FG-CLIP Base ViT-B/16"
    assert payload["metadata"]["model_id"] == encoder.model_id
    assert payload["metadata"]["embedding_dim"] == encoder.embedding_dim
    assert payload["metadata"]["normalized"] is True
    assert payload["metadata"]["source_vocab"] == "data/abe/splice_concepts.txt"
    assert payload["metadata"]["duplicates_removed"] == 1


def test_concept_bank_save_and_reload_round_trips(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    encoder = _FakeAbeEncoder()
    payload = build_concept_bank_payload(
        ["shirt", "packet"],
        encoder,
        checkpoint="fake.pth",
        model_id=encoder.model_id,
        batch_size=2,
    )
    output = tmp_path / "concept_bank.pt"

    save_concept_bank(payload, output)
    bank = AbeConceptBank.load(
        output,
        expected_model_id=encoder.model_id,
        expected_embedding_dim=encoder.embedding_dim,
    )

    assert bank.concepts == ["shirt", "packet"]
    assert bank.embeddings.shape == (2, encoder.embedding_dim)


def test_beit3_concept_bank_metadata_is_rejected_for_fgclip_runtime() -> None:
    encoder = _FakeAbeEncoder()
    bank = AbeConceptBank(
        concepts=["shirt"],
        embeddings=encoder.encode_text(["shirt"]),
        metadata={
            "encoder": "beit3",
            "model_id": encoder.model_id,
            "embedding_dim": encoder.embedding_dim,
            "normalization": "l2",
            "normalized": True,
            "dtype": "float16",
        },
    )

    with pytest.raises(ValueError, match="encoder mismatch"):
        bank.validate_compatibility(
            expected_model_id=encoder.model_id,
            expected_embedding_dim=encoder.embedding_dim,
            expected_encoder_kind="fg_clip",
        )


def test_concept_lookup_returns_sorted_top_p_and_uses_cache() -> None:
    encoder = _FakeAbeEncoder()
    bank = _concept_bank(encoder)
    lookup = AbeConceptLookup(bank, encoder, top_p=3)
    calls_before_lookup = len(encoder.encode_text_calls)

    nearest = lookup.nearest_concepts(" packet ")
    cached = lookup.nearest_concepts("packet")

    assert [item.concept for item in nearest][0] == "packet"
    assert [item.similarity for item in nearest] == sorted(
        [item.similarity for item in nearest], reverse=True
    )
    assert cached is nearest
    assert lookup.cache_size == 1
    assert len(encoder.encode_text_calls) == calls_before_lookup + 1


def test_abe_pair_semantics_assign_other_attributes_as_negatives() -> None:
    assignments = assign_abe_pair_attributes(
        [AbePair("red", "packet"), AbePair("white", "uniform")]
    )

    assert assignments[0].positive_attributes == ("red",)
    assert assignments[0].negative_attributes == ("white",)
    assert assignments[1].positive_attributes == ("white",)
    assert assignments[1].negative_attributes == ("red",)


def test_single_abe_pair_has_no_negative_attributes() -> None:
    assignments = assign_abe_pair_attributes([AbePair("red", "packet")])

    assert assignments[0].positive_attributes == ("red",)
    assert assignments[0].negative_attributes == ()


def test_local_alignment_uses_mean_of_top_k_patches_not_only_max() -> None:
    text = np.asarray([[1, 0], [0, 1]], dtype=np.float32)
    patches = np.asarray([[1, 0], [0, 1], [0, 0]], dtype=np.float32)

    result = local_alignment_score(text, patches, patch_top_k=2, labels=["x", "y"])

    assert result.score == pytest.approx(0.5)
    assert len(result.top_patches) == 4


def test_abe_raw_score_uses_refined_local_binding_delta_and_global_weight() -> None:
    binding_delta, local_score, raw_score = abe_raw_score(
        base_local=0.2,
        refined_local=0.5,
        global_abe=0.8,
        global_weight=0.3,
    )

    assert binding_delta == pytest.approx(0.3)
    assert local_score == pytest.approx(0.8)
    assert raw_score == pytest.approx(0.7 * 0.8 + 0.3 * 0.8)


def test_patch_cache_batches_stores_fp16_and_reuses_same_candidate_pool(tmp_path: Path) -> None:
    encoder = _FakeAbeEncoder()
    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"
    _write_image(first, (255, 0, 0))
    _write_image(second, (0, 0, 255))
    cache = AbePatchCache(dtype="float16")

    patches = cache.get_or_encode(
        candidate_ids=["A", "B"],
        image_paths_by_id={"A": first, "B": second},
        encoder=encoder,
        batch_size=2,
    )
    cached = cache.get_or_encode(
        candidate_ids=["A", "B"],
        image_paths_by_id={"A": first, "B": second},
        encoder=encoder,
        batch_size=2,
    )

    assert patches["A"].shape == (4, encoder.embedding_dim)
    assert patches["A"].dtype == np.float16
    assert np.isfinite(patches["A"]).all()
    np.testing.assert_allclose(
        np.linalg.norm(patches["A"].astype(np.float32), axis=1), 1.0, atol=2e-3
    )
    assert cached["A"] is patches["A"]
    assert encoder.encode_image_patches_calls == 1


def test_full_abe_scorer_reports_formula_components_and_rank_percentiles(
    tmp_path: Path,
) -> None:
    encoder = _FakeAbeEncoder()
    red_image = tmp_path / "red.jpg"
    blue_image = tmp_path / "blue.jpg"
    _write_image(red_image, (255, 0, 0))
    _write_image(blue_image, (0, 0, 255))
    config = KisConfig(
        candidate_multiplier=2,
        candidate_min=4,
        abe_concept_top_p=2,
        abe_patch_top_k=2,
        abe_global_weight=0.3,
        abe_patch_batch_size=2,
    )
    scorer = AbeScorer(encoder=encoder, concept_bank=_concept_bank(encoder), config=config)

    ranked = scorer.score_queries(
        candidate_ids=["A", "B"],
        queries=[
            KisQuery(
                query_index=0,
                type=QueryType.ABE,
                pairs=(AbePair("red", "packet"), AbePair("white", "uniform")),
            )
        ],
        image_paths_by_id={"A": red_image, "B": blue_image},
        global_image_vectors_by_id={
            "A": encoder._text_vector("red packet, white uniform"),
            "B": encoder._text_vector("white packet, red uniform"),
        },
        patch_cache=AbePatchCache(),
    )

    components = ranked[0].components_by_id["A"]
    assert components.pairs[0].positive_attributes == ["red"]
    assert components.pairs[0].negative_attributes == ["white"]
    assert components.binding_delta == pytest.approx(
        abs(components.refined_local - components.base_local)
    )
    assert components.local_score == pytest.approx(
        components.refined_local + components.binding_delta
    )
    assert components.raw_score == pytest.approx(
        0.7 * components.local_score + 0.3 * components.global_abe
    )
    assert 0 <= ranked[0].rankings["A"].rank_percentile <= 1
    assert 0 <= ranked[0].rankings["B"].rank_percentile <= 1
    assert components.patch_heatmap is not None
    assert components.patch_heatmap.patch_count == 4
