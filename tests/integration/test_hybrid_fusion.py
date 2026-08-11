"""Integration test for hybrid mode (plan §P1.3, verify step
`test_hybrid_not_worse_than_best_single_channel`) -- the test that decides
F4's fate: "Hybrid **kem hon** kenh don tot nhat => chinh trong so, van kem
=> **bo han F4** va ghi ly do vao bao cao phase" (plan §P1.3).

The decisive comparison uses the TRANSLATED semantic leg
(`_build_live_translated_search_service`, the same second SearchService
eval/run.py wires for the `semantic_live_en` channel), not the raw-Vietnamese
`semantic` channel eval/run.py reports standalone. Reason (found running this
test 2026-08-12): production's SearchService (app/dependencies.py) ALWAYS
translates the semantic leg via its injected QueryTranslator, so a hybrid
measured on the untranslated service tests something no live query actually
goes through -- and BEiT-3's text tower is English-only, so untranslated VI
semantic alone measured 0.0012 AIC Final Score (near noise) against ocr's
0.0254, which drags any RRF fusion of the two below ocr alone by construction
(RRF weights channel RANK, not channel quality). `semantic_live_en` measured
0.6208 in the P0.9 report (reports/20260811T080907Z_fc0edde) -- an entirely
different regime for hybrid to fuse against.

Both the untranslated `semantic`/`hybrid` and the translated
`semantic_live_en`/`hybrid_live_en` are scored and reported so a human
reviewing the phase report sees both regimes, but the ASSERTION below is
against the translated pair -- that is the one a live user's query experiences.

Runs against the real self-generated eval set (`configs/eval/aic_pilot.yaml`)
via the real BEiT-3 checkpoint, the local VI->EN NMT translator
(configs/model/query_translator.yaml), and live Milvus/Elasticsearch/MongoDB
-- reuses `eval.run`'s own loading/scoring/service-building functions rather
than re-deriving them, so this test can never silently drift from what
`eval/run.py` itself measures.

Requires the full P0 pipeline already run against the pilot corpus (ingest
complete, `pilot_available_v1.txt` populated) AND the real BEiT-3 checkpoint
-- neither exists on a bare host, so this whole module is skipped unless both
are present (same guard as tests/integration/test_search_service_image.py).

PowerShell (this project's primary shell):
    docker compose up -d
    $env:BEIT3_CHECKPOINT_PATH = "models/beit3/beit3_base_itc_patch16_224.pth"
    $env:BEIT3_SPM_PATH = "models/beit3/beit3.spm"
    pytest -m "integration and model" tests/integration/test_hybrid_fusion.py -v

bash:
    docker compose up -d
    BEIT3_CHECKPOINT_PATH=models/beit3/beit3_base_itc_patch16_224.pth \
    BEIT3_SPM_PATH=models/beit3/beit3.spm \
    pytest -m "integration and model" tests/integration/test_hybrid_fusion.py -v

This runs 4 channels x every ground-truth query in
`configs/eval/aic_pilot.yaml`'s split, plus loading a second (translation)
model -- the same cost as one `eval/run.py --config configs/eval/aic_pilot.yaml`
pass with `translate_semantic_live: true` (minutes, not seconds). CLAUDE.md
§10: prepared here for the project owner to run themselves, not run
automatically by this agent.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

pytestmark = [pytest.mark.integration, pytest.mark.model]

logger = logging.getLogger(__name__)

_CHECKPOINT_PATH = os.environ.get("BEIT3_CHECKPOINT_PATH")
_SPM_PATH = os.environ.get("BEIT3_SPM_PATH")

if not _CHECKPOINT_PATH or not _SPM_PATH:
    pytest.skip(
        "BEIT3_CHECKPOINT_PATH / BEIT3_SPM_PATH not set — run scripts/setup_beit3.sh first",
        allow_module_level=True,
    )

from app.config import Settings  # noqa: E402
from datasets.aic2026 import AIC2026Adapter  # noqa: E402
from eval.metrics.retrieval import aic_final_score  # noqa: E402
from eval.run import (  # noqa: E402
    _build_encoder,
    _build_live_translated_search_service,
    _build_stores,
    _load_config,
    _prepare_ground_truth,
    _SemanticSearcher,
    score_channel,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_CONFIG_PATH = _REPO_ROOT / "configs" / "eval" / "aic_pilot.yaml"


async def test_hybrid_not_worse_than_best_single_channel():
    """plan §P1.3 decisive test for F4: hybrid's segment-level AIC Final
    Score, measured on the TRANSLATED semantic leg (matching what production
    actually serves), must be >= max(semantic_live_en, ocr) on the same eval
    set. If this fails, the plan's instruction is: adjust
    configs/search/search.yaml's hybrid_semantic_weight/hybrid_ocr_weight and
    rerun; if it STILL fails, drop F4 entirely and record why in the phase
    report (CLAUDE.md §8) -- do not massage this test to pass."""
    config = _load_config(_CONFIG_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    adapter = AIC2026Adapter(
        dataset_root=_REPO_ROOT / config.dataset_root,
        eval_set_path=_REPO_ROOT / config.eval_set_path,
    )
    ground_truth, _available = _prepare_ground_truth(config, adapter)
    encoder = _build_encoder(device)
    settings = Settings()
    mongo, milvus, es, search_service = _build_stores(settings, encoder)
    live_translated_search_service = _build_live_translated_search_service(
        encoder, milvus, es, mongo
    )

    top_k = max(config.top_ks)
    channel_scores: dict[str, float] = {}
    try:
        # Untranslated pair -- informational only, matches eval/run.py's
        # standalone `semantic`/`hybrid` channels (module docstring).
        for mode in ("semantic", "hybrid"):
            scored = await score_channel(
                _SemanticSearcher(search_service, mode), ground_truth, top_k
            )
            segment_ranks = [rank_segment for _gt, _rank_video, rank_segment in scored]
            channel_scores[mode] = aic_final_score(segment_ranks, config.top_ks)

        # ocr never translates (SearchService._search_ocr ignores the query
        # translator), so either service instance measures the same channel.
        scored = await score_channel(_SemanticSearcher(search_service, "ocr"), ground_truth, top_k)
        channel_scores["ocr"] = aic_final_score(
            [rank_segment for _gt, _rank_video, rank_segment in scored], config.top_ks
        )

        # Decisive pair -- the translated semantic leg a live query goes through.
        for mode in ("semantic", "hybrid"):
            scored = await score_channel(
                _SemanticSearcher(live_translated_search_service, mode), ground_truth, top_k
            )
            segment_ranks = [rank_segment for _gt, _rank_video, rank_segment in scored]
            channel_scores[f"{mode}_live_en"] = aic_final_score(segment_ranks, config.top_ks)
    finally:
        await mongo.close()
        await es.close()

    logger.info("channel AIC Final Scores: %s", channel_scores)

    best_single = max(channel_scores["semantic_live_en"], channel_scores["ocr"])
    assert channel_scores["hybrid_live_en"] >= best_single, (
        f"hybrid_live_en AIC Final Score {channel_scores['hybrid_live_en']:.4f} < best "
        f"single channel {best_single:.4f} (semantic_live_en="
        f"{channel_scores['semantic_live_en']:.4f}, ocr={channel_scores['ocr']:.4f}; "
        f"untranslated for reference: semantic={channel_scores['semantic']:.4f}, "
        f"hybrid={channel_scores['hybrid']:.4f}) -- plan §P1.3: adjust "
        "hybrid_semantic_weight/hybrid_ocr_weight in configs/search/search.yaml and "
        "rerun; if still worse, drop F4 and record why in the phase report."
    )
