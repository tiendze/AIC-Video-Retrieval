"""CLI: is the semantic channel's embedding space degenerate?

Answers the question left open by reports/20260810T161049Z_a40aac8, where
`semantic` (AIC 0.0451) scored BELOW `random` (0.0809). `per_query.csv` stored
only ranks, not the retrieved ids, so "every Vietnamese query returns the same
keyframes" could neither be confirmed nor ruled out from that report. This
script re-runs the queries and measures the space directly.

Pass `--translations` to run the SAME collection twice -- once with the
Vietnamese captions, once with their English translations -- which separates
"the text tower cannot read Vietnamese" from "the image side is broken".
Nothing is re-embedded and no label is touched.

Per CLAUDE.md §10 this is prepared for the project owner to run, not run
automatically: it needs live Milvus and a loaded BEiT-3 checkpoint on GPU.

  docker compose up -d
  python -m scripts.diagnose_embedding_space
  python -m scripts.diagnose_embedding_space \
      --translations configs/eval/aic_eval_v1.queries_en.json
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from app.config import Settings
from app.encoders.beit3 import BEiT3Encoder
from app.stores.milvus_client import MilvusStore
from datasets.aic2026 import AIC2026Adapter
from datasets.base import GroundTruthSegment
from eval.diagnostics.embedding_space import (
    CollapseThresholds,
    assess_collapse,
    pairwise_cosine_stats,
    render_report_md,
    result_overlap_stats,
    score_spread_stats,
)
from eval.query_translation import build_query_transform, load_translations
from eval.run import filter_ground_truth_to_available, load_pilot_available

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_BEIT3_CONFIG_PATH = _REPO_ROOT / "configs" / "model" / "beit3.yaml"
_MILVUS_CONFIG_PATH = _REPO_ROOT / "configs" / "stores" / "milvus.yaml"
_EVAL_CONFIG_PATH = _REPO_ROOT / "configs" / "eval" / "aic_pilot.yaml"
_THRESHOLDS_CONFIG_PATH = _REPO_ROOT / "configs" / "eval" / "diagnostics.yaml"
_DIAGNOSTICS_DIR = "reports/diagnostics"
_SEARCH_PROGRESS_EVERY = 25


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, OSError, subprocess.TimeoutExpired):
        return "unknown"


def _load_thresholds(path: Path) -> CollapseThresholds:
    raw = _load_yaml(path)
    return CollapseThresholds(
        query_cosine_mean=float(raw["query_cosine_mean"]),
        result_jaccard_mean=float(raw["result_jaccard_mean"]),
        union_ratio=float(raw["union_ratio"]),
        score_gap=float(raw["score_gap"]),
        min_signals=int(raw["min_signals"]),
    )


def _build_encoder(device: str) -> BEiT3Encoder:
    config = _load_yaml(_BEIT3_CONFIG_PATH)
    return BEiT3Encoder(
        checkpoint_path=str(_REPO_ROOT / config["checkpoint_path"]),
        sentencepiece_model_path=str(_REPO_ROOT / config["sentencepiece_model_path"]),
        device=device,
        batch_size=config["batch_size"],
        model_id=config["model_id"],
    )


def _build_milvus_store(settings: Settings, encoder: BEiT3Encoder) -> MilvusStore:
    config = _load_yaml(_MILVUS_CONFIG_PATH)
    return MilvusStore(
        uri=settings.milvus_uri,
        token=settings.milvus_token.get_secret_value(),
        model_id=encoder.model_id,
        embedding_dim=encoder.embedding_dim,
        hnsw_m=config["hnsw_m"],
        hnsw_ef_construction=config["hnsw_ef_construction"],
        search_ef=config["search_ef"],
    )


def _load_ground_truth(eval_config: dict[str, Any]) -> list[GroundTruthSegment]:
    """Same population eval/run.py scores -- reusing its two loaders rather
    than re-implementing the filter, so the diagnostic can never be measured
    over a different query set than the report it is explaining."""
    adapter = AIC2026Adapter(
        dataset_root=_REPO_ROOT / eval_config["dataset_root"],
        eval_set_path=_REPO_ROOT / eval_config["eval_set_path"],
    )
    available = load_pilot_available(_REPO_ROOT / eval_config["pilot_available_path"])
    return filter_ground_truth_to_available(
        adapter.load_ground_truth(eval_config["split"]), available
    )


def _search_all(
    store: MilvusStore, vectors: np.ndarray, top_k: int
) -> tuple[list[list[str]], list[list[float]]]:
    """One ANN search per query row. Returns `(keyframe_id lists, score lists)`
    in query order -- the two inputs the overlap/spread measures need."""
    id_lists: list[list[str]] = []
    score_lists: list[list[float]] = []
    for index in range(vectors.shape[0]):
        hits = store.search(vectors[index], top_k)
        id_lists.append([hit["keyframe_id"] for hit in hits])
        score_lists.append([float(hit["distance"]) for hit in hits])
        if (index + 1) % _SEARCH_PROGRESS_EVERY == 0:
            logger.info("searched %d/%d queries", index + 1, vectors.shape[0])
    return id_lists, score_lists


def _run_one_label(
    *,
    label: str,
    queries: Sequence[str],
    encoder: BEiT3Encoder,
    store: MilvusStore,
    top_k: int,
    thresholds: CollapseThresholds,
    git_sha: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Encodes `queries`, searches, measures, writes both report files, and
    returns the JSON document so the caller can print a comparison."""
    logger.info("[%s] encoding %d quer(ies)", label, len(queries))
    vectors = encoder.encode_text(list(queries))

    logger.info("[%s] searching Milvus (top_k=%d)", label, top_k)
    id_lists, score_lists = _search_all(store, vectors, top_k)

    cosine = pairwise_cosine_stats(vectors)
    overlap = result_overlap_stats(id_lists)
    spread = score_spread_stats(score_lists)
    verdict = assess_collapse(cosine, overlap, spread, thresholds)

    document = {
        "label": label,
        "git_sha": git_sha,
        "model_id": encoder.model_id,
        "collection": store.collection_name,
        "num_queries": len(queries),
        "top_k": top_k,
        "cosine": asdict(cosine),
        "overlap": asdict(overlap),
        "spread": asdict(spread),
        "verdict": asdict(verdict),
    }
    (output_dir / f"embedding_space_{label}.json").write_text(
        json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / f"embedding_space_{label}.md").write_text(
        render_report_md(
            label=label,
            git_sha=git_sha,
            model_id=encoder.model_id,
            collection_name=store.collection_name,
            num_queries=len(queries),
            cosine=cosine,
            overlap=overlap,
            spread=spread,
            verdict=verdict,
        ),
        encoding="utf-8",
    )
    return document


def _print_summary(documents: Sequence[dict[str, Any]]) -> None:
    print("\n=== embedding-space diagnostic ===")
    header = f"{'label':<6} {'collapsed':<10} {'q-cos':>9} {'jaccard':>9} {'union':>8} {'gap':>8}"
    print(header)
    for doc in documents:
        print(
            f"{doc['label']:<6} {str(doc['verdict']['collapsed']):<10} "
            f"{doc['cosine']['mean']:>9.4f} {doc['overlap']['mean_jaccard']:>9.4f} "
            f"{doc['overlap']['union_ratio']:>8.4f} {doc['spread']['mean_gap']:>8.4f}"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=_EVAL_CONFIG_PATH)
    parser.add_argument("--thresholds", type=Path, default=_THRESHOLDS_CONFIG_PATH)
    parser.add_argument(
        "--translations",
        type=Path,
        default=None,
        help="VI->EN sidecar; when given, an 'en' run is measured alongside 'vi'",
    )
    parser.add_argument("--device", default="cuda", help="'cuda' or 'cpu'")
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="defaults to max(top_ks) from --config, i.e. the submission limit",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = _parse_args()

    eval_config = _load_yaml(args.config)
    thresholds = _load_thresholds(args.thresholds)
    top_k = args.top_k or max(eval_config["top_ks"])

    ground_truth = _load_ground_truth(eval_config)
    if not ground_truth:
        raise SystemExit(
            "no ground-truth record survived the pilot_available filter -- "
            "nothing to diagnose (check configs/eval/pilot_available_v1.txt)"
        )

    settings = Settings()
    encoder = _build_encoder(args.device)
    store = _build_milvus_store(settings, encoder)

    git_sha = _git_sha()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = _REPO_ROOT / _DIAGNOSTICS_DIR / f"{timestamp}_{git_sha}"
    output_dir.mkdir(parents=True, exist_ok=True)

    runs: list[tuple[str, list[str]]] = [("vi", [gt.caption for gt in ground_truth])]
    if args.translations is not None:
        transform = build_query_transform(ground_truth, load_translations(args.translations))
        runs.append(("en", [transform(gt) for gt in ground_truth]))

    documents = [
        _run_one_label(
            label=label,
            queries=queries,
            encoder=encoder,
            store=store,
            top_k=top_k,
            thresholds=thresholds,
            git_sha=git_sha,
            output_dir=output_dir,
        )
        for label, queries in runs
    ]

    _print_summary(documents)
    logger.info("Wrote diagnostics to %s", output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
