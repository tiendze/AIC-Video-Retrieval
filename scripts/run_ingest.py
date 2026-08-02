"""CLI: ingest the AIC 2026 dataset end to end -- VISION + TEXT/OCR branches.

Every video this script processes runs the VISION branch (shot detect ->
keyframe -> embed -> Milvus/MongoDB) and then the TEXT/OCR branch (text
filter -> phash dedupe -> Gemini OCR -> Elasticsearch), reaching
ingest_status="complete" as long as the shared Gemini token budget
(configs/ingest/token_budget.yaml) is not exhausted first -- a video that
runs out of budget stays at "embedded" and resumes OCR-only on the next
invocation of this script (pipelines/ingest.py: resume_ocr_for_video).

Needs a live GEMINI_API_KEY (spends real money per call, PRD risk R2) and a
running Elasticsearch (docker compose).

WARNING: resource-heavy (GPU inference over the whole split; needs the
Docker storage stack up) -- per CLAUDE.md section 10, prepared here for you
to run yourself, not run automatically.

Prerequisites:
  docker compose up -d                 # Milvus + MongoDB + Elasticsearch
  bash scripts/setup_beit3.sh          # models/beit3/*
  bash scripts/setup_transnetv2.sh     # models/transnetv2/*
  pip install -e ".[ocr]"              # extends [ingest] with google-genai

Usage:
  python scripts/run_ingest.py --split pilot
  python scripts/run_ingest.py --split pilot --device cpu   # no GPU available
  python scripts/run_ingest.py --split pilot --batches L21  # narrow to one batch

After a run, writes configs/eval/pilot_available_v1.txt (+ .sha256) -- the
video_ids that reached at least "embedded", across this run and any prior
one. Per plan P0.7, this is the frozen denominator every later metric
report must cite (CLAUDE.md section 8: the unique video_id count actually
evaluated).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import logging
import sys
from pathlib import Path

import yaml

from app.config import Settings
from app.encoders.beit3 import BEiT3Encoder
from app.stores.es_client import EsStore
from app.stores.milvus_client import MilvusStore
from app.stores.mongo_client import MongoStore
from datasets.aic2026 import AIC2026Adapter
from pipelines.ingest import run_ingest
from pipelines.ocr import GeminiOcrClient, OcrConfig
from pipelines.shot_detect import ShotDetector

# Windows consoles often default to cp1252, which cannot encode the Vietnamese
# text this project logs (video titles, error messages) -- reconfigure early
# so a UnicodeEncodeError never crashes a multi-hour ingest run partway
# through (discovered via `--help` on this exact script: argparse's
# print_help() hit this before any real ingest work even started).
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_BEIT3_CONFIG_PATH = _REPO_ROOT / "configs" / "model" / "beit3.yaml"
_TRANSNETV2_CONFIG_PATH = _REPO_ROOT / "configs" / "model" / "transnetv2.yaml"
_MILVUS_CONFIG_PATH = _REPO_ROOT / "configs" / "stores" / "milvus.yaml"
_OCR_CONFIG_PATH = _REPO_ROOT / "configs" / "ingest" / "ocr.yaml"
_TOKEN_BUDGET_CONFIG_PATH = _REPO_ROOT / "configs" / "ingest" / "token_budget.yaml"
_ES_OCR_INDEX_NAME = "ocr_keyframes"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--split", default="pilot", choices=["pilot", "all"])
    parser.add_argument(
        "--batches",
        type=str,
        default=None,
        help=(
            "Comma-separated batch prefixes to narrow --split down to, e.g. "
            "'L21' or 'L21,L22' (default: every batch in --split). Useful for "
            "timing a smaller slice before running the full split."
        ),
    )
    parser.add_argument("--dataset-root", type=Path, default=_REPO_ROOT / "data")
    parser.add_argument(
        "--eval-set-path", type=Path, default=_REPO_ROOT / "configs" / "eval" / "aic_eval_v1.json"
    )
    parser.add_argument("--image-dir", type=Path, default=_REPO_ROOT / "data" / "keyframes")
    parser.add_argument(
        "--pilot-list-out",
        type=Path,
        default=_REPO_ROOT / "configs" / "eval" / "pilot_available_v1.txt",
    )
    parser.add_argument("--device", default="cuda", help="'cuda' or 'cpu'")
    parser.add_argument("--beit3-batch-size", type=int, default=None, help="overrides beit3.yaml")
    return parser.parse_args()


def _load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_encoder(args: argparse.Namespace) -> BEiT3Encoder:
    config = _load_yaml(_BEIT3_CONFIG_PATH)
    batch_size = args.beit3_batch_size or config["batch_size"]
    return BEiT3Encoder(
        checkpoint_path=str(_REPO_ROOT / config["checkpoint_path"]),
        sentencepiece_model_path=str(_REPO_ROOT / config["sentencepiece_model_path"]),
        device=args.device,
        batch_size=batch_size,
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


def _batch_of(video_id: str) -> str:
    return video_id.split("_V", 1)[0]


def _write_pilot_available_list(video_ids: list[str], out_path: Path) -> None:
    """Merges `video_ids` from this run into any list already on disk (union
    by video_id), so a narrower `--batches` run never erases entries a prior
    full-split run already recorded -- this file is the frozen denominator
    every later metric report must cite (CLAUDE.md §8)."""
    existing_ids: set[str] = set()
    if out_path.exists():
        lines = out_path.read_text(encoding="utf-8").splitlines()
        existing_ids = {line.strip() for line in lines if line.strip()}
    all_ids = existing_ids | set(video_ids)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(sorted(all_ids)) + "\n"
    out_path.write_text(content, encoding="utf-8")
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    out_path.with_name(out_path.name + ".sha256").write_text(digest + "\n", encoding="utf-8")
    logger.info(
        "Wrote %d video_id(s) to %s (%d new this run, sha256=%s)",
        len(all_ids),
        out_path,
        len(all_ids) - len(existing_ids),
        digest,
    )


async def _main() -> int:
    args = _parse_args()
    settings = Settings()

    adapter = AIC2026Adapter(dataset_root=args.dataset_root, eval_set_path=args.eval_set_path)
    transnetv2_config = _load_yaml(_TRANSNETV2_CONFIG_PATH)
    transnetv2_checkpoint = _REPO_ROOT / transnetv2_config["checkpoint_path"]
    shot_detector = ShotDetector(checkpoint_path=str(transnetv2_checkpoint), device=args.device)
    encoder = _build_encoder(args)

    mongo = MongoStore(uri=settings.mongo_uri)
    await mongo.ensure_indexes()
    milvus = _build_milvus_store(settings, encoder)

    ocr_config = OcrConfig(**_load_yaml(_OCR_CONFIG_PATH))
    ocr_client = GeminiOcrClient(
        api_key=settings.gemini_api_key.get_secret_value(), config=ocr_config
    )
    es = EsStore(
        url=settings.elasticsearch_url,
        index_name=_ES_OCR_INDEX_NAME,
        api_key=settings.elastic_api_key.get_secret_value() if settings.elastic_api_key else None,
    )
    await es.ensure_index()
    ocr_token_budget = _load_yaml(_TOKEN_BUDGET_CONFIG_PATH)["token_budget"]

    videos = list(adapter.iter_videos(args.split))
    if args.batches:
        wanted_batches = {b.strip() for b in args.batches.split(",")}
        videos = [v for v in videos if _batch_of(v.video_id) in wanted_batches]
        if not videos:
            logger.warning("--batches=%s matched no video in split=%s", args.batches, args.split)
    logger.info(
        "Ingesting %d video(s) (split=%s%s)",
        len(videos),
        args.split,
        f", batches={args.batches}" if args.batches else "",
    )

    summary = await run_ingest(
        videos,
        shot_detector=shot_detector,
        encoder=encoder,
        mongo=mongo,
        milvus=milvus,
        image_dir=args.image_dir,
        ocr_client=ocr_client,
        es=es,
        ocr_token_budget=ocr_token_budget,
        ocr_config=ocr_config,
    )

    logger.info(
        "Done: %d completed, %d skipped (already ocr_done/complete), %d failed",
        len(summary.completed_video_ids),
        len(summary.skipped_video_ids),
        len(summary.failed_video_ids),
    )
    if summary.failed_video_ids:
        logger.warning("Failed video_id(s): %s", summary.failed_video_ids)
    logger.info(
        "OCR (R2 cost evidence): after_text_filter=%d after_dedupe=%d gemini_calls=%d "
        "tokens_used=%d/%d",
        summary.ocr_after_text_filter,
        summary.ocr_after_dedupe,
        summary.ocr_gemini_calls,
        summary.ocr_tokens_used,
        ocr_token_budget,
    )
    if summary.ocr_failed_keyframe_ids:
        logger.warning(
            "OCR failed for %d keyframe(s): %s",
            len(summary.ocr_failed_keyframe_ids),
            summary.ocr_failed_keyframe_ids,
        )

    _write_pilot_available_list(
        summary.completed_video_ids + summary.skipped_video_ids, args.pilot_list_out
    )

    await mongo.close()
    await es.close()
    return 1 if summary.failed_video_ids else 0


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    sys.exit(asyncio.run(_main()))


if __name__ == "__main__":
    main()
