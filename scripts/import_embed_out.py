"""CLI: import a Kaggle-produced embed_out/ artifact into Milvus + MongoDB —
giai đoạn 2 of `.claude/plans/kaggle-offline-embed.plan.md`.

Runs the parity gate from `pipelines/verify_embed_parity.py` first (same
checks as `scripts/verify_embed_parity.py` — checksum + probe cosine), then
for every `video_id` under `data/embed_out/vectors/`: joins its embedding
matrix with `data/shots.json` (real shot boundaries -> `shot_id`) and the
raw `.mp4`'s fps (-> `timestamp`), assigns a contiguous `global_index`, and
writes rows to both Milvus (`MilvusStore.upsert`) and MongoDB
(`KeyframeDocument` + `VideoDocument`) — CLAUDE.md §3: `frame_idx` must
travel with every keyframe in *both* stores, not just Milvus.

Prerequisites:
  docker compose up -d                     # Milvus + MongoDB
  bash scripts/setup_beit3.sh               # models/beit3/*  (parity probe)
  data/embed_out/ downloaded and unzipped from Kaggle
  data/shots.json, data/keyframe/, data/Videos_L*/video/*.mp4 present locally

WARNING: per CLAUDE.md §10, this writes to a live Milvus/MongoDB and — with
--reset — drops/deletes existing collections. Prepared here for you to run
yourself, not run automatically.

Usage:
  python scripts/import_embed_out.py --reset
  python scripts/import_embed_out.py --limit-videos 1      # smoke test
  python scripts/import_embed_out.py --device cpu           # no GPU for the probe
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import yaml

from app.config import Settings
from app.encoders.beit3 import BEiT3Encoder
from app.stores.milvus_client import MilvusStore
from app.stores.mongo_client import MongoStore
from datasets.aic2026 import AIC2026Adapter
from datasets.base import VideoRef
from pipelines.import_embed_out import (
    build_imported_keyframes,
    build_video_document,
    discover_video_ids,
    is_video_already_imported,
    load_shot_index,
    load_video_vectors,
    to_keyframe_document,
    to_milvus_vector,
)
from pipelines.ingest import next_global_index_start
from pipelines.verify_embed_parity import (
    EmbedParityError,
    ParityReport,
    load_manifest,
    load_probe,
    run_parity_probe,
    verify_checksums,
    verify_manifest_fields,
)
from pipelines.video_io import read_video_info

# Windows consoles often default to cp1252, which cannot encode the
# Vietnamese metadata this script logs (video titles/authors) — same fix as
# run_ingest.py / verify_embed_parity.py.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_BEIT3_CONFIG_PATH = _REPO_ROOT / "configs" / "model" / "beit3.yaml"
_MILVUS_CONFIG_PATH = _REPO_ROOT / "configs" / "stores" / "milvus.yaml"
_COSINE_THRESHOLD = 0.9999  # kaggle-offline-embed.plan.md §3 "Parity probe"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--embed-out-dir", default=str(_REPO_ROOT / "data" / "embed_out"), help="Path to embed_out/"
    )
    parser.add_argument(
        "--dataset-root", type=Path, default=_REPO_ROOT / "data", help="AIC2026Adapter root"
    )
    parser.add_argument("--shots-json", type=Path, default=_REPO_ROOT / "data" / "shots.json")
    parser.add_argument("--image-dir", type=Path, default=_REPO_ROOT / "data" / "keyframe")
    parser.add_argument("--device", default="cuda", help="'cuda' or 'cpu' (parity probe only)")
    parser.add_argument(
        "--limit-videos",
        type=int,
        default=None,
        help="Import only the first N video_ids (smoke test)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help=(
            "Drop the Milvus collection and delete every MongoDB videos/keyframes "
            "document before importing (plan §9.5: this keyframe source replaces "
            "the P0.7 TransNetV2 pilot data wholesale, not alongside it)."
        ),
    )
    return parser.parse_args()


def _load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _run_parity_gate(embed_out_dir: Path, beit3_config: dict, device: str) -> BEiT3Encoder:
    """Blocks import outright on any mismatch (CLAUDE.md §3) — mirrors
    scripts/verify_embed_parity.py exactly, returning the encoder built for
    the probe so its identity (model_id/embedding_dim) drives MilvusStore."""
    manifest = load_manifest(embed_out_dir / "manifest.json")
    try:
        verify_checksums(
            manifest,
            checkpoint_path=_REPO_ROOT / beit3_config["checkpoint_path"],
            spm_path=_REPO_ROOT / beit3_config["sentencepiece_model_path"],
        )
    except EmbedParityError as exc:
        logger.error("BLOCKED: %s", exc)
        sys.exit(1)

    encoder = BEiT3Encoder(
        checkpoint_path=str(_REPO_ROOT / beit3_config["checkpoint_path"]),
        sentencepiece_model_path=str(_REPO_ROOT / beit3_config["sentencepiece_model_path"]),
        device=device,
        batch_size=beit3_config["batch_size"],
        model_id=beit3_config["model_id"],
    )
    try:
        verify_manifest_fields(
            manifest, model_id=encoder.model_id, embedding_dim=encoder.embedding_dim
        )
    except EmbedParityError as exc:
        logger.error("BLOCKED: %s", exc)
        sys.exit(1)

    texts, remote_vectors = load_probe(embed_out_dir / "probe")
    probe_results = run_parity_probe(texts, remote_vectors, encoder.encode_text)
    report = ParityReport(probe_results=probe_results, threshold=_COSINE_THRESHOLD)
    logger.info("parity probe: min_cosine=%.6f passed=%s", report.min_cosine, report.passed)
    if not report.passed:
        logger.error("BLOCKED: parity probe failed -- refusing to import data/embed_out/vectors/")
        sys.exit(1)
    return encoder


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


async def _import_video(
    video_id: str,
    *,
    embed_out_dir: Path,
    shot_index: dict,
    video_map: dict[str, VideoRef],
    image_dir: Path,
    milvus: MilvusStore,
    mongo: MongoStore,
    global_index_start: int,
) -> int:
    """Imports one video's keyframes into Milvus + MongoDB. Returns the
    number of keyframes written, so the caller can advance global_index_start
    for the next video without a DB round trip per video."""
    video_ref = video_map.get(video_id)
    if video_ref is None:
        raise ValueError(f"{video_id}: in embed_out/vectors/ but not found by AIC2026Adapter")
    shot_ranges = shot_index.get(video_id)
    if shot_ranges is None:
        raise ValueError(f"{video_id}: in embed_out/vectors/ but missing from data/shots.json")

    vectors, jsonl_rows = load_video_vectors(embed_out_dir / "vectors", video_id)
    fps = read_video_info(video_ref.source_path).fps
    keyframes = build_imported_keyframes(
        video_id,
        vectors,
        jsonl_rows,
        shot_ranges,
        fps=fps,
        global_index_start=global_index_start,
        author=video_ref.category or "",
    )

    milvus.upsert([to_milvus_vector(k) for k in keyframes])
    await mongo.bulk_insert_keyframes([to_keyframe_document(k, image_dir) for k in keyframes])
    await mongo.upsert_video(build_video_document(video_ref, keyframes, fps))
    return len(keyframes)


async def _reset_stores(milvus: MilvusStore, mongo: MongoStore) -> None:
    logger.warning(
        "--reset: dropping Milvus collection %s and MongoDB videos/keyframes",
        milvus.collection_name,
    )
    milvus.drop_collection()
    await mongo.keyframes.delete_many({})
    await mongo.videos.delete_many({})


async def _main() -> int:
    args = _parse_args()
    settings = Settings()
    embed_out_dir = Path(args.embed_out_dir)
    beit3_config = _load_yaml(_BEIT3_CONFIG_PATH)

    encoder = _run_parity_gate(embed_out_dir, beit3_config, args.device)
    milvus = _build_milvus_store(settings, encoder)
    mongo = MongoStore(uri=settings.mongo_uri)
    await mongo.ensure_indexes()

    if args.reset:
        await _reset_stores(milvus, mongo)

    adapter = AIC2026Adapter(dataset_root=args.dataset_root, eval_set_path=Path("/dev/null"))
    video_map = {v.video_id: v for v in adapter.iter_videos("all")}
    shot_index = load_shot_index(args.shots_json)

    video_ids = discover_video_ids(embed_out_dir / "vectors")
    if args.limit_videos is not None:
        video_ids = video_ids[: args.limit_videos]
    logger.info("Importing %d video(s) from %s", len(video_ids), embed_out_dir)

    global_index_start = await next_global_index_start(mongo)
    total_keyframes = 0
    skipped_video_ids: list[str] = []
    failed_video_ids: list[str] = []
    for video_id in video_ids:
        # Re-running without --reset must not re-import a video that already
        # finished: doing so desyncs global_index between Milvus (upsert
        # overwrites it) and MongoDB keyframes (duplicate-key inserts are
        # skipped, so the old value survives) — is_video_already_imported's
        # docstring has the full mechanism.
        if is_video_already_imported(await mongo.get_video(video_id)):
            skipped_video_ids.append(video_id)
            continue
        try:
            written = await _import_video(
                video_id,
                embed_out_dir=embed_out_dir,
                shot_index=shot_index,
                video_map=video_map,
                image_dir=args.image_dir,
                milvus=milvus,
                mongo=mongo,
                global_index_start=global_index_start,
            )
        except (ValueError, OSError) as exc:
            logger.error("import failed for video_id=%s: %s", video_id, exc)
            failed_video_ids.append(video_id)
            continue
        global_index_start += written
        total_keyframes += written

    logger.info(
        "Done: %d video(s) imported, %d skipped (already complete), "
        "%d keyframe(s) written, %d failed",
        len(video_ids) - len(failed_video_ids) - len(skipped_video_ids),
        len(skipped_video_ids),
        total_keyframes,
        len(failed_video_ids),
    )
    if failed_video_ids:
        logger.warning("Failed video_id(s): %s", failed_video_ids)

    await mongo.close()
    return 1 if failed_video_ids else 0


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    sys.exit(asyncio.run(_main()))


if __name__ == "__main__":
    main()
