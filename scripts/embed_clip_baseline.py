"""CLI: embed the ALREADY-INGESTED keyframe corpus with CLIP ViT-B/32 into
its own Milvus collection -- the P1.4 mandatory baseline (PRD §6.3): "CLIP
ViT-B/32 ... phải chạy bằng encoder của mình trên keyframe của mình".

Reindexes keyframes `scripts/run_ingest.py` already wrote to MongoDB/disk --
does NOT redo shot detection or keyframe extraction, only re-encodes the
existing images with a different encoder and writes to a NEW `model_id`'d
collection (`keyframes_clip_vit_b32_openai`), never mixing into the BEiT-3
collection (CLAUDE.md §3).

Resumable: appends a video_id to a local checkpoint file after all its
keyframes are embedded, and skips any video_id already recorded there on a
re-run. Milvus's own upsert is idempotent per keyframe_id (CLAUDE.md T1)
regardless, but re-encoding a whole video's images just to overwrite
identical vectors wastes GPU time on a run this long.

WARNING: resource-heavy (GPU inference over the full corpus, ~221,903
keyframes). Per CLAUDE.md section 10, prepared here for the project owner to
run themselves, not run automatically by this agent.

Prerequisites:
  docker compose up -d                    # Milvus + MongoDB (keyframes already ingested)
  pip install -e ".[baseline-clip]"
  pip install --no-deps open-clip-torch==3.3.0   # see pyproject.toml comment: avoids a
                                                  # real timm==0.4.12 (beit3) vs timm>=1.0.17
                                                  # (open-clip-torch's own metadata) conflict

Usage:
  # time a pilot slice FIRST (CLAUDE.md §10.4 -- do not guess throughput):
  python scripts/embed_clip_baseline.py --limit-videos 5
  python scripts/embed_clip_baseline.py                    # full corpus
  python scripts/embed_clip_baseline.py --device cpu       # no GPU available
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import yaml
from PIL import Image

from app.config import Settings
from app.encoders.clip_baseline import ClipEncoder
from app.stores.milvus_client import KeyframeVector, MilvusStore
from app.stores.mongo_client import MongoStore
from datasets.base import batch_of

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CLIP_CONFIG_PATH = _REPO_ROOT / "configs" / "model" / "clip.yaml"
_MILVUS_CONFIG_PATH = _REPO_ROOT / "configs" / "stores" / "milvus.yaml"
_DEFAULT_CHECKPOINT_PATH = _REPO_ROOT / "data" / "clip_baseline_embed_checkpoint.txt"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--device", default=None, help="overrides configs/model/clip.yaml")
    parser.add_argument(
        "--batch-size", type=int, default=None, help="overrides configs/model/clip.yaml"
    )
    parser.add_argument(
        "--preprocess-workers",
        type=int,
        default=None,
        help="overrides configs/model/clip.yaml -- CPU threads decoding/resizing images "
        "in parallel; this, not --batch-size, is the real throughput lever (measured "
        "2026-08-12: image decode dominates, not GPU compute)",
    )
    parser.add_argument("--checkpoint-path", type=Path, default=_DEFAULT_CHECKPOINT_PATH)
    parser.add_argument(
        "--limit-videos",
        type=int,
        default=None,
        help="process only the first N not-yet-done videos -- for timing a "
        "pilot slice before committing to the full corpus (CLAUDE.md §10.4)",
    )
    return parser.parse_args()


def _load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_encoder(args: argparse.Namespace) -> ClipEncoder:
    config = _load_yaml(_CLIP_CONFIG_PATH)
    return ClipEncoder(
        model_name=config["model_name"],
        pretrained_tag=config["pretrained_tag"],
        device=args.device or config["device"],
        batch_size=args.batch_size or config["batch_size"],
        model_id=config["model_id"],
        preprocess_workers=args.preprocess_workers or config["preprocess_workers"],
    )


def _build_milvus_store(settings: Settings, encoder: ClipEncoder) -> MilvusStore:
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


def _load_checkpoint(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def _append_checkpoint(path: Path, video_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(video_id + "\n")


async def _build_author_map(mongo: MongoStore) -> dict[str, str]:
    """`author` is a VideoDocument field, not a KeyframeDocument one -- keyframe
    rows don't carry it, so this joins by video_id at embed time (same source
    `scripts/run_ingest.py` wrote to `upsert_video`, no dataset-adapter reread)."""
    authors: dict[str, str] = {}
    async for doc in mongo.scroll_all_videos():
        authors[doc["video_id"]] = doc.get("author", "unknown")
    return authors


async def _group_keyframes_by_video(mongo: MongoStore) -> dict[str, list[dict]]:
    by_video: dict[str, list[dict]] = defaultdict(list)
    async for doc in mongo.scroll_all_keyframes():
        by_video[doc["video_id"]].append(doc)
    return by_video


def _embed_video(
    video_id: str,
    keyframe_docs: list[dict],
    author: str,
    encoder: ClipEncoder,
    milvus: MilvusStore,
) -> None:
    images = [Image.open(doc["image_path"]) for doc in keyframe_docs]
    vectors = encoder.encode_image(images)
    rows = [
        KeyframeVector(
            keyframe_id=doc["keyframe_id"],
            embedding=vectors[i],
            video_id=doc["video_id"],
            global_index=doc["global_index"],
            frame_idx=doc["frame_idx"],
            timestamp=doc["timestamp"],
            author=author,
            batch=batch_of(video_id),
        )
        for i, doc in enumerate(keyframe_docs)
    ]
    milvus.upsert(rows)


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()
    settings = Settings()

    encoder = _build_encoder(args)
    milvus = _build_milvus_store(settings, encoder)
    mongo = MongoStore(uri=settings.mongo_uri)

    logger.info(
        "ClipEncoder ready: model_id=%s, collection=%s", encoder.model_id, milvus.collection_name
    )

    done = _load_checkpoint(args.checkpoint_path)
    logger.info("%d video(s) already embedded per checkpoint %s", len(done), args.checkpoint_path)

    author_map = await _build_author_map(mongo)
    by_video = await _group_keyframes_by_video(mongo)

    pending = [vid for vid in by_video if vid not in done]
    if args.limit_videos is not None:
        pending = pending[: args.limit_videos]

    total_keyframes = sum(len(by_video[v]) for v in pending)
    logger.info("Embedding %d video(s), %d keyframe(s) total", len(pending), total_keyframes)

    start = time.monotonic()
    embedded_keyframes = 0
    for i, video_id in enumerate(pending, start=1):
        keyframe_docs = by_video[video_id]
        author = author_map.get(video_id, "unknown")
        _embed_video(video_id, keyframe_docs, author, encoder, milvus)
        _append_checkpoint(args.checkpoint_path, video_id)
        embedded_keyframes += len(keyframe_docs)
        logger.info(
            "[%d/%d] %s: %d keyframe(s) embedded", i, len(pending), video_id, len(keyframe_docs)
        )

    elapsed = time.monotonic() - start
    await mongo.close()
    if embedded_keyframes:
        logger.info(
            "Done. %d video(s), %d keyframe(s) embedded this run in %.1fs (%.1f keyframes/s).",
            len(pending),
            embedded_keyframes,
            elapsed,
            embedded_keyframes / elapsed,
        )
    else:
        logger.info("Nothing to do -- every video already in the checkpoint.")


if __name__ == "__main__":
    asyncio.run(main())
