"""CLI: import a Kaggle-produced embed_out/ artifact into Milvus (+ MongoDB
for the primary encoder) — giai đoạn 2 of `.claude/plans/kaggle-offline-embed.plan.md`,
extended by `.claude/plans/encoder-swap-siglip2-pecore.plan.md` §T5 with a
second, non-destructive import path for candidate encoders.

Two modes, selected by --encoder:

  --encoder beit3 (default)
      The PRIMARY pipeline. Runs the parity gate from
      `pipelines/verify_embed_parity.py` (checksum + text-probe cosine), then
      for every `video_id` under `data/embed_out/vectors/`: joins its
      embedding matrix with `data/shots.json` (real shot boundaries ->
      `shot_id`) and `data/map-keyframes/<video_id>.csv`'s `fps` column
      (-> `timestamp`) -- NOT the raw `.mp4`, so this import runs without any
      local video file. Only that CSV's `fps` value is read, never its
      `frame_idx`/`pts_time` rows: this project's own keyframes are a
      different extraction than the organiser's `map-keyframes`, so only the
      video-level constant (`fps`) is safe to reuse from it. Assigns a
      contiguous `global_index`, and writes rows to BOTH Milvus
      (`MilvusStore.upsert`) and MongoDB (`KeyframeDocument` +
      `VideoDocument`) — CLAUDE.md §3: `frame_idx` must travel with every
      keyframe in *both* stores.

  --encoder <candidate> (e.g. siglip2)
      Vectors-only mode. Runs a candidate-specific parity gate (HF repo/
      revision + weight sha256 + text probe — no local checkpoint file to
      hash). Writes ONLY to that encoder's own Milvus collection
      (`keyframes_<model_id>`), reading `global_index`/`frame_idx`/
      `timestamp`/`author`/`batch` from the MongoDB rows the PRIMARY import
      already wrote — never touches MongoDB. This is what keeps a second
      encoder's collection from ever desyncing `global_index` from the
      `s_v`/`e_v` DANTE already relies on (encoder-swap plan §"Quyết định
      chiến lược #2"). Requires the primary (`--encoder beit3`) import to
      have already run for every video this artifact covers.

⚠️ Image probe is NOT yet verified for candidate encoders (encoder-swap plan
§T4's image-probe design is not implemented here) — identity for candidates
currently rests on hf_repo_id/hf_revision + weight_files_sha256 + the text
probe only. `_run_candidate_parity_gate` logs this explicitly every run.

Prerequisites:
  docker compose up -d                     # Milvus + MongoDB
  bash scripts/setup_beit3.sh               # models/beit3/*  (--encoder beit3 parity probe)
  data/embed_out/ downloaded and unzipped from Kaggle
  data/shots.json, data/keyframe/, data/map-keyframes/<video_id>.csv present locally
                                             # (--encoder beit3 only; the raw
                                             # .mp4 is NOT required -- fps is
                                             # read from map-keyframes instead)

WARNING: per CLAUDE.md §10, this writes to a live Milvus/MongoDB and — with
--reset — drops/deletes existing collections (MongoDB rows too, but ONLY for
--encoder beit3; other encoders' --reset only ever drops their own Milvus
collection). Prepared here for you to run yourself, not run automatically.

Usage:
  python scripts/import_embed_out.py --reset
  python scripts/import_embed_out.py --limit-videos 1      # smoke test
  python scripts/import_embed_out.py --device cpu           # no GPU for the probe
  python scripts/import_embed_out.py --encoder siglip2 --embed-out-dir data/embed_out_siglip2
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

from app.config import Settings
from app.encoders.base import MultimodalEncoder
from app.encoders.beit3 import BEiT3Encoder
from app.encoders.factory import build_encoder
from app.stores.milvus_client import MilvusStore
from app.stores.mongo_client import MongoStore
from datasets.aic2026 import AIC2026Adapter
from datasets.base import VideoRef
from pipelines.import_embed_out import (
    build_imported_keyframes,
    build_vectors_only_rows,
    build_video_document,
    discover_video_ids,
    is_video_already_imported,
    load_shot_index,
    load_video_vectors,
    to_keyframe_document,
    to_milvus_vector,
    vectors_only_row_to_milvus_vector,
)
from pipelines.ingest import next_global_index_start
from pipelines.verify_embed_parity import (
    EmbedParityError,
    ParityReport,
    hf_hub_weight_sha256,
    load_manifest,
    load_probe,
    run_parity_probe,
    verify_checksums,
    verify_hf_identity,
    verify_manifest_fields,
    verify_weight_sha256,
)

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

# --encoder <key> -> configs/model/<key>.yaml. "beit3" is the default and
# only key that ever reaches the primary (Mongo-writing) import path below.
_ENCODER_CONFIG_PATHS = {
    "beit3": _BEIT3_CONFIG_PATH,
    "siglip2": _REPO_ROOT / "configs" / "model" / "siglip2.yaml",
    "fg_clip": _REPO_ROOT / "configs" / "model" / "fg_clip.yaml",
}


@dataclass(frozen=True)
class _VectorArtifactIdentity:
    """Encoder identity for explicit no-parity vectors-only imports."""

    model_id: str
    embedding_dim: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--embed-out-dir", default=str(_REPO_ROOT / "data" / "embed_out"), help="Path to embed_out/"
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=_REPO_ROOT / "data",
        help="AIC2026Adapter root (--encoder beit3 only)",
    )
    parser.add_argument(
        "--shots-json",
        type=Path,
        default=_REPO_ROOT / "data" / "shots.json",
        help="(--encoder beit3 only)",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=_REPO_ROOT / "data" / "keyframe",
        help="(--encoder beit3 only)",
    )
    parser.add_argument("--device", default="cuda", help="'cuda' or 'cpu' (parity probe only)")
    parser.add_argument(
        "--encoder",
        choices=sorted(_ENCODER_CONFIG_PATHS),
        default="beit3",
        help=(
            "Which encoder's artifact to import. 'beit3' (default) is the PRIMARY pipeline "
            "and is UNCHANGED from before this flag existed: writes both Milvus and MongoDB. "
            "Any other value runs vectors-only mode: writes ONLY that encoder's own Milvus "
            "collection, reading global_index/frame_idx/timestamp/author/batch from the "
            "MongoDB rows the primary import already wrote for these video_ids — never "
            "touches MongoDB. Requires the primary import to have already run first."
        ),
    )
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
            "--encoder beit3: drop the Milvus collection AND delete every MongoDB "
            "videos/keyframes document before importing (plan §9.5: this keyframe source "
            "replaces the P0.7 TransNetV2 pilot data wholesale). "
            "Any other --encoder: drop ONLY that encoder's own Milvus collection — "
            "MongoDB is NEVER touched by --reset outside the beit3 path."
        ),
    )
    parser.add_argument(
        "--skip-parity-check",
        action="store_true",
        help=(
            "Unsafe vectors-only import for artifacts that only have vectors/ and no "
            "manifest.json or probe/. Only allowed for non-beit3 candidate encoders. "
            "Uses configs/model/<encoder>.yaml's model_id/embedding_dim unless "
            "--model-id/--embedding-dim override them."
        ),
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Override the Milvus collection suffix for --skip-parity-check.",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=None,
        help="Override the vector dimension for --skip-parity-check.",
    )
    return parser.parse_args()


def _load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _run_parity_gate(embed_out_dir: Path, beit3_config: dict, device: str) -> BEiT3Encoder:
    """UNCHANGED from before --encoder existed. Blocks import outright on any
    mismatch (CLAUDE.md §3), returning the encoder built for the probe so its
    identity (model_id/embedding_dim) drives MilvusStore."""
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


def _check_manifest_encoder_kind(manifest: dict, expected: str) -> None:
    """Cross-checks the artifact's own manifest against --encoder BEFORE
    running any candidate-specific verification -- catches "pointed
    --embed-out-dir at the wrong artifact" immediately with a clear message,
    instead of a confusing downstream hf_repo_id/hf_revision mismatch."""
    actual = manifest.get("encoder_kind", "beit3")
    if actual != expected:
        logger.error(
            "BLOCKED: manifest.encoder_kind=%r does not match --encoder %r -- wrong "
            "--embed-out-dir for this --encoder?",
            actual,
            expected,
        )
        sys.exit(1)


def _run_candidate_parity_gate(
    embed_out_dir: Path, manifest: dict, config_path: Path, config: dict, device: str
) -> MultimodalEncoder:
    """Parity gate for non-beit3 (HF-hub-backed) encoders: no local checkpoint
    FILE to hash, so identity is (hf_repo_id, hf_revision) + a hash of the
    weight file(s) actually behind that revision, plus the same text probe
    beit3 uses. Blocks on any mismatch, same as _run_parity_gate."""
    try:
        verify_hf_identity(
            manifest, hf_repo_id=config["hf_repo_id"], hf_revision=config["hf_revision"]
        )
        verify_weight_sha256(
            manifest,
            compute_sha256=hf_hub_weight_sha256(config["hf_repo_id"], config["hf_revision"]),
        )
    except EmbedParityError as exc:
        logger.error("BLOCKED: %s", exc)
        sys.exit(1)

    encoder = build_encoder(config_path, device=device)

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
    logger.info("text parity probe: min_cosine=%.6f passed=%s", report.min_cosine, report.passed)
    if not report.passed:
        logger.error(
            "BLOCKED: text parity probe failed -- refusing to import %s",
            embed_out_dir / "vectors",
        )
        sys.exit(1)

    logger.warning(
        "Image probe chua duoc xac minh (encoder-swap plan T4 chua implement) -- do tin cay "
        "cua artifact nay dua tren hf_revision + weight_files_sha256 + text probe, KHONG "
        "co image probe."
    )
    return encoder


def _infer_embedding_dim(vectors_dir: Path) -> int:
    vector_files = sorted(vectors_dir.glob("*.npy"))
    if not vector_files:
        raise ValueError(f"{vectors_dir}: no .npy vector files found")
    vectors, _ = load_video_vectors(vectors_dir, vector_files[0].stem)
    if vectors.ndim != 2:
        raise ValueError(f"{vector_files[0]}: vectors must be rank-2, got shape {vectors.shape}")
    return int(vectors.shape[1])


def _build_unverified_vector_identity(
    args: argparse.Namespace, config: dict, embed_out_dir: Path
) -> _VectorArtifactIdentity:
    if args.encoder == "beit3":
        logger.error(
            "BLOCKED: --skip-parity-check is only allowed for vectors-only candidate imports, "
            "not the primary --encoder beit3 path that writes MongoDB metadata"
        )
        sys.exit(1)

    model_id = args.model_id or config.get("model_id")
    if not model_id:
        logger.error(
            "BLOCKED: --skip-parity-check needs model_id from configs/model/%s.yaml or --model-id",
            args.encoder,
        )
        sys.exit(1)

    try:
        inferred_dim = _infer_embedding_dim(embed_out_dir / "vectors")
    except ValueError as exc:
        logger.error("BLOCKED: %s", exc)
        sys.exit(1)

    embedding_dim = args.embedding_dim or config.get("embedding_dim") or inferred_dim
    if int(embedding_dim) != inferred_dim:
        logger.error(
            "BLOCKED: embedding_dim=%s but the first vector file has dim=%s",
            embedding_dim,
            inferred_dim,
        )
        sys.exit(1)

    logger.warning(
        "UNVERIFIED import: skipping manifest/probe checks for encoder=%s, "
        "model_id=%s, embedding_dim=%s. Make sure query-time uses the exact "
        "same FG_CLIP checkpoint/tokenizer/preprocess that produced %s.",
        args.encoder,
        model_id,
        embedding_dim,
        embed_out_dir / "vectors",
    )
    return _VectorArtifactIdentity(model_id=str(model_id), embedding_dim=int(embedding_dim))


def _build_milvus_store(
    settings: Settings, encoder: MultimodalEncoder | _VectorArtifactIdentity
) -> MilvusStore:
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
    adapter: AIC2026Adapter,
    image_dir: Path,
    milvus: MilvusStore,
    mongo: MongoStore,
    global_index_start: int,
) -> int:
    """Imports one video's keyframes into Milvus + MongoDB. Returns the
    number of keyframes written, so the caller can advance global_index_start
    for the next video without a DB round trip per video.

    `fps` always comes from `data/map-keyframes/<video_id>.csv` (the
    organiser's `fps` column -- a video-level constant, not a per-keyframe
    one) via `adapter.load_reference_keyframes`, never from decoding the raw
    `.mp4` -- this import must run without any local video file. Only `.fps`
    is read off that CSV; `frame_idx`/`pts_time` are never used here, since
    this project's own keyframes are a different extraction than the
    organiser's `map-keyframes`.

    `video_map` (built from `adapter.iter_videos("all")`, which requires the
    `.mp4` to exist) still supplies title/author/publish_date/... when
    available. When it doesn't have this `video_id` -- no local `.mp4` --
    this falls back to a metadata-less `VideoRef` (title/author/... left
    `None`, which `build_video_document` already treats as optional) rather
    than raising: `frame_idx`/`global_index`/`shot_id` never depended on the
    `.mp4` in the first place, only `fps` did, and that now comes from
    map-keyframes."""
    shot_ranges = shot_index.get(video_id)
    if shot_ranges is None:
        raise ValueError(f"{video_id}: in embed_out/vectors/ but missing from data/shots.json")

    reference_keyframes = adapter.load_reference_keyframes(video_id)
    if not reference_keyframes:
        raise ValueError(f"{video_id}: data/map-keyframes/{video_id}.csv has no rows")
    fps = reference_keyframes[0].fps

    video_ref = video_map.get(video_id)
    if video_ref is None:
        video_ref = VideoRef(
            video_id=video_id,
            source_path="",
            duration_sec=reference_keyframes[-1].frame_idx / fps,
        )

    vectors, jsonl_rows = load_video_vectors(embed_out_dir / "vectors", video_id)
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


async def _import_video_vectors_only(
    video_id: str, *, embed_out_dir: Path, milvus: MilvusStore, mongo: MongoStore
) -> int:
    """Imports one video's candidate-encoder vectors into Milvus ONLY —
    global_index/frame_idx/timestamp/author/batch all come from MongoDB,
    which the PRIMARY (--encoder beit3) import already wrote. Never calls
    mongo.bulk_insert_keyframes / mongo.upsert_video — see module docstring
    and encoder-swap plan §"Quyết định chiến lược #2"."""
    vectors, jsonl_rows = load_video_vectors(embed_out_dir / "vectors", video_id)
    mongo_keyframe_docs = await mongo.get_keyframes_for_video(video_id)
    if not mongo_keyframe_docs:
        raise ValueError(
            f"{video_id}: no keyframes in MongoDB -- import the primary (--encoder beit3) "
            "artifact for this video first"
        )
    video_doc = await mongo.get_video(video_id)
    author = (video_doc or {}).get("author", "")

    rows = build_vectors_only_rows(video_id, vectors, jsonl_rows, mongo_keyframe_docs, author)
    milvus.upsert([vectors_only_row_to_milvus_vector(r) for r in rows])
    return len(rows)


async def _reset_stores(milvus: MilvusStore, mongo: MongoStore) -> None:
    """--encoder beit3 --reset ONLY. UNCHANGED from before --encoder existed."""
    logger.warning(
        "--reset: dropping Milvus collection %s and MongoDB videos/keyframes",
        milvus.collection_name,
    )
    milvus.drop_collection()
    await mongo.keyframes.delete_many({})
    await mongo.videos.delete_many({})


def _reset_milvus_only(milvus: MilvusStore) -> None:
    """Any non-beit3 --encoder --reset. Deliberately never touches MongoDB —
    vectors-only mode's whole point is that MongoDB stays exactly what the
    primary import wrote (encoder-swap plan risk R11)."""
    logger.warning(
        "--reset: dropping Milvus collection %s (vectors-only mode -- MongoDB untouched)",
        milvus.collection_name,
    )
    milvus.drop_collection()


async def _run_primary_import(
    args: argparse.Namespace,
    video_ids: list[str],
    embed_out_dir: Path,
    milvus: MilvusStore,
    mongo: MongoStore,
) -> int:
    """UNCHANGED from before --encoder existed (extracted from the former
    _main() body verbatim, so --encoder beit3's behavior is identical)."""
    adapter = AIC2026Adapter(dataset_root=args.dataset_root, eval_set_path=Path("/dev/null"))
    video_map = {v.video_id: v for v in adapter.iter_videos("all")}
    shot_index = load_shot_index(args.shots_json)

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
                adapter=adapter,
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


async def _run_vectors_only_import(
    args: argparse.Namespace,
    video_ids: list[str],
    embed_out_dir: Path,
    milvus: MilvusStore,
    mongo: MongoStore,
) -> int:
    """No resume/checkpoint bookkeeping here, unlike _run_primary_import:
    MilvusStore.upsert is idempotent per keyframe_id regardless, and unlike
    the primary path's GPU-bound embed step, re-upserting an already-done
    video here just re-reads a small .npy/.jsonl pair and re-sends the same
    vectors — cheap enough that a full re-run on retry is an acceptable
    simplification, not a footgun."""
    total_keyframes = 0
    failed_video_ids: list[str] = []
    for video_id in video_ids:
        try:
            written = await _import_video_vectors_only(
                video_id, embed_out_dir=embed_out_dir, milvus=milvus, mongo=mongo
            )
        except (ValueError, OSError) as exc:
            logger.error("import failed for video_id=%s: %s", video_id, exc)
            failed_video_ids.append(video_id)
            continue
        total_keyframes += written
        logger.info("[%s] %s: %d keyframe(s) upserted", args.encoder, video_id, written)

    logger.info(
        "Done (vectors-only, encoder=%s): %d video(s) imported, %d keyframe(s) written, %d failed",
        args.encoder,
        len(video_ids) - len(failed_video_ids),
        total_keyframes,
        len(failed_video_ids),
    )
    if failed_video_ids:
        logger.warning("Failed video_id(s): %s", failed_video_ids)

    await mongo.close()
    return 1 if failed_video_ids else 0


async def _main() -> int:
    args = _parse_args()
    settings = Settings()
    embed_out_dir = Path(args.embed_out_dir)
    config_path = _ENCODER_CONFIG_PATHS[args.encoder]
    config = _load_yaml(config_path)

    if args.skip_parity_check:
        encoder = _build_unverified_vector_identity(args, config, embed_out_dir)
    elif args.encoder == "beit3":
        encoder = _run_parity_gate(embed_out_dir, config, args.device)
    else:
        manifest = load_manifest(embed_out_dir / "manifest.json")
        _check_manifest_encoder_kind(manifest, args.encoder)
        encoder = _run_candidate_parity_gate(
            embed_out_dir, manifest, config_path, config, args.device
        )

    milvus = _build_milvus_store(settings, encoder)
    mongo = MongoStore(uri=settings.mongo_uri)
    await mongo.ensure_indexes()

    if args.reset:
        if args.encoder == "beit3":
            await _reset_stores(milvus, mongo)
        else:
            _reset_milvus_only(milvus)

    video_ids = discover_video_ids(embed_out_dir / "vectors")
    if args.limit_videos is not None:
        video_ids = video_ids[: args.limit_videos]
    logger.info(
        "Importing %d video(s) from %s (encoder=%s)", len(video_ids), embed_out_dir, args.encoder
    )

    if args.encoder == "beit3":
        return await _run_primary_import(args, video_ids, embed_out_dir, milvus, mongo)
    return await _run_vectors_only_import(args, video_ids, embed_out_dir, milvus, mongo)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    sys.exit(asyncio.run(_main()))


if __name__ == "__main__":
    main()
