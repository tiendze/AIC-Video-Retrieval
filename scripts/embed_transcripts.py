"""CLI: embed ASR transcript segments into a dense vector store (P0.11b).

Reads segments back from Elasticsearch (P0.11a's `asr_segments` index) via
the scroll API -- never re-runs Whisper (transcription is the expensive,
irreversible step; embedding is cheap to redo if the encoder changes).
Requires scripts/run_asr.py to have indexed at least some segments first.

WARNING: GPU-heavy (bge-m3 download + inference) -- per CLAUDE.md section 10,
prepared here for you to run yourself, not run automatically. Sequential
only on 8GB VRAM if run alongside scripts/run_asr.py or scripts/run_ingest.py.

Prerequisites:
  docker compose up -d          # Elasticsearch + Milvus (MongoDB not needed here)
  pip install -e ".[asr-embed]"

Usage:
  python scripts/embed_transcripts.py
  python scripts/embed_transcripts.py --device cpu
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import yaml

from app.config import Settings
from app.encoders.bge_m3 import Bgem3Encoder
from app.stores.es_client import AsrEsStore
from app.stores.milvus_client import TranscriptMilvusStore
from pipelines.embed_transcripts import TranscriptToEmbed, embed_and_write_transcripts

# Windows consoles often default to cp1252, which cannot encode the
# Vietnamese transcript text this project logs -- same fix as run_ingest.py.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_BGE_M3_CONFIG_PATH = _REPO_ROOT / "configs" / "model" / "bge_m3.yaml"
_MILVUS_CONFIG_PATH = _REPO_ROOT / "configs" / "stores" / "milvus.yaml"
_ES_ASR_INDEX_NAME = "asr_segments"
_UPSERT_CHUNK_SIZE = 256  # segments per Milvus upsert call, not the encoder's own GPU batch size


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--device", default="cuda", help="'cuda' or 'cpu'")
    return parser.parse_args()


def _load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _to_transcript_to_embed(doc: dict) -> TranscriptToEmbed:
    return TranscriptToEmbed(
        segment_id=doc["segment_id"],
        text=doc["text"],
        video_id=doc["video_id"],
        start_frame_idx=doc["start_frame_idx"],
        end_frame_idx=doc["end_frame_idx"],
        start_time=doc["start_time"],
        end_time=doc["end_time"],
    )


async def _main() -> int:
    args = _parse_args()
    settings = Settings()

    bge_m3_config = _load_yaml(_BGE_M3_CONFIG_PATH)
    encoder = Bgem3Encoder(
        model_name=bge_m3_config["model_name"],
        device=args.device,
        batch_size=bge_m3_config["batch_size"],
        model_id=bge_m3_config["model_id"],
    )

    milvus_config = _load_yaml(_MILVUS_CONFIG_PATH)
    milvus = TranscriptMilvusStore(
        uri=settings.milvus_uri,
        token=settings.milvus_token.get_secret_value(),
        model_id=encoder.model_id,
        embedding_dim=encoder.embedding_dim,
        hnsw_m=milvus_config["hnsw_m"],
        hnsw_ef_construction=milvus_config["hnsw_ef_construction"],
        search_ef=milvus_config["search_ef"],
    )

    es = AsrEsStore(
        url=settings.elasticsearch_url,
        index_name=_ES_ASR_INDEX_NAME,
        api_key=settings.elastic_api_key.get_secret_value() if settings.elastic_api_key else None,
    )

    total_embedded = 0
    chunk: list[TranscriptToEmbed] = []
    async for doc in es.scroll_all():
        chunk.append(_to_transcript_to_embed(doc))
        if len(chunk) >= _UPSERT_CHUNK_SIZE:
            embed_and_write_transcripts(chunk, encoder, milvus)
            total_embedded += len(chunk)
            logger.info("Embedded %d segment(s) so far", total_embedded)
            chunk = []
    if chunk:
        embed_and_write_transcripts(chunk, encoder, milvus)
        total_embedded += len(chunk)

    logger.info("Done: %d segment(s) embedded into %s", total_embedded, milvus.collection_name)
    await es.close()
    return 0


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    sys.exit(asyncio.run(_main()))


if __name__ == "__main__":
    main()
