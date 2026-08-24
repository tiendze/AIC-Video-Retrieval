"""Owner-only destructive reset of Elasticsearch `asr_segments` and Milvus
`transcripts_bge_m3` for a separately reviewed ASR model replacement.

This legacy N3 operation is not a routine import prerequisite. It recreates
the Elasticsearch index and drops the Milvus collection. Run it only as an
explicit owner action after reviewing the replacement procedure and supplying
existing, non-empty Elasticsearch/Milvus backups plus the reviewed restore
procedure. No backup path is assumed to exist in a clean checkout.

Does NOT load the bge-m3 model (GPU) -- embedding_dim=1024 is a known
constant (app/encoders/bge_m3.py:_EMBEDDING_DIM), so TranscriptMilvusStore
can be constructed without instantiating Bgem3Encoder.

Usage:
  python scripts/n3_reset_asr_stores.py --confirm-owner-reviewed-reset `
    --elasticsearch-backup <path> --milvus-backup <path> `
    --restore-procedure <path>
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence
from pathlib import Path

from app.config import Settings
from app.stores.es_client import AsrEsStore
from app.stores.milvus_client import TranscriptMilvusStore

_ES_ASR_INDEX_NAME = "asr_segments"
_BGE_M3_MODEL_ID = "bge_m3"
_BGE_M3_EMBEDDING_DIM = 1024
_HNSW_M = 16
_HNSW_EF_CONSTRUCTION = 200
_SEARCH_EF = 64


def _existing_nonempty_file(value: str) -> Path:
    path = Path(value)
    if not path.is_file() or path.stat().st_size == 0:
        raise argparse.ArgumentTypeError(f"expected an existing non-empty file: {path}")
    return path


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--confirm-owner-reviewed-reset",
        action="store_true",
        required=True,
        help="confirm the project owner separately reviewed this destructive replacement",
    )
    parser.add_argument(
        "--elasticsearch-backup",
        type=_existing_nonempty_file,
        required=True,
        help="existing non-empty backup input for restoring the ASR Elasticsearch index",
    )
    parser.add_argument(
        "--milvus-backup",
        type=_existing_nonempty_file,
        required=True,
        help="existing non-empty backup input for restoring the transcript Milvus collection",
    )
    parser.add_argument(
        "--restore-procedure",
        type=_existing_nonempty_file,
        required=True,
        help="existing non-empty owner-reviewed restore procedure",
    )
    return parser.parse_args(argv)


async def _main(args: argparse.Namespace) -> None:
    print(
        "Owner-confirmed destructive ASR reset with restore inputs: "
        f"{args.elasticsearch_backup}, {args.milvus_backup}, {args.restore_procedure}"
    )
    settings = Settings()

    es = AsrEsStore(
        url=settings.elasticsearch_url,
        index_name=_ES_ASR_INDEX_NAME,
        api_key=settings.elastic_api_key.get_secret_value() if settings.elastic_api_key else None,
    )
    await es.recreate_index()
    await es.close()
    print(f"Elasticsearch: recreated empty index '{_ES_ASR_INDEX_NAME}'")

    milvus = TranscriptMilvusStore(
        uri=settings.milvus_uri,
        token=settings.milvus_token.get_secret_value(),
        model_id=_BGE_M3_MODEL_ID,
        embedding_dim=_BGE_M3_EMBEDDING_DIM,
        hnsw_m=_HNSW_M,
        hnsw_ef_construction=_HNSW_EF_CONSTRUCTION,
        search_ef=_SEARCH_EF,
    )
    milvus.drop_collection()
    print(f"Milvus: recreated empty collection '{milvus.collection_name}'")


def main() -> None:
    asyncio.run(_main(_parse_args()))


if __name__ == "__main__":
    main()
