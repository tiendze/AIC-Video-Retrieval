"""N3 (asr-phowhisper-batched.plan.md, Nhip 2): delete old large-v3 ASR data
from Elasticsearch `asr_segments` and Milvus `transcripts_bge_m3` before
re-running under PhoWhisper-large-ct2.

Safe to run: N2's backup (data/asr_transcripts.large-v3.json,
data/asr_by_shot.large-v3.json) already captured this data. Uses
AsrEsStore.recreate_index() / TranscriptMilvusStore.drop_collection()
(CLAUDE.md F5) -- never a manual DROP INDEX / row-level delete.

Does NOT load the bge-m3 model (GPU) -- embedding_dim=1024 is a known
constant (app/encoders/bge_m3.py:_EMBEDDING_DIM), so TranscriptMilvusStore
can be constructed without instantiating Bgem3Encoder.

Usage:
  python scripts/n3_reset_asr_stores.py
"""

from __future__ import annotations

import asyncio

from app.config import Settings
from app.stores.es_client import AsrEsStore
from app.stores.milvus_client import TranscriptMilvusStore

_ES_ASR_INDEX_NAME = "asr_segments"
_BGE_M3_MODEL_ID = "bge_m3"
_BGE_M3_EMBEDDING_DIM = 1024
_HNSW_M = 16
_HNSW_EF_CONSTRUCTION = 200
_SEARCH_EF = 64


async def _main() -> None:
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


if __name__ == "__main__":
    asyncio.run(_main())
