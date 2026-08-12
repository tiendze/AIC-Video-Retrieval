"""Integration test for `SearchService.search_by_image` (plan §P1.1, verify
step `test_uploaded_keyframe_retrieves_itself_as_top1`) -- proves the
image-to-image invariant end to end: a keyframe embedded at "ingest" time and
the SAME image bytes re-uploaded through the query path must land as the #1
Milvus hit. A top-1 miss here means query-time preprocessing has drifted from
ingest-time preprocessing (CLAUDE.md §3) -- a silent failure, not a crash.

Requires the real BEiT-3 checkpoint (torch/timm, plan §P0.5 timebox) AND the
docker-compose stack (Milvus + MongoDB) running -- neither exists on a bare
host, so this whole module is skipped unless both are present:

    docker compose up -d
    BEIT3_CHECKPOINT_PATH=models/beit3/beit3_base_itc_patch16_224.pth \
    BEIT3_SPM_PATH=models/beit3/beit3.spm \
    pytest -m "integration and model" tests/integration/test_search_service_image.py -v
"""

from __future__ import annotations

import io
import os
import uuid
from collections.abc import AsyncIterator

import numpy as np
import pytest
from PIL import Image
from pymilvus import MilvusClient
from pymongo import AsyncMongoClient

from app.services.search_service import SearchService
from app.stores.es_client import EsStore
from app.stores.milvus_client import KeyframeVector, MilvusStore
from app.stores.mongo_client import KeyframeDocument, MongoStore

torch = pytest.importorskip("torch")

pytestmark = [pytest.mark.integration, pytest.mark.model]

_CHECKPOINT_PATH = os.environ.get("BEIT3_CHECKPOINT_PATH")
_SPM_PATH = os.environ.get("BEIT3_SPM_PATH")

if not _CHECKPOINT_PATH or not _SPM_PATH:
    pytest.skip(
        "BEIT3_CHECKPOINT_PATH / BEIT3_SPM_PATH not set — run scripts/setup_beit3.sh first",
        allow_module_level=True,
    )

# See tests/integration/test_milvus_client.py / test_mongo_client.py for why
# these are overridable via env var.
_MILVUS_URI = os.environ.get("MILVUS_URI", "http://localhost:19530")
_MILVUS_TOKEN = os.environ.get("MILVUS_TOKEN", "root:Milvus")
_MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
_ES_URL = os.environ.get("ES_URL", "http://localhost:9200")
_HNSW_M = 16
_HNSW_EF_CONSTRUCTION = 200
_SEARCH_EF = 64


@pytest.fixture(scope="module")
def encoder():
    from app.encoders.beit3 import BEiT3Encoder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return BEiT3Encoder(
        checkpoint_path=_CHECKPOINT_PATH, sentencepiece_model_path=_SPM_PATH, device=device
    )


@pytest.fixture()
def milvus_store(encoder) -> AsyncIterator[MilvusStore]:
    model_id = f"test_{uuid.uuid4().hex[:8]}"
    created = MilvusStore(
        uri=_MILVUS_URI,
        token=_MILVUS_TOKEN,
        model_id=model_id,
        embedding_dim=encoder.embedding_dim,
        hnsw_m=_HNSW_M,
        hnsw_ef_construction=_HNSW_EF_CONSTRUCTION,
        search_ef=_SEARCH_EF,
    )
    yield created
    MilvusClient(uri=_MILVUS_URI, token=_MILVUS_TOKEN).drop_collection(created.collection_name)


@pytest.fixture()
async def mongo_store() -> AsyncIterator[MongoStore]:
    database_name = f"test_{uuid.uuid4().hex[:8]}"
    created = MongoStore(uri=_MONGO_URI, database_name=database_name)
    await created.ensure_indexes()
    yield created

    cleanup_client: AsyncMongoClient = AsyncMongoClient(_MONGO_URI)
    await cleanup_client.drop_database(database_name)
    await cleanup_client.close()
    await created.close()


async def test_uploaded_keyframe_retrieves_itself_as_top1(encoder, milvus_store, mongo_store):
    # A "decoy" keyframe embedded first so top-1 is a real discrimination,
    # not a trivial single-row search.
    decoy_image = Image.new("RGB", (224, 224), color=(200, 30, 30))
    answer_image = Image.new("RGB", (224, 224), color=(30, 150, 60))
    decoy_vector, answer_vector = encoder.encode_image([decoy_image, answer_image])

    milvus_store.upsert(
        [
            KeyframeVector(
                keyframe_id="decoy_f0",
                embedding=decoy_vector,
                video_id="decoy_video",
                global_index=0,
                frame_idx=0,
                timestamp=0.0,
                author="channel_a",
                batch="L21",
            ),
            KeyframeVector(
                keyframe_id="answer_f5",
                embedding=answer_vector,
                video_id="answer_video",
                global_index=1,
                frame_idx=5,
                timestamp=0.2,
                author="channel_b",
                batch="L21",
            ),
        ]
    )
    await mongo_store.bulk_insert_keyframes(
        [
            KeyframeDocument(
                keyframe_id="answer_f5",
                video_id="answer_video",
                shot_id="answer_video_shot0",
                frame_idx=5,
                timestamp=0.2,
                global_index=1,
                image_path="answer_f5.jpg",
                thumbnail_path="thumbs/answer_f5.jpg",
            )
        ]
    )

    service = SearchService(
        encoder=encoder,
        milvus_store=milvus_store,
        es_store=EsStore(url=_ES_URL, index_name="unused_by_this_test"),
        mongo_store=mongo_store,
    )

    # Round-trip through PNG bytes -- the same decode path `POST
    # /api/search/image` uses, not the raw in-memory `answer_image` object,
    # so this actually exercises query-time preprocessing.
    buf = io.BytesIO()
    answer_image.save(buf, format="PNG")
    buf.seek(0)
    reloaded = Image.open(buf)
    reloaded.load()

    results = await service.search_by_image(reloaded, top_k=5)

    assert results[0].keyframe_id == "answer_f5"
    assert np.isclose(results[0].score, 1.0, atol=1e-3)  # cosine self-similarity
