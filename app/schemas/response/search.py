"""Response shape for `POST /api/search` (plan §P0.8) -- every field the
P0.8 verify step requires: score, video_id, timestamp, keyframe_id,
thumbnail_path, plus frame_idx (CLAUDE.md §3: mandatory on every keyframe
result).

`thumbnail_url`/`image_url` (plan §P1.6) are added on top of the original
`thumbnail_path` -- the frontend fetches images through
`GET /api/keyframes/{id}/thumbnail`/`/image`, never by reading a raw
filesystem path (which may not even exist client-side, and differs between
the two ingest paths' directory conventions). `thumbnail_path` is kept as-is
for existing callers/tests/debugging."""

from __future__ import annotations

from pydantic import BaseModel


class TextQueryScoreResponse(BaseModel):
    query_index: int
    query: str
    raw_cosine: float
    rank: int
    rank_percentile: float


class OcrQueryScoreResponse(BaseModel):
    query_index: int
    query: str
    raw_bm25: float
    bm25_reference: float
    bm25_normalized: float
    coverage: float
    query_score: float
    rank: int
    rank_percentile: float


class TextModalityScoreResponse(BaseModel):
    geometric_mean: float
    score: float
    queries: list[TextQueryScoreResponse]


class OcrModalityScoreResponse(BaseModel):
    geometric_mean: float
    score: float
    queries: list[OcrQueryScoreResponse]


class SearchModalitiesResponse(BaseModel):
    text: TextModalityScoreResponse | None = None
    ocr: OcrModalityScoreResponse | None = None


class Stage1TextScoreResponse(BaseModel):
    query: str
    raw_cosine: float
    rank: int


class Stage1OcrScoreResponse(BaseModel):
    query: str
    raw_bm25: float
    bm25_reference: float
    bm25_normalized: float
    coverage: float
    query_score: float
    rank: int


class Stage1RetrievalResponse(BaseModel):
    source_modalities: list[str]
    text: Stage1TextScoreResponse | None = None
    ocr: Stage1OcrScoreResponse | None = None


class SearchHitResponse(BaseModel):
    keyframe_id: str
    video_id: str
    score: float
    frame_idx: int
    timestamp: float
    thumbnail_path: str
    thumbnail_url: str
    image_url: str


class SearchResponse(BaseModel):
    results: list[SearchHitResponse]


class KisSearchHitResponse(SearchHitResponse):
    final_score: float
    active_modalities: int
    modalities: SearchModalitiesResponse


class KisCandidateHitResponse(SearchHitResponse):
    stage1: Stage1RetrievalResponse


class KisCandidateResponse(BaseModel):
    candidate_ids: list[str]
    candidate_pool_size: int
    text_candidate_count: int
    ocr_candidate_count: int
    results: list[KisCandidateHitResponse]


class KisSearchResponse(BaseModel):
    results: list[KisSearchHitResponse]


class KisRerankResponse(BaseModel):
    candidate_pool_size: int
    final_top_k: int
    results: list[KisSearchHitResponse]
