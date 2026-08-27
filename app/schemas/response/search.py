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

from pydantic import BaseModel, Field


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


class AbePairScoreResponse(BaseModel):
    attribute: str
    object: str
    positive_attributes: list[str]
    negative_attributes: list[str]


class AbePatchHitResponse(BaseModel):
    label: str
    patch_index: int
    score: float


class AbePatchHeatmapResponse(BaseModel):
    patch_count: int
    grid_width: int
    grid_height: int
    base_top_patches: list[AbePatchHitResponse]
    refined_top_patches: list[AbePatchHitResponse]


class AbeQueryScoreResponse(BaseModel):
    query_index: int
    pairs: list[AbePairScoreResponse]
    base_local: float
    refined_local: float
    binding_delta: float
    local_score: float
    global_abe: float
    raw_score: float
    rank: int
    rank_percentile: float
    patch_heatmap: AbePatchHeatmapResponse | None = None


class TextModalityScoreResponse(BaseModel):
    geometric_mean: float
    score: float
    queries: list[TextQueryScoreResponse]


class OcrModalityScoreResponse(BaseModel):
    geometric_mean: float
    score: float
    queries: list[OcrQueryScoreResponse]


class AbeModalityScoreResponse(BaseModel):
    geometric_mean: float
    score: float
    queries: list[AbeQueryScoreResponse]


class SearchModalitiesResponse(BaseModel):
    text: TextModalityScoreResponse | None = None
    ocr: OcrModalityScoreResponse | None = None
    abe: AbeModalityScoreResponse | None = None


class Stage1TextScoreResponse(BaseModel):
    query: str
    raw_cosine: float
    rank: int


class Stage1TextQueryScoreResponse(BaseModel):
    query_index: int
    query: str
    raw_cosine: float
    original_rank: int | None
    rank: int


class Stage1OcrScoreResponse(BaseModel):
    query: str
    raw_bm25: float
    bm25_reference: float
    bm25_normalized: float
    coverage: float
    query_score: float
    rank: int


class Stage1TitleScoreResponse(BaseModel):
    query: str
    video_title: str
    title_score: float
    rank: int
    match_type: str | None = None
    matched_text: str | None = None
    matched_tokens: list[str] = Field(default_factory=list)


class Stage1RetrievalResponse(BaseModel):
    source_modalities: list[str]
    text: Stage1TextScoreResponse | None = None
    ocr: Stage1OcrScoreResponse | None = None
    title: Stage1TitleScoreResponse | None = None
    text_queries: list[Stage1TextQueryScoreResponse] | None = None
    fused_image_score: float | None = None
    fused_image_rank: int | None = None
    local_score: float | None = None
    local_query_scores: list[float] | None = None
    local_retrieval_used: bool | None = None
    local_embedding_available: bool | None = None


class SearchHitResponse(BaseModel):
    keyframe_id: str
    video_id: str
    score: float
    fused_image_score: float | None = None
    fused_image_rank: int | None = None
    video_title: str | None = None
    frame_idx: int
    timestamp: float
    thumbnail_path: str
    thumbnail_url: str
    image_url: str


class SearchResponse(BaseModel):
    results: list[SearchHitResponse]


class KisSearchHitResponse(SearchHitResponse):
    local_score: float | None = None
    local_query_scores: list[float] | None = None
    local_retrieval_used: bool | None = None
    local_embedding_available: bool | None = None
    final_score: float
    active_modalities: int
    modalities: SearchModalitiesResponse


class KisCandidateHitResponse(SearchHitResponse):
    local_score: float | None = None
    local_query_scores: list[float] | None = None
    local_retrieval_used: bool | None = None
    local_embedding_available: bool | None = None
    stage1: Stage1RetrievalResponse


class AsrEvidenceSegmentResponse(BaseModel):
    segment_id: str
    text: str
    start_time: float
    end_time: float


class AsrQueryMatchResponse(BaseModel):
    query: str
    detected: bool
    matches: list[AsrEvidenceSegmentResponse] = Field(default_factory=list)


class AsrVideoEvidenceResponse(BaseModel):
    video_id: str
    detected: bool
    asr_matches: list[AsrQueryMatchResponse] = Field(default_factory=list)
    segments: list[AsrEvidenceSegmentResponse]


class AsrGateResponse(BaseModel):
    query: str
    queries: list[str]
    match_mode: str
    detected: bool
    asr_detected_any: bool
    matched_segment_count: int
    scanned_segment_count: int
    candidate_video_count: int
    asr_matches: list[AsrQueryMatchResponse] = Field(default_factory=list)
    videos: list[AsrVideoEvidenceResponse]


class KisCandidateResponse(BaseModel):
    candidate_ids: list[str]
    candidate_pool_size: int
    text_candidate_count: int
    ocr_candidate_count: int
    title_candidate_count: int = 0
    rrf_k: int = 60
    results: list[KisCandidateHitResponse]
    asr_gate: AsrGateResponse | None = None


class KisSearchResponse(BaseModel):
    results: list[KisSearchHitResponse]


class KisRerankResponse(BaseModel):
    candidate_pool_size: int
    final_top_k: int
    results: list[KisSearchHitResponse]
