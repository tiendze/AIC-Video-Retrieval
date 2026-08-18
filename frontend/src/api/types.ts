// Mirrors the Pydantic response/request schemas backing these endpoints --
// keep in sync by hand (plan §P1.6). Source of truth for each type is noted
// inline; if a backend field renames, this file must change in the same PR.

// app/schemas/request/search.py: SearchRequest
export type SearchMode = 'semantic' | 'ocr'

// app/schemas/request/search.py: KisSearchRequest
export type KisQueryType = 'text' | 'ocr'

export interface KisQueryPayload {
  type: KisQueryType
  text: string
}

export interface KisCandidatePayload {
  text_query?: string | null
  text_top_k: number
  ocr_query?: string | null
  ocr_top_k: number
}

export interface KisRerankPayload {
  candidate_ids: string[]
  subqueries: KisQueryPayload[]
  final_top_k: number
}

// UI state for the editable KIS Stage 2 subquery list.
export interface KisQueryRow extends KisQueryPayload {
  id: string
}

// UI-only: KIS uses the two-stage /api/search/kis/* flow, while TRAKE/DANTE
// uses POST /api/trake with ordered event rows.
export type AppMode = 'kis' | 'dante'

export interface TextQueryScore {
  query_index: number
  query: string
  raw_cosine: number
  rank: number
  rank_percentile: number
}

export interface OcrQueryScore {
  query_index: number
  query: string
  raw_bm25: number
  bm25_reference: number
  bm25_normalized: number
  coverage: number
  query_score: number
  rank: number
  rank_percentile: number
}

export interface TextModalityScore {
  geometric_mean: number
  score: number
  queries: TextQueryScore[]
}

export interface OcrModalityScore {
  geometric_mean: number
  score: number
  queries: OcrQueryScore[]
}

export interface SearchModalities {
  text?: TextModalityScore
  ocr?: OcrModalityScore
}

export interface Stage1TextScore {
  query: string
  raw_cosine: number
  rank: number
}

export interface Stage1OcrScore {
  query: string
  raw_bm25: number
  bm25_reference: number
  bm25_normalized: number
  coverage: number
  query_score: number
  rank: number
}

export interface Stage1RetrievalScore {
  source_modalities: KisQueryType[]
  text?: Stage1TextScore
  ocr?: Stage1OcrScore
}

// app/schemas/response/search.py: SearchHitResponse
export interface SearchHit {
  keyframe_id: string
  video_id: string
  score: number
  final_score?: number
  active_modalities?: number
  modalities?: SearchModalities
  stage1?: Stage1RetrievalScore
  candidate_pool_size?: number
  frame_idx: number
  timestamp: number
  thumbnail_path: string
  thumbnail_url: string
  image_url: string
}

// app/schemas/response/search.py: SearchResponse
export interface SearchResponse {
  results: SearchHit[]
}

export interface KisCandidateResponse {
  candidate_ids: string[]
  candidate_pool_size: number
  text_candidate_count: number
  ocr_candidate_count: number
  results: SearchHit[]
}

export interface KisRerankResponse {
  candidate_pool_size: number
  final_top_k: number
  results: SearchHit[]
}

// app/schemas/response/quest.py: QueryRewriteResponse -- QUEST branch 1
// (PRD §F5). The operator picks which of original_query/rewritten_query to
// actually search with; this endpoint never searches by itself.
// rewritten_query is null exactly when status is "fallback" (Gemini
// error/timeout/quota) -- never treated as an error, the operator still has
// original_query to search with.
export interface QueryRewriteResult {
  status: 'rewritten' | 'fallback'
  original_query: string
  rewritten_query: string | null
}

// app/schemas/response/quest.py: ImageCandidateResponse -- QUEST branch 2
// (PRD §F5). thumbnail_url is the real external web URL (safe to render
// directly in <img>); the server keeps the full-res image_url private and
// only downloads it itself once the operator picks a candidate_id (SSRF
// guard, app/services/quest_service.py).
export interface ImageCandidate {
  candidate_id: string
  thumbnail_url: string
  title: string
}

// app/schemas/response/quest.py: ImageCandidatesResponse. status="fallback"
// means the web image search itself failed/timed out -- fallback_hits are
// regular semantic-search results for the same query, ready to render
// immediately with no candidate to pick (candidates is always []).
export interface ImageCandidatesResult {
  status: 'candidates' | 'fallback'
  candidate_set_id: string | null
  candidates: ImageCandidate[]
  fallback_hits: SearchHit[]
}

// app/schemas/response/quest.py: ImageSearchResponse -- result of searching
// by the operator's picked candidate_id.
export interface ImageSearchResult {
  status: 'ok' | 'fallback'
  hits: SearchHit[]
}

// app/schemas/response/trake.py: TrakeKeyframeResponse
export interface TrakeKeyframe {
  keyframe_id: string
  frame_idx: number
  timestamp: number
  thumbnail_url: string
  image_url: string
}

// app/schemas/response/trake.py: TrakeVideoResultResponse. Each item is one
// sequence candidate; multiple items may share a video_id. `keyframes` has
// exactly N entries (N = number of events submitted) in strict time order
// -- never re-sort this array (CLAUDE.md §3: DANTE's ordering invariant,
// τ < t). This is also the exact AIC submission shape for TRAKE:
// `<video_id>, <frame_id₁>, …, <frame_idₙ>` (references.md §1.2).
export interface TrakeVideoResultItem {
  video_id: string
  score: number
  keyframes: TrakeKeyframe[]
}

// app/schemas/response/trake.py: TrakeResponse
export interface TrakeResult {
  results: TrakeVideoResultItem[]
}

// app/schemas/response/error.py: ErrorResponse -- every API error takes
// this shape (CLAUDE.md §7).
export interface ApiErrorBody {
  detail: string
  error_code: string
  request_id: string
}

// app/schemas/response/health.py: HealthResponse -- polled by useHealth
// (plan §P1.8) to detect Elasticsearch being down (degraded mode, PRD T5).
export interface HealthStatus {
  status: 'ok' | 'degraded'
  milvus_ok: boolean
  elasticsearch_ok: boolean
  mongodb_ok: boolean
}

// app/schemas/response/video.py: VideoResponse
export interface VideoInfo {
  video_id: string
  watch_url: string
  title: string
  author: string
  publish_date: string
  duration: number
  fps: number
  num_keyframes: number
  s_v: number
  e_v: number
}

// app/schemas/response/video.py: VideoKeyframeResponse
export interface VideoKeyframe {
  keyframe_id: string
  frame_idx: number
  timestamp: number
  global_index: number
  thumbnail_url: string
  image_url: string
}

// app/schemas/response/video.py: VideoKeyframesResponse
export interface VideoKeyframesResponse {
  keyframes: VideoKeyframe[]
}
