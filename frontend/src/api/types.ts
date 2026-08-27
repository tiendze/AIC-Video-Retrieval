// Mirrors the Pydantic response/request schemas backing these endpoints --
// keep in sync by hand (plan §P1.6). Source of truth for each type is noted
// inline; if a backend field renames, this file must change in the same PR.

// app/schemas/request/search.py: SearchRequest
export type SearchMode = 'semantic' | 'ocr'

// app/schemas/request/search.py: KisSearchRequest
export type KisQueryType = 'text' | 'ocr' | 'abe'
export type Stage1SourceModality = 'text' | 'ocr' | 'title' | 'asr'

export interface AbePairPayload {
  attribute: string
  object: string
}

export type KisQueryPayload =
  | {
      type: 'text' | 'ocr'
      text: string
    }
  | {
      type: 'abe'
      pairs: AbePairPayload[]
    }

export interface AbePairRow extends AbePairPayload {
  id: string
}

export interface KisQueryRow {
  id: string
  text: string
  type: KisQueryType
  pairs: AbePairRow[]
}

export interface KisCandidatePayload {
  text_query?: string | null
  text_queries?: string[]
  local_phrases_by_query?: string[][]
  text_top_k: number
  ocr_query?: string | null
  ocr_top_k: number
  title_query?: string | null
  title_top_k: number
  asr_query?: string | null
  asr_queries?: string[]
  asr_match_mode?: 'any' | 'all'
  asr_video_top_k: number
}

export interface KisRerankPayload {
  candidate_ids: string[]
  subqueries: KisQueryPayload[]
  final_top_k: number
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

export interface AbePairScore {
  attribute: string
  object: string
  positive_attributes: string[]
  negative_attributes: string[]
}

export interface AbePatchHit {
  label: string
  patch_index: number
  score: number
}

export interface AbePatchHeatmap {
  patch_count: number
  grid_width: number
  grid_height: number
  base_top_patches: AbePatchHit[]
  refined_top_patches: AbePatchHit[]
}

export interface AbeQueryScore {
  query_index: number
  pairs: AbePairScore[]
  base_local: number
  refined_local: number
  binding_delta: number
  local_score: number
  global_abe: number
  raw_score: number
  rank: number
  rank_percentile: number
  patch_heatmap?: AbePatchHeatmap
}

export interface AbeModalityScore {
  geometric_mean: number
  score: number
  queries: AbeQueryScore[]
}

export interface SearchModalities {
  text?: TextModalityScore
  ocr?: OcrModalityScore
  abe?: AbeModalityScore
}

export interface Stage1TextScore {
  query: string
  raw_cosine: number
  rank: number
}

export interface Stage1TextQueryScore {
  query_index: number
  query: string
  raw_cosine: number
  original_rank: number | null
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

export interface Stage1TitleScore {
  query: string
  video_title: string
  title_score: number
  rank: number
  match_type?: string
  matched_text?: string
  matched_tokens?: string[]
}

export interface Stage1RetrievalScore {
  source_modalities: Stage1SourceModality[]
  text?: Stage1TextScore
  text_queries?: Stage1TextQueryScore[]
  ocr?: Stage1OcrScore
  title?: Stage1TitleScore
  fused_image_score?: number
  fused_image_rank?: number
  local_score?: number
  local_query_scores?: number[]
  local_retrieval_used?: boolean
  local_embedding_available?: boolean
}

export interface AsrEvidenceSegment {
  segment_id: string
  text: string
  start_time: number
  end_time: number
}

export interface AsrQueryMatch {
  query: string
  detected: boolean
  matches: AsrEvidenceSegment[]
}

export interface AsrVideoEvidence {
  video_id: string
  detected: boolean
  asr_matches: AsrQueryMatch[]
  segments: AsrEvidenceSegment[]
}

export interface AsrGate {
  query: string
  queries: string[]
  match_mode: 'any' | 'all'
  detected: boolean
  asr_detected_any: boolean
  matched_segment_count: number
  scanned_segment_count: number
  candidate_video_count: number
  asr_matches: AsrQueryMatch[]
  videos: AsrVideoEvidence[]
}

// app/schemas/response/search.py: SearchHitResponse
export interface SearchHit {
  keyframe_id: string
  video_id: string
  score: number
  fused_image_score?: number
  fused_image_rank?: number
  local_score?: number
  local_query_scores?: number[]
  local_retrieval_used?: boolean
  local_embedding_available?: boolean
  video_title?: string
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
  // UI-only decoration sourced from the top-level ASR gate. The backend
  // deliberately does not duplicate this video evidence on every hit.
  asr_evidence?: AsrVideoEvidence
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
  title_candidate_count: number
  rrf_k: number
  results: SearchHit[]
  asr_gate?: AsrGate | null
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

// app/schemas/request/trake.py: TrakeEventRequest. Each event row carries its
// own Text / OCR / Title / ASR boxes, same four channels as KIS Stage 1 --
// TRAKE runs one retrieve_kis_candidates() call per event.
export interface TrakeEventPayload {
  text_query?: string | null
  text_queries?: string[]
  text_top_k?: number
  local_phrases?: string[]
  ocr_query?: string | null
  ocr_top_k?: number
  title_query?: string | null
  title_top_k?: number
  asr_query?: string | null
  asr_queries?: string[]
  asr_match_mode?: 'any' | 'all'
  asr_video_top_k?: number
}

// app/schemas/request/trake.py: TrakeRequest
export interface TrakeSearchPayload {
  events: TrakeEventPayload[]
  top_k: number
  threshold: number
}

// app/schemas/response/trake.py: TrakeKeyframeResponse. `score`, `semantic_z`
// and `kis_rank` are diagnostics: they separate "the encoder liked this frame"
// from "KIS retrieved this frame", which is what an operator being timed needs
// in order to trust or reject a row without reading server logs.
export interface TrakeKeyframe {
  keyframe_id: string
  frame_idx: number
  timestamp: number
  thumbnail_url: string
  image_url: string
  score?: number | null
  semantic_z?: number | null
  kis_rank?: number | null
  // Phrase-to-patch MaxSim diagnostics (only set when the event carried a
  // local phrase and this frame's video was inside local_rerank.max_videos).
  local_score?: number | null
  local_z?: number | null
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
  // Weak link: TRAKE grades every event, so this is the number that predicts
  // whether the whole sequence is right, not `score`'s blend.
  min_score?: number | null
  mean_score?: number | null
  coverage?: number | null
  // Video-level rank score, shared by every sequence of this video.
  video_score?: number | null
}

// app/schemas/response/trake.py: TrakeEventDiagnosticsResponse
export interface TrakeEventDiagnostics {
  event_index: number
  kis_candidate_count: number
  kis_video_count: number
  has_text: boolean
  // True when this event's text exceeded the encoder's 77-token cap and was
  // silently truncated by the tokenizer -- FG-CLIP has no length-agnostic
  // path wired in here (app/encoders/fg_clip.py), so this is a warning, not
  // a fix.
  text_truncated?: boolean
}

// app/schemas/response/trake.py: TrakeResponse. `results` arrives in
// submission order already -- round-robin across videos, not a flat sort by
// sequence score -- so the UI must never re-sort it.
export interface TrakeResult {
  results: TrakeVideoResultItem[]
  events?: TrakeEventDiagnostics[]
  candidate_video_count?: number
  chained_video_count?: number
  dropped_video_count?: number
  // Per-video "Details" view: each video's own top sequences (before the
  // submission cap), keyed by video_id. Not part of the submission; inspection
  // only. See app/schemas/response/trake.py: TrakeResponse.sequences_by_video.
  sequences_by_video?: Record<string, TrakeVideoResultItem[]>
  // Local phrase-to-patch rerank diagnostics for the whole request (B2b).
  local_rerank_video_count?: number
  local_missing_frame_count?: number
}

// app/schemas/response/error.py: ErrorResponse -- every API error takes
// this shape (CLAUDE.md §7).
export interface ApiErrorBody {
  detail: string
  error_code: string
  request_id: string
}

export type AsrStatus = 'unreachable' | 'unavailable' | 'empty' | 'ready'

// app/schemas/response/health.py: HealthResponse -- polled by useHealth
// (plan §P1.8) to detect Elasticsearch being down (degraded mode, PRD T5).
export interface HealthStatus {
  status: 'ok' | 'degraded'
  milvus_ok: boolean
  elasticsearch_ok: boolean
  mongodb_ok: boolean
  asr_status: AsrStatus
  asr_ready: boolean
  asr_segment_count: number | null
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
