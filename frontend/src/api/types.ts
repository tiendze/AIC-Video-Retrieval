// Mirrors the Pydantic response/request schemas backing these endpoints --
// keep in sync by hand (plan §P1.6). Source of truth for each type is noted
// inline; if a backend field renames, this file must change in the same PR.

// app/schemas/request/search.py: SearchRequest
export type SearchMode = 'semantic' | 'ocr'

// app/schemas/response/search.py: SearchHitResponse
export interface SearchHit {
  keyframe_id: string
  video_id: string
  score: number
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
