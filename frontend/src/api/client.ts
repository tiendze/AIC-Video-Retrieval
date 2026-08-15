import type {
  ApiErrorBody,
  HealthStatus,
  ImageCandidatesResult,
  ImageSearchResult,
  QueryRewriteResult,
  SearchMode,
  SearchResponse,
  TrakeResult,
  VideoInfo,
  VideoKeyframesResponse,
} from './types'

// Thrown for every non-2xx response -- callers branch on `errorCode`
// (app/schemas/response/error.py's {detail, error_code, request_id}).
export class ApiError extends Error {
  readonly status: number
  readonly errorCode: string
  readonly requestId: string

  constructor(status: number, body: ApiErrorBody) {
    super(body.detail)
    this.status = status
    this.errorCode = body.error_code
    this.requestId = body.request_id
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init)
  if (!response.ok) {
    const body = (await response.json()) as ApiErrorBody
    throw new ApiError(response.status, body)
  }
  return response.json() as Promise<T>
}

// POST /api/search (app/api/search.py)
export function search(
  query: string,
  mode: SearchMode,
  topK: number,
  signal?: AbortSignal,
): Promise<SearchResponse> {
  return request<SearchResponse>('/api/search', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, mode, top_k: topK }),
    signal,
  })
}

// POST /api/search/image (app/api/search.py) -- used both for direct
// upload and for "Find Similar Images" (plan §P1.6: fetch a result's own
// full image, then re-search with it as the query).
export function searchByImage(
  file: Blob,
  topK: number,
  signal?: AbortSignal,
): Promise<SearchResponse> {
  const formData = new FormData()
  formData.append('file', file)
  formData.append('top_k', String(topK))
  return request<SearchResponse>('/api/search/image', {
    method: 'POST',
    body: formData,
    signal,
  })
}

// POST /api/quest/rewrite-query (app/api/quest.py) -- QUEST branch 1 (PRD
// §F5). Never triggers a search itself; the caller decides which of
// original_query/rewritten_query to search with (see QueryRewriteResult).
export function rewriteQuery(query: string, signal?: AbortSignal): Promise<QueryRewriteResult> {
  return request<QueryRewriteResult>('/api/quest/rewrite-query', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query }),
    signal,
  })
}

// POST /api/quest/image-candidates (app/api/quest.py) -- QUEST branch 2
// (PRD §F5). thumbnail_url in the response is the real external web
// thumbnail; the server never hands out the full-res image URL (SSRF
// guard).
export function findImageCandidates(
  query: string,
  topK: number,
  signal?: AbortSignal,
): Promise<ImageCandidatesResult> {
  return request<ImageCandidatesResult>('/api/quest/image-candidates', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, top_k: topK }),
    signal,
  })
}

// POST /api/quest/image-search (app/api/quest.py) -- runs image-to-image
// search on the operator's picked candidate_id (never auto-picked, PRD
// §F5 acceptance criteria).
export function searchByCandidate(
  candidateSetId: string,
  candidateId: string,
  topK: number,
  signal?: AbortSignal,
): Promise<ImageSearchResult> {
  return request<ImageSearchResult>('/api/quest/image-search', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      candidate_set_id: candidateSetId,
      candidate_id: candidateId,
      top_k: topK,
    }),
    signal,
  })
}

// POST /api/trake (app/api/trake.py) -- DANTE-backed TRAKE search.
// `events` must already be in time order; the response's keyframes come
// back in that same order, never re-sorted (CLAUDE.md §3). `lambdaPenalty`
// null means "use configs/search/dante.yaml's configured default".
export function trakeSearch(
  events: string[],
  topK: number,
  lambdaPenalty: number | null,
  signal?: AbortSignal,
): Promise<TrakeResult> {
  return request<TrakeResult>('/api/trake', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ events, top_k: topK, lambda_penalty: lambdaPenalty }),
    signal,
  })
}

// GET /health (app/api/health.py) -- polled by useHealth (plan §P1.8) to
// detect Elasticsearch being down and show a degraded-mode banner.
export function getHealth(): Promise<HealthStatus> {
  return request<HealthStatus>('/health')
}

// GET /api/videos/{id} (app/api/videos.py)
export function getVideo(videoId: string): Promise<VideoInfo> {
  return request<VideoInfo>(`/api/videos/${encodeURIComponent(videoId)}`)
}

// GET /api/videos/{id}/keyframes (app/api/videos.py)
export function getVideoKeyframes(videoId: string): Promise<VideoKeyframesResponse> {
  return request<VideoKeyframesResponse>(
    `/api/videos/${encodeURIComponent(videoId)}/keyframes`,
  )
}

// Backs "Find Similar Images": fetch a keyframe's own full-res image bytes
// (GET /api/keyframes/{id}/image) so they can be re-submitted to
// POST /api/search/image -- goes through the exact same encoder path as any
// other image upload (CLAUDE.md §3: one encoder, one preprocessing).
export async function fetchImageBlob(imageUrl: string): Promise<Blob> {
  const response = await fetch(imageUrl)
  if (!response.ok) {
    throw new Error(`Failed to fetch image at ${imageUrl}: HTTP ${response.status}`)
  }
  return response.blob()
}
