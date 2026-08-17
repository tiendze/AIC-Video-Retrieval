import { useCallback, useRef, useState } from 'react'
import { DEFAULT_TOP_K } from '../api/constants'
import { search as searchApi, searchByImage as searchByImageApi } from '../api/client'
import { isAbortError, messageFor } from '../api/requestHelpers'
import type { SearchHit, SearchMode } from '../api/types'

export interface OcrQueryRow {
  id: string
  text: string
}

export interface UseSearchResult {
  query: string
  setQuery: (query: string) => void
  mode: SearchMode
  setMode: (mode: SearchMode) => void
  // Semantic mode's supplementary OCR conditions (frontend UI + state only
  // -- POST /api/search does not accept ocr_queries yet, so runSearch below
  // does not read this). Rows are keyed by a stable `id`, not array index,
  // because OcrQueryList needs a React key that survives removing a row
  // from the middle of the list without reassigning a surviving row's
  // focus/caret to the wrong input.
  ocrQueries: OcrQueryRow[]
  setOcrQuery: (id: string, text: string) => void
  addOcrQuery: () => void
  removeOcrQuery: (id: string) => void
  // Was hard-coded to DEFAULT_TOP_K with no UI control -- results always
  // capped at 20 regardless of how many the backend could return (up to
  // max_top_k=100). Now user-adjustable via TopKControl.
  topK: number
  setTopK: (topK: number) => void
  results: SearchHit[]
  // QUEST (plan §P2.5) pushes hits into this same grid via its own
  // hooks/panels -- rather than duplicating ResultGrid rendering for a
  // "QUEST results" case, useQueryRewrite/useImageCandidates just call this
  // setter directly once the operator picks q_r or a candidate image.
  setResults: (hits: SearchHit[]) => void
  isLoading: boolean
  error: string | null
  // Measured client-side (request start -> response parsed) -- prep for
  // P1.7's p50/p95 latency table (plan §P1.6: "useSearch.ts đã đo latency
  // sẵn để P1.7 chỉ việc thu thập"). QUEST-populated results reset this to
  // null (via setLatencyMs) -- P1.7's measurement is specific to
  // POST /api/search, so a stale number here would misrepresent a result
  // set that came from a different endpoint entirely.
  latencyMs: number | null
  setLatencyMs: (latencyMs: number | null) => void
  // `overrideQuery` lets a caller search by exact text without first
  // waiting for `setQuery` to re-render -- needed by QueryRewritePanel's
  // "use q_r" action, which must search by the just-fetched rewritten text
  // in the same click, not whatever `query` still held from the render
  // before `setQuery` was called.
  runSearch: (overrideQuery?: string) => void
  runImageSearch: (file: Blob) => void
}

// `query` and `mode` are independent state -- switching mode can never
// touch `query` (plan §P1.6 invariant: "Đổi mode không mất truy vấn đang
// gõ"). AbortController cancels an in-flight request when a newer one
// starts, so a slow first response can't overwrite a later one's results.
let ocrRowIdCounter = 0
function nextOcrRowId(): string {
  ocrRowIdCounter += 1
  return `ocr-${ocrRowIdCounter}`
}

export function useSearch(): UseSearchResult {
  const [query, setQuery] = useState('')
  const [mode, setMode] = useState<SearchMode>('semantic')
  const [topK, setTopK] = useState(DEFAULT_TOP_K)
  const [results, setResults] = useState<SearchHit[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [latencyMs, setLatencyMs] = useState<number | null>(null)
  const [ocrQueries, setOcrQueries] = useState<OcrQueryRow[]>(() => [
    { id: nextOcrRowId(), text: '' },
  ])
  const abortRef = useRef<AbortController | null>(null)

  const setOcrQuery = useCallback((id: string, text: string) => {
    setOcrQueries((prev) => prev.map((row) => (row.id === id ? { ...row, text } : row)))
  }, [])

  const addOcrQuery = useCallback(() => {
    setOcrQueries((prev) => [...prev, { id: nextOcrRowId(), text: '' }])
  }, [])

  const removeOcrQuery = useCallback((id: string) => {
    setOcrQueries((prev) => (prev.length <= 1 ? prev : prev.filter((row) => row.id !== id)))
  }, [])

  const runSearch = useCallback((overrideQuery?: string) => {
    const q = overrideQuery ?? query
    if (!q.trim()) return
    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller
    setIsLoading(true)
    setError(null)
    const start = performance.now()

    searchApi(q, mode, topK, controller.signal)
      .then((response) => {
        setResults(response.results)
        setLatencyMs(performance.now() - start)
      })
      .catch((err: unknown) => {
        if (isAbortError(err)) return
        setError(messageFor(err, 'Search failed'))
      })
      .finally(() => {
        if (abortRef.current === controller) setIsLoading(false)
      })
  }, [query, mode, topK])

  const runImageSearch = useCallback((file: Blob) => {
    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller
    setIsLoading(true)
    setError(null)
    const start = performance.now()

    searchByImageApi(file, topK, controller.signal)
      .then((response) => {
        setResults(response.results)
        setLatencyMs(performance.now() - start)
      })
      .catch((err: unknown) => {
        if (isAbortError(err)) return
        setError(messageFor(err, 'Image search failed'))
      })
      .finally(() => {
        if (abortRef.current === controller) setIsLoading(false)
      })
  }, [topK])

  return {
    query,
    setQuery,
    mode,
    setMode,
    ocrQueries,
    setOcrQuery,
    addOcrQuery,
    removeOcrQuery,
    topK,
    setTopK,
    results,
    setResults,
    isLoading,
    error,
    latencyMs,
    setLatencyMs,
    runSearch,
    runImageSearch,
  }
}
