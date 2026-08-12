import { useCallback, useRef, useState } from 'react'
import { DEFAULT_TOP_K } from '../api/constants'
import { ApiError, search as searchApi, searchByImage as searchByImageApi } from '../api/client'
import type { SearchHit, SearchMode } from '../api/types'

export interface UseSearchResult {
  query: string
  setQuery: (query: string) => void
  mode: SearchMode
  setMode: (mode: SearchMode) => void
  // Was hard-coded to DEFAULT_TOP_K with no UI control -- results always
  // capped at 20 regardless of how many the backend could return (up to
  // max_top_k=100). Now user-adjustable via TopKControl.
  topK: number
  setTopK: (topK: number) => void
  results: SearchHit[]
  isLoading: boolean
  error: string | null
  // Measured client-side (request start -> response parsed) -- prep for
  // P1.7's p50/p95 latency table (plan §P1.6: "useSearch.ts đã đo latency
  // sẵn để P1.7 chỉ việc thu thập").
  latencyMs: number | null
  runSearch: () => void
  runImageSearch: (file: Blob) => void
}

function isAbortError(err: unknown): boolean {
  return err instanceof DOMException && err.name === 'AbortError'
}

function messageFor(err: unknown, fallback: string): string {
  if (err instanceof ApiError) return err.message
  return fallback
}

// `query` and `mode` are independent state -- switching mode can never
// touch `query` (plan §P1.6 invariant: "Đổi mode không mất truy vấn đang
// gõ"). AbortController cancels an in-flight request when a newer one
// starts, so a slow first response can't overwrite a later one's results.
export function useSearch(): UseSearchResult {
  const [query, setQuery] = useState('')
  const [mode, setMode] = useState<SearchMode>('semantic')
  const [topK, setTopK] = useState(DEFAULT_TOP_K)
  const [results, setResults] = useState<SearchHit[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [latencyMs, setLatencyMs] = useState<number | null>(null)
  const abortRef = useRef<AbortController | null>(null)

  const runSearch = useCallback(() => {
    if (!query.trim()) return
    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller
    setIsLoading(true)
    setError(null)
    const start = performance.now()

    searchApi(query, mode, topK, controller.signal)
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
    topK,
    setTopK,
    results,
    isLoading,
    error,
    latencyMs,
    runSearch,
    runImageSearch,
  }
}
