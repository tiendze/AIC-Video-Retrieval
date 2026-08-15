import { useCallback, useRef, useState } from 'react'
import { ApiError, rewriteQuery as rewriteQueryApi } from '../api/client'
import type { QueryRewriteResult } from '../api/types'

export interface UseQueryRewriteResult {
  result: QueryRewriteResult | null
  isLoading: boolean
  error: string | null
  rewrite: (query: string) => void
  dismiss: () => void
}

function isAbortError(err: unknown): boolean {
  return err instanceof DOMException && err.name === 'AbortError'
}

function messageFor(err: unknown, fallback: string): string {
  if (err instanceof ApiError) return err.message
  return fallback
}

// QUEST branch 1 (plan §P2.5, PRD §F5). Deliberately never triggers a
// search itself -- per chủ dự án's decision (2026-08-15), the operator
// picks which of original_query/rewritten_query to search with, so this
// hook's only job is to fetch and hold that pair for a panel to render.
export function useQueryRewrite(): UseQueryRewriteResult {
  const [result, setResult] = useState<QueryRewriteResult | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const abortRef = useRef<AbortController | null>(null)

  const rewrite = useCallback((query: string) => {
    if (!query.trim()) return
    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller
    setIsLoading(true)
    setError(null)

    rewriteQueryApi(query, controller.signal)
      .then(setResult)
      .catch((err: unknown) => {
        if (isAbortError(err)) return
        setError(messageFor(err, 'Query rewrite failed'))
      })
      .finally(() => {
        if (abortRef.current === controller) setIsLoading(false)
      })
  }, [])

  const dismiss = useCallback(() => setResult(null), [])

  return { result, isLoading, error, rewrite, dismiss }
}
