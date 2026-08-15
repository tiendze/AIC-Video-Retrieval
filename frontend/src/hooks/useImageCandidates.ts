import { useCallback, useRef, useState } from 'react'
import { findImageCandidates, searchByCandidate } from '../api/client'
import { isAbortError, messageFor } from '../api/requestHelpers'
import type { ImageCandidatesResult, SearchHit } from '../api/types'

export interface UseImageCandidatesResult {
  result: ImageCandidatesResult | null
  isLoading: boolean
  error: string | null
  findCandidates: (query: string) => void
  pickCandidate: (candidateId: string) => void
  dismiss: () => void
}

// QUEST branch 2 (plan §P2.5, PRD §F5): web sample-image search for an
// Out-of-Knowledge entity. `result` only ever holds a *pickable* candidate
// list (status="candidates") -- a backend fallback routes straight to
// `onHits` since there is nothing to pick (PRD §F5: never auto-pick, so a
// pickable list must always come from an explicit operator choice, never
// synthesized here).
export function useImageCandidates(
  topK: number,
  onHits: (hits: SearchHit[]) => void,
): UseImageCandidatesResult {
  const [result, setResult] = useState<ImageCandidatesResult | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const abortRef = useRef<AbortController | null>(null)

  const findCandidates = useCallback(
    (query: string) => {
      if (!query.trim()) return
      abortRef.current?.abort()
      const controller = new AbortController()
      abortRef.current = controller
      setIsLoading(true)
      setError(null)

      findImageCandidates(query, topK, controller.signal)
        .then((response) => {
          if (response.status === 'candidates') {
            setResult(response)
          } else {
            onHits(response.fallback_hits)
          }
        })
        .catch((err: unknown) => {
          if (isAbortError(err)) return
          setError(messageFor(err, 'Web image search failed'))
        })
        .finally(() => {
          if (abortRef.current === controller) setIsLoading(false)
        })
    },
    [topK, onHits],
  )

  const pickCandidate = useCallback(
    (candidateId: string) => {
      if (!result || result.status !== 'candidates' || !result.candidate_set_id) return
      const candidateSetId = result.candidate_set_id
      setIsLoading(true)
      setError(null)

      searchByCandidate(candidateSetId, candidateId, topK)
        .then((response) => {
          onHits(response.hits)
          setResult(null)
        })
        .catch((err: unknown) => {
          setError(messageFor(err, 'Image search failed'))
        })
        .finally(() => setIsLoading(false))
    },
    [result, topK, onHits],
  )

  const dismiss = useCallback(() => setResult(null), [])

  return { result, isLoading, error, findCandidates, pickCandidate, dismiss }
}
