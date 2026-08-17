import { useCallback, useRef, useState } from 'react'
import { trakeSearch } from '../api/client'
import { TRAKE_DEFAULT_THRESHOLD } from '../api/constants'
import { isAbortError, messageFor } from '../api/requestHelpers'
import type { TrakeVideoResultItem } from '../api/types'

export interface UseTrakeResult {
  events: string[]
  setEvent: (index: number, value: string) => void
  addEvent: () => void
  removeEvent: (index: number) => void
  threshold: string
  setThreshold: (value: string) => void
  results: TrakeVideoResultItem[]
  isLoading: boolean
  error: string | null
  search: () => void
}

const INITIAL_EVENTS = ['', '']
const INITIAL_THRESHOLD = String(TRAKE_DEFAULT_THRESHOLD)

// DANTE/TRAKE (plan §P2.2, PRD §F6/US-4): the operator types N event
// descriptions IN TIME ORDER; POST /api/trake returns each matching
// video's keyframes in that same order (CLAUDE.md §3, τ < t) -- this hook
// never reorders `events` on submit or `results` on return, it only trims
// and drops blank rows so the operator doesn't have to manually delete an
// empty row before searching.
export function useTrake(topK: number): UseTrakeResult {
  const [events, setEvents] = useState<string[]>(INITIAL_EVENTS)
  const [threshold, setThreshold] = useState(INITIAL_THRESHOLD)
  const [results, setResults] = useState<TrakeVideoResultItem[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const abortRef = useRef<AbortController | null>(null)

  const setEvent = useCallback((index: number, value: string) => {
    setEvents((prev) => prev.map((event, i) => (i === index ? value : event)))
  }, [])

  const addEvent = useCallback(() => {
    setEvents((prev) => [...prev, ''])
  }, [])

  const removeEvent = useCallback((index: number) => {
    setEvents((prev) => (prev.length <= 1 ? prev : prev.filter((_, i) => i !== index)))
  }, [])

  const search = useCallback(() => {
    const trimmed = events.map((event) => event.trim()).filter((event) => event.length > 0)
    if (trimmed.length === 0) return
    const parsedThreshold = Number(threshold)
    if (threshold.trim() === '' || !Number.isFinite(parsedThreshold)) {
      setError('Threshold phải là số hợp lệ')
      return
    }
    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller
    setIsLoading(true)
    setError(null)

    trakeSearch(trimmed, topK, null, parsedThreshold, controller.signal)
      .then((response) => setResults(response.results))
      .catch((err: unknown) => {
        if (isAbortError(err)) return
        setError(messageFor(err, 'TRAKE search failed'))
      })
      .finally(() => {
        if (abortRef.current === controller) setIsLoading(false)
      })
  }, [events, threshold, topK])

  return {
    events,
    setEvent,
    addEvent,
    removeEvent,
    threshold,
    setThreshold,
    results,
    isLoading,
    error,
    search,
  }
}
