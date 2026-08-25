import { useCallback, useRef, useState } from 'react'
import { trakeSearch } from '../api/client'
import {
  TRAKE_DEFAULT_THRESHOLD,
  TRAKE_MAX_EVENT_TEXT_TOP_K,
  TRAKE_MIN_EVENT_TEXT_TOP_K,
} from '../api/constants'
import { isAbortError, messageFor } from '../api/requestHelpers'
import type {
  TrakeEventDiagnostics,
  TrakeEventPayload,
  TrakeVideoResultItem,
} from '../api/types'

// One row of the operator's form. All four channels are per-event: OCR pins a
// moment (a scoreboard that only appears at the last event), and Title/ASR,
// although they gate at the video level, are allowed to differ per event so a
// commentary line tied to one moment does not constrain the others.
// `textTopK` is a string like `threshold` (editable, may be blank) -- blank
// means "use the backend default" (configs/search/trake.yaml event_text_top_k).
export interface TrakeEventDraft {
  text: string
  ocr: string
  title: string
  asr: string
  textTopK: string
}

export type TrakeEventField = keyof TrakeEventDraft

export interface UseTrakeResult {
  events: TrakeEventDraft[]
  setEventField: (index: number, field: TrakeEventField, value: string) => void
  addEvent: () => void
  removeEvent: (index: number) => void
  threshold: string
  setThreshold: (value: string) => void
  results: TrakeVideoResultItem[]
  sequencesByVideo: Record<string, TrakeVideoResultItem[]>
  eventDiagnostics: TrakeEventDiagnostics[]
  candidateVideoCount: number | null
  droppedVideoCount: number | null
  isLoading: boolean
  error: string | null
  latencyMs: number | null
  search: () => void
}

export const EMPTY_TRAKE_EVENT: TrakeEventDraft = {
  text: '',
  ocr: '',
  title: '',
  asr: '',
  textTopK: '',
}

const INITIAL_EVENTS: TrakeEventDraft[] = [{ ...EMPTY_TRAKE_EVENT }, { ...EMPTY_TRAKE_EVENT }]
const INITIAL_THRESHOLD = String(TRAKE_DEFAULT_THRESHOLD)

// An event row counts as filled when ANY of its four boxes has text -- an
// OCR-only or ASR-only row is a legitimate event, it just contributes no
// semantic axis (app/services/trake_service.py handles that case explicitly).
export function isTrakeEventFilled(event: TrakeEventDraft): boolean {
  return (
    event.text.trim().length > 0 ||
    event.ocr.trim().length > 0 ||
    event.title.trim().length > 0 ||
    event.asr.trim().length > 0
  )
}

export function countFilledTrakeEvents(events: TrakeEventDraft[]): number {
  return events.filter(isTrakeEventFilled).length
}

function toPayload(event: TrakeEventDraft): TrakeEventPayload {
  const payload: TrakeEventPayload = {}
  const text = event.text.trim()
  const ocr = event.ocr.trim()
  const title = event.title.trim()
  const asr = event.asr.trim()
  // Blank boxes are omitted rather than sent as "": the backend treats an
  // absent channel and an empty one differently (an empty OCR query would be
  // an empty BM25 search, not "no OCR filter").
  if (text) payload.text_query = text
  if (ocr) payload.ocr_query = ocr
  if (title) payload.title_query = title
  if (asr) payload.asr_query = asr
  // A blank/invalid Top K also omits the field -- the backend then falls back
  // to its own default (event_text_top_k), same "absent means default" rule
  // as the other channels above.
  const parsedTopK = Number(event.textTopK)
  if (event.textTopK.trim() !== '' && Number.isFinite(parsedTopK)) {
    payload.text_top_k = Math.min(
      TRAKE_MAX_EVENT_TEXT_TOP_K,
      Math.max(TRAKE_MIN_EVENT_TEXT_TOP_K, Math.trunc(parsedTopK)),
    )
  }
  return payload
}

// TRAKE (PRD §F6/US-4): the operator types N event rows IN TIME ORDER;
// POST /api/trake returns each sequence's keyframes in that same order. This
// hook never reorders `events` on submit or `results` on return -- the response
// already arrives in submission order (round-robin across videos), so
// re-sorting it here would undo the slot allocation. It only drops rows where
// every box is blank, so the operator does not have to delete an empty row
// before searching.
export function useTrake(topK: number): UseTrakeResult {
  const [events, setEvents] = useState<TrakeEventDraft[]>(INITIAL_EVENTS)
  const [threshold, setThreshold] = useState(INITIAL_THRESHOLD)
  const [results, setResults] = useState<TrakeVideoResultItem[]>([])
  const [sequencesByVideo, setSequencesByVideo] = useState<
    Record<string, TrakeVideoResultItem[]>
  >({})
  const [eventDiagnostics, setEventDiagnostics] = useState<TrakeEventDiagnostics[]>([])
  const [candidateVideoCount, setCandidateVideoCount] = useState<number | null>(null)
  const [droppedVideoCount, setDroppedVideoCount] = useState<number | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [latencyMs, setLatencyMs] = useState<number | null>(null)
  const abortRef = useRef<AbortController | null>(null)

  const setEventField = useCallback(
    (index: number, field: TrakeEventField, value: string) => {
      setEvents((prev) =>
        prev.map((event, i) => (i === index ? { ...event, [field]: value } : event)),
      )
    },
    [],
  )

  const addEvent = useCallback(() => {
    setEvents((prev) => [...prev, { ...EMPTY_TRAKE_EVENT }])
  }, [])

  const removeEvent = useCallback((index: number) => {
    setEvents((prev) => (prev.length <= 1 ? prev : prev.filter((_, i) => i !== index)))
  }, [])

  const search = useCallback(() => {
    const filled = events.filter(isTrakeEventFilled)
    if (filled.length === 0) return
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
    setLatencyMs(null)
    const start = performance.now()

    trakeSearch(
      { events: filled.map(toPayload), top_k: topK, threshold: parsedThreshold },
      controller.signal,
    )
      .then((response) => {
        setResults(response.results)
        setSequencesByVideo(response.sequences_by_video ?? {})
        setEventDiagnostics(response.events ?? [])
        setCandidateVideoCount(response.candidate_video_count ?? null)
        setDroppedVideoCount(response.dropped_video_count ?? null)
        setLatencyMs(performance.now() - start)
      })
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
    setEventField,
    addEvent,
    removeEvent,
    threshold,
    setThreshold,
    results,
    sequencesByVideo,
    eventDiagnostics,
    candidateVideoCount,
    droppedVideoCount,
    isLoading,
    error,
    latencyMs,
    search,
  }
}
