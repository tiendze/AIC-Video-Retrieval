import { useCallback, useRef, useState } from 'react'
import {
  DEFAULT_ASR_VIDEO_CANDIDATE_K,
  DEFAULT_OCR_CANDIDATE_K,
  DEFAULT_TEXT_CANDIDATE_K,
  DEFAULT_TOP_K,
} from '../api/constants'
import {
  rerankKisCandidates as rerankKisCandidatesApi,
  retrieveKisCandidates as retrieveKisCandidatesApi,
  searchByImage as searchByImageApi,
} from '../api/client'
import { isAbortError, messageFor } from '../api/requestHelpers'
import type {
  AsrGate,
  AsrVideoEvidence,
  KisQueryPayload,
  KisQueryRow,
  KisQueryType,
  SearchHit,
} from '../api/types'

export interface KisCandidatePoolState {
  candidateIds: string[]
  candidatePoolSize: number
  textCandidateCount: number
  ocrCandidateCount: number
  asrGate: AsrGate | null
  hits: SearchHit[]
}

export interface UseSearchResult {
  globalTextQuery: string
  setGlobalTextQuery: (query: string) => void
  textCandidateK: number
  setTextCandidateK: (topK: number) => void
  ocrQuery: string
  setOcrQuery: (query: string) => void
  ocrCandidateK: number
  setOcrCandidateK: (topK: number) => void
  asrQuery: string
  setAsrQuery: (query: string) => void
  asrVideoCandidateK: number
  setAsrVideoCandidateK: (topK: number) => void
  candidatePool: KisCandidatePoolState | null
  stage2Queries: KisQueryRow[]
  setStage2QueryText: (id: string, text: string) => void
  setStage2QueryType: (id: string, type: KisQueryType) => void
  setAbePairAttribute: (queryId: string, pairId: string, value: string) => void
  setAbePairObject: (queryId: string, pairId: string, value: string) => void
  addAbePair: (queryId: string) => void
  removeAbePair: (queryId: string, pairId: string) => void
  addStage2Query: () => void
  removeStage2Query: (id: string) => void
  finalTopK: number
  setFinalTopK: (topK: number) => void
  primaryTextQuery: string
  // Compatibility helpers for QUEST/WebImageSearch: they operate on the
  // global Text Q0, while Stage 2 has its own independent subquery rows.
  query: string
  setQuery: (query: string) => void
  // Shared with non-KIS features still using a simple Top-K.
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
  retrieveCandidates: (overrideQuery?: string) => void
  rerankCandidates: () => void
  runImageSearch: (file: Blob) => void
}

// KIS Stage 2 rows are keyed by a stable id so removing a middle row never
// re-associates focus/caret with the wrong input. AbortController cancels
// an in-flight request when a newer one starts, so a slow first response
// can't overwrite a later one's results.
let kisRowIdCounter = 0
function nextKisRowId(): string {
  kisRowIdCounter += 1
  return `kis-${kisRowIdCounter}`
}

let abePairIdCounter = 0
function nextAbePairId(): string {
  abePairIdCounter += 1
  return `abe-pair-${abePairIdCounter}`
}

function emptyAbePair() {
  return { id: nextAbePairId(), attribute: '', object: '' }
}

function activePayload(rows: KisQueryRow[]): KisQueryPayload[] {
  return rows
    .map((row): KisQueryPayload | null => {
      if (row.type === 'abe') {
        const pairs = row.pairs
          .map((pair) => ({
            attribute: pair.attribute.trim(),
            object: pair.object.trim(),
          }))
          .filter((pair) => pair.attribute.length > 0 && pair.object.length > 0)
        return pairs.length > 0 ? { type: 'abe', pairs } : null
      }
      const text = row.text.trim()
      return text.length > 0 ? { type: row.type, text } : null
    })
    .filter((row): row is KisQueryPayload => row !== null)
}

function withAsrEvidence(
  hit: SearchHit,
  evidenceByVideo: Map<string, AsrVideoEvidence>,
): SearchHit {
  const evidence = evidenceByVideo.get(hit.video_id)
  return evidence ? { ...hit, asr_evidence: evidence } : hit
}

export function useSearch(): UseSearchResult {
  const [globalTextQueryValue, setGlobalTextQueryValue] = useState('')
  const [textCandidateKValue, setTextCandidateKValue] = useState(DEFAULT_TEXT_CANDIDATE_K)
  const [ocrQueryValue, setOcrQueryValue] = useState('')
  const [ocrCandidateKValue, setOcrCandidateKValue] = useState(DEFAULT_OCR_CANDIDATE_K)
  const [asrQueryValue, setAsrQueryValue] = useState('')
  const [asrVideoCandidateKValue, setAsrVideoCandidateKValue] = useState(
    DEFAULT_ASR_VIDEO_CANDIDATE_K,
  )
  const [candidatePool, setCandidatePool] = useState<KisCandidatePoolState | null>(null)
  const [stage2Queries, setStage2Queries] = useState<KisQueryRow[]>([])
  const [finalTopK, setFinalTopK] = useState(DEFAULT_TOP_K)
  const [results, setResults] = useState<SearchHit[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [latencyMs, setLatencyMs] = useState<number | null>(null)
  const abortRef = useRef<AbortController | null>(null)
  const requestGenerationRef = useRef(0)
  const primaryTextQuery = globalTextQueryValue

  const invalidateCandidatePool = useCallback(() => {
    requestGenerationRef.current += 1
    abortRef.current?.abort()
    abortRef.current = null
    setCandidatePool(null)
    setStage2Queries([])
    setResults([])
    setIsLoading(false)
    setError(null)
    setLatencyMs(null)
  }, [])

  const setGlobalTextQuery = useCallback((query: string) => {
    setGlobalTextQueryValue(query)
    invalidateCandidatePool()
  }, [invalidateCandidatePool])

  const setOcrQuery = useCallback((query: string) => {
    setOcrQueryValue(query)
    invalidateCandidatePool()
  }, [invalidateCandidatePool])

  const setTextCandidateK = useCallback((topK: number) => {
    setTextCandidateKValue(topK)
    invalidateCandidatePool()
  }, [invalidateCandidatePool])

  const setOcrCandidateK = useCallback((topK: number) => {
    setOcrCandidateKValue(topK)
    invalidateCandidatePool()
  }, [invalidateCandidatePool])

  const setAsrQuery = useCallback((query: string) => {
    setAsrQueryValue(query)
    invalidateCandidatePool()
  }, [invalidateCandidatePool])

  const setAsrVideoCandidateK = useCallback((topK: number) => {
    setAsrVideoCandidateKValue(topK)
    invalidateCandidatePool()
  }, [invalidateCandidatePool])

  const setStage2QueryText = useCallback((id: string, text: string) => {
    setStage2Queries((prev) => prev.map((row) => (row.id === id ? { ...row, text } : row)))
  }, [])

  const setStage2QueryType = useCallback((id: string, type: KisQueryType) => {
    setStage2Queries((prev) =>
      prev.map((row) => {
        if (row.id !== id) return row
        if (type === 'abe' && row.pairs.length === 0) {
          return { ...row, type, text: '', pairs: [emptyAbePair()] }
        }
        return { ...row, type }
      }),
    )
  }, [])

  const addStage2Query = useCallback(() => {
    setStage2Queries((prev) => [
      ...prev,
      { id: nextKisRowId(), type: 'text', text: '', pairs: [] },
    ])
  }, [])

  const removeStage2Query = useCallback((id: string) => {
    setStage2Queries((prev) => prev.filter((row) => row.id !== id))
  }, [])

  const setAbePairAttribute = useCallback((queryId: string, pairId: string, value: string) => {
    setStage2Queries((prev) =>
      prev.map((row) =>
        row.id === queryId
          ? {
              ...row,
              pairs: row.pairs.map((pair) =>
                pair.id === pairId ? { ...pair, attribute: value } : pair,
              ),
            }
          : row,
      ),
    )
  }, [])

  const setAbePairObject = useCallback((queryId: string, pairId: string, value: string) => {
    setStage2Queries((prev) =>
      prev.map((row) =>
        row.id === queryId
          ? {
              ...row,
              pairs: row.pairs.map((pair) =>
                pair.id === pairId ? { ...pair, object: value } : pair,
              ),
            }
          : row,
      ),
    )
  }, [])

  const addAbePair = useCallback((queryId: string) => {
    setStage2Queries((prev) =>
      prev.map((row) =>
        row.id === queryId ? { ...row, pairs: [...row.pairs, emptyAbePair()] } : row,
      ),
    )
  }, [])

  const removeAbePair = useCallback((queryId: string, pairId: string) => {
    setStage2Queries((prev) =>
      prev.map((row) => {
        if (row.id !== queryId) return row
        const remaining = row.pairs.filter((pair) => pair.id !== pairId)
        return { ...row, pairs: remaining.length > 0 ? remaining : [emptyAbePair()] }
      }),
    )
  }, [])

  const setQuery = useCallback((text: string) => {
    setGlobalTextQuery(text)
  }, [setGlobalTextQuery])

  const applyExternalResults = useCallback((hits: SearchHit[]) => {
    setCandidatePool(null)
    setResults(hits)
  }, [])

  const retrieveCandidates = useCallback((overrideQuery?: string) => {
    const textQuery = (overrideQuery ?? globalTextQueryValue).trim()
    const ocrQuery = ocrQueryValue.trim()
    const asrQuery = asrQueryValue.trim()
    if (!textQuery && !ocrQuery) return
    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller
    requestGenerationRef.current += 1
    const requestGeneration = requestGenerationRef.current
    setIsLoading(true)
    setError(null)
    const start = performance.now()

    retrieveKisCandidatesApi(
      {
        text_query: textQuery || null,
        text_top_k: textCandidateKValue,
        ocr_query: ocrQuery || null,
        ocr_top_k: ocrCandidateKValue,
        asr_query: asrQuery || null,
        asr_video_top_k: asrVideoCandidateKValue,
      },
      controller.signal,
    )
      .then((response) => {
        if (requestGeneration !== requestGenerationRef.current) return
        const asrGate = response.asr_gate ?? null
        const asrEvidenceByVideo = new Map(
          (asrGate?.videos ?? []).map((video) => [video.video_id, video]),
        )
        const hits = response.results.map((hit) =>
          withAsrEvidence(
            { ...hit, candidate_pool_size: response.candidate_pool_size },
            asrEvidenceByVideo,
          ),
        )
        const pool = {
          candidateIds: response.candidate_ids,
          candidatePoolSize: response.candidate_pool_size,
          textCandidateCount: response.text_candidate_count,
          ocrCandidateCount: response.ocr_candidate_count,
          asrGate,
          hits,
        }
        setCandidatePool(pool)
        setResults(hits)
        if (response.candidate_pool_size > 0 && finalTopK > response.candidate_pool_size) {
          setFinalTopK(response.candidate_pool_size)
        }
        setLatencyMs(performance.now() - start)
      })
      .catch((err: unknown) => {
        if (requestGeneration !== requestGenerationRef.current) return
        if (isAbortError(err)) return
        setError(messageFor(err, 'Candidate retrieval failed'))
      })
      .finally(() => {
        if (
          requestGeneration === requestGenerationRef.current &&
          abortRef.current === controller
        ) {
          setIsLoading(false)
        }
      })
  }, [
    asrQueryValue,
    asrVideoCandidateKValue,
    finalTopK,
    globalTextQueryValue,
    ocrCandidateKValue,
    ocrQueryValue,
    textCandidateKValue,
  ])

  const rerankCandidates = useCallback(() => {
    const subqueries = activePayload(stage2Queries)
    if (!candidatePool) return
    if (subqueries.length === 0) {
      setError('Add at least one Stage 2 constraint to calculate final reranking')
      return
    }
    if (finalTopK < 1 || finalTopK > candidatePool.candidatePoolSize) {
      setError('Final Top-K must be between 1 and the candidate pool size')
      return
    }

    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller
    requestGenerationRef.current += 1
    const requestGeneration = requestGenerationRef.current
    setIsLoading(true)
    setError(null)
    const start = performance.now()
    const stage1ById = new Map(candidatePool.hits.map((hit) => [hit.keyframe_id, hit]))
    const asrEvidenceByVideo = new Map(
      (candidatePool.asrGate?.videos ?? []).map((video) => [video.video_id, video]),
    )

    rerankKisCandidatesApi(
      {
        candidate_ids: candidatePool.candidateIds,
        subqueries,
        final_top_k: finalTopK,
      },
      controller.signal,
    )
      .then((response) => {
        if (requestGeneration !== requestGenerationRef.current) return
        setResults(
          response.results.map((hit) =>
            withAsrEvidence(
              {
                ...hit,
                stage1: stage1ById.get(hit.keyframe_id)?.stage1,
                candidate_pool_size: response.candidate_pool_size,
              },
              asrEvidenceByVideo,
            ),
          ),
        )
        setLatencyMs(performance.now() - start)
      })
      .catch((err: unknown) => {
        if (requestGeneration !== requestGenerationRef.current) return
        if (isAbortError(err)) return
        setError(messageFor(err, 'Candidate reranking failed'))
      })
      .finally(() => {
        if (
          requestGeneration === requestGenerationRef.current &&
          abortRef.current === controller
        ) {
          setIsLoading(false)
        }
      })
  }, [candidatePool, finalTopK, stage2Queries])

  const runImageSearch = useCallback((file: Blob) => {
    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller
    setIsLoading(true)
    setError(null)
    const start = performance.now()

    searchByImageApi(file, finalTopK, controller.signal)
      .then((response) => {
        setCandidatePool(null)
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
  }, [finalTopK])

  return {
    globalTextQuery: globalTextQueryValue,
    setGlobalTextQuery,
    textCandidateK: textCandidateKValue,
    setTextCandidateK,
    ocrQuery: ocrQueryValue,
    setOcrQuery,
    ocrCandidateK: ocrCandidateKValue,
    setOcrCandidateK,
    asrQuery: asrQueryValue,
    setAsrQuery,
    asrVideoCandidateK: asrVideoCandidateKValue,
    setAsrVideoCandidateK,
    candidatePool,
    stage2Queries,
    setStage2QueryText,
    setStage2QueryType,
    setAbePairAttribute,
    setAbePairObject,
    addAbePair,
    removeAbePair,
    addStage2Query,
    removeStage2Query,
    finalTopK,
    setFinalTopK,
    primaryTextQuery,
    query: primaryTextQuery,
    setQuery,
    topK: finalTopK,
    setTopK: setFinalTopK,
    results,
    setResults: applyExternalResults,
    isLoading,
    error,
    latencyMs,
    setLatencyMs,
    runSearch: retrieveCandidates,
    retrieveCandidates,
    rerankCandidates,
    runImageSearch,
  }
}
