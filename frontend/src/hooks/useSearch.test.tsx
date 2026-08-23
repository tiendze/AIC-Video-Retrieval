import { act, renderHook, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { rerankKisCandidates, retrieveKisCandidates, searchByImage } from '../api/client'
import type {
  KisCandidateResponse,
  KisRerankResponse,
  SearchHit,
  SearchResponse,
} from '../api/types'
import { useSearch } from './useSearch'

vi.mock('../api/client', () => ({
  retrieveKisCandidates: vi.fn().mockResolvedValue({
    candidate_ids: [],
    candidate_pool_size: 0,
    text_candidate_count: 0,
    ocr_candidate_count: 0,
    results: [],
  }),
  rerankKisCandidates: vi.fn().mockResolvedValue({
    candidate_pool_size: 0,
    final_top_k: 0,
    results: [],
  }),
  searchByImage: vi.fn().mockResolvedValue({ results: [] }),
}))

const ASR_VIDEO = {
  video_id: 'V1',
  video_rank: 1,
  best_bm25_score: 8.73,
  segments: [
    {
      segment_id: 'V1_seg1',
      text: 'công trình thủy lợi',
      start_time: 151.2,
      end_time: 158.4,
      raw_bm25: 8.73,
    },
  ],
}

const ASR_GATE = {
  query: 'thủy lợi',
  segment_hit_count: 57,
  candidate_video_count: 1,
  videos: [ASR_VIDEO],
}

const STAGE1_HIT: SearchHit = {
  keyframe_id: 'A',
  video_id: 'V1',
  score: 0.9,
  frame_idx: 10,
  timestamp: 154.0,
  thumbnail_path: 'a.jpg',
  thumbnail_url: '/api/keyframes/A/thumbnail',
  image_url: '/api/keyframes/A/image',
  stage1: { source_modalities: ['text'] },
}

function deferred<T>() {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((resolvePromise) => {
    resolve = resolvePromise
  })
  return { promise, resolve }
}

describe('useSearch staged KIS flow', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(retrieveKisCandidates).mockResolvedValue({
      candidate_ids: [],
      candidate_pool_size: 0,
      text_candidate_count: 0,
      ocr_candidate_count: 0,
      results: [],
    })
    vi.mocked(rerankKisCandidates).mockResolvedValue({
      candidate_pool_size: 0,
      final_top_k: 0,
      results: [],
    })
  })

  it('starts with blank Stage 1 controls and no Stage 2 subqueries', () => {
    const { result } = renderHook(() => useSearch())

    expect(result.current.globalTextQuery).toBe('')
    expect(result.current.ocrQuery).toBe('')
    expect(result.current.textCandidateK).toBe(300)
    expect(result.current.ocrCandidateK).toBe(100)
    expect(result.current.stage2Queries).toEqual([])
    expect(result.current.candidatePool).toBeNull()
  })

  it('edits Stage 2 rows independently and allows removing all constraints', () => {
    const { result } = renderHook(() => useSearch())

    act(() => result.current.addStage2Query())
    const firstId = result.current.stage2Queries[0]!.id
    act(() => result.current.setStage2QueryText(firstId, 'night scene'))
    act(() => result.current.setStage2QueryType(firstId, 'ocr'))
    act(() => result.current.removeStage2Query(firstId))

    expect(result.current.stage2Queries).toEqual([])
  })

  it('clears Stage 2 subqueries when a Stage 1 input changes', () => {
    const { result } = renderHook(() => useSearch())

    act(() => result.current.addStage2Query())
    const firstId = result.current.stage2Queries[0]!.id
    act(() => result.current.setStage2QueryText(firstId, 'old constraint'))
    act(() => result.current.setGlobalTextQuery('new global q0'))

    expect(result.current.stage2Queries).toEqual([])
    expect(result.current.candidatePool).toBeNull()
  })

  it('sends Stage 1 Text Q0 and optional OCR with independent candidate counts', async () => {
    const { result } = renderHook(() => useSearch())

    act(() => result.current.setGlobalTextQuery('  people on a boat  '))
    act(() => result.current.setTextCandidateK(300))
    act(() => result.current.setOcrQuery(' SAIGON '))
    act(() => result.current.setOcrCandidateK(100))
    act(() => result.current.retrieveCandidates())

    expect(retrieveKisCandidates).toHaveBeenCalledWith(
      {
        text_query: 'people on a boat',
        text_top_k: 300,
        ocr_query: 'SAIGON',
        ocr_top_k: 100,
        asr_query: null,
        asr_video_top_k: 20,
      },
      expect.any(AbortSignal),
    )
    await waitFor(() => expect(result.current.isLoading).toBe(false))
  })

  it('sends trimmed ASR Stage 1 fields and retains video evidence across Stage 2', async () => {
    vi.mocked(retrieveKisCandidates).mockResolvedValueOnce({
      candidate_ids: ['A'],
      candidate_pool_size: 1,
      text_candidate_count: 1,
      ocr_candidate_count: 0,
      asr_gate: ASR_GATE,
      results: [STAGE1_HIT],
    })
    vi.mocked(rerankKisCandidates).mockResolvedValueOnce({
      candidate_pool_size: 1,
      final_top_k: 1,
      results: [{ ...STAGE1_HIT, final_score: 0.95, active_modalities: 1 }],
    })
    const { result } = renderHook(() => useSearch())

    act(() => {
      result.current.setGlobalTextQuery('dam')
      result.current.setAsrQuery('  thủy lợi  ')
      result.current.setAsrVideoCandidateK(20)
    })
    act(() => result.current.retrieveCandidates())

    await waitFor(() => expect(result.current.candidatePool?.asrGate).toEqual(ASR_GATE))
    expect(retrieveKisCandidates).toHaveBeenCalledWith(
      expect.objectContaining({ asr_query: 'thủy lợi', asr_video_top_k: 20 }),
      expect.any(AbortSignal),
    )
    expect(result.current.results[0]?.asr_evidence).toEqual(ASR_VIDEO)
    expect(result.current.results[0]?.frame_idx).toBe(10)

    act(() => result.current.addStage2Query())
    const queryId = result.current.stage2Queries[0]!.id
    act(() => result.current.setStage2QueryText(queryId, 'night'))
    act(() => result.current.setFinalTopK(1))
    act(() => result.current.rerankCandidates())

    await waitFor(() => expect(result.current.results[0]?.final_score).toBe(0.95))
    expect(result.current.results[0]?.asr_evidence).toEqual(ASR_VIDEO)
    expect(result.current.results[0]?.frame_idx).toBe(10)
  })

  it('ignores a deferred Stage 1 response after the ASR query changes', async () => {
    const pending = deferred<KisCandidateResponse>()
    vi.mocked(retrieveKisCandidates).mockReturnValueOnce(pending.promise)
    const { result } = renderHook(() => useSearch())

    act(() => result.current.setGlobalTextQuery('dam'))
    act(() => result.current.setAsrQuery('old topic'))
    act(() => result.current.retrieveCandidates())
    expect(result.current.isLoading).toBe(true)

    act(() => result.current.setAsrQuery('new topic'))
    expect(result.current.candidatePool).toBeNull()
    expect(result.current.results).toEqual([])

    await act(async () => {
      pending.resolve({
        candidate_ids: ['A'],
        candidate_pool_size: 1,
        text_candidate_count: 1,
        ocr_candidate_count: 0,
        asr_gate: ASR_GATE,
        results: [STAGE1_HIT],
      })
      await pending.promise
    })

    expect(result.current.candidatePool).toBeNull()
    expect(result.current.results).toEqual([])
  })

  it('ignores a deferred Stage 2 response after the ASR video count changes', async () => {
    vi.mocked(retrieveKisCandidates).mockResolvedValueOnce({
      candidate_ids: ['A'],
      candidate_pool_size: 1,
      text_candidate_count: 1,
      ocr_candidate_count: 0,
      asr_gate: ASR_GATE,
      results: [STAGE1_HIT],
    })
    const pending = deferred<KisRerankResponse>()
    vi.mocked(rerankKisCandidates).mockReturnValueOnce(pending.promise)
    const { result } = renderHook(() => useSearch())

    act(() => result.current.setGlobalTextQuery('dam'))
    act(() => result.current.setAsrQuery('thủy lợi'))
    act(() => result.current.retrieveCandidates())
    await waitFor(() => expect(result.current.candidatePool?.candidateIds).toEqual(['A']))

    act(() => result.current.addStage2Query())
    const queryId = result.current.stage2Queries[0]!.id
    act(() => result.current.setStage2QueryText(queryId, 'night'))
    act(() => result.current.setFinalTopK(1))
    act(() => result.current.rerankCandidates())
    expect(result.current.isLoading).toBe(true)

    act(() => result.current.setAsrVideoCandidateK(10))
    expect(result.current.candidatePool).toBeNull()
    expect(result.current.results).toEqual([])

    await act(async () => {
      pending.resolve({
        candidate_pool_size: 1,
        final_top_k: 1,
        results: [{ ...STAGE1_HIT, final_score: 0.99, active_modalities: 1 }],
      })
      await pending.promise
    })

    expect(result.current.candidatePool).toBeNull()
    expect(result.current.results).toEqual([])
  })

  it('keeps an unrelated deferred image search active when ASR controls change', async () => {
    const pending = deferred<SearchResponse>()
    vi.mocked(searchByImage).mockReturnValueOnce(pending.promise)
    const { result } = renderHook(() => useSearch())

    act(() => result.current.runImageSearch(new Blob(['image'], { type: 'image/jpeg' })))
    const imageSignal = vi.mocked(searchByImage).mock.calls[0]?.[2]
    expect(result.current.isLoading).toBe(true)

    act(() => result.current.setAsrQuery('thủy lợi'))
    act(() => result.current.setAsrVideoCandidateK(10))

    expect(imageSignal?.aborted).toBe(false)
    expect(result.current.isLoading).toBe(true)

    await act(async () => {
      pending.resolve({ results: [STAGE1_HIT] })
      await pending.promise
    })

    expect(result.current.results).toEqual([STAGE1_HIT])
    expect(result.current.isLoading).toBe(false)
  })

  it('changing ASR query or candidate video count invalidates the candidate pool', async () => {
    vi.mocked(retrieveKisCandidates)
      .mockResolvedValueOnce({
        candidate_ids: ['A'],
        candidate_pool_size: 1,
        text_candidate_count: 1,
        ocr_candidate_count: 0,
        asr_gate: ASR_GATE,
        results: [STAGE1_HIT],
      })
      .mockResolvedValueOnce({
        candidate_ids: ['A'],
        candidate_pool_size: 1,
        text_candidate_count: 1,
        ocr_candidate_count: 0,
        asr_gate: ASR_GATE,
        results: [STAGE1_HIT],
      })
    const { result } = renderHook(() => useSearch())

    act(() => result.current.setGlobalTextQuery('dam'))
    act(() => result.current.retrieveCandidates())
    await waitFor(() => expect(result.current.candidatePool).not.toBeNull())

    act(() => result.current.setAsrQuery('new topic'))
    expect(result.current.candidatePool).toBeNull()

    act(() => result.current.retrieveCandidates())
    await waitFor(() => expect(result.current.candidatePool).not.toBeNull())

    act(() => result.current.setAsrVideoCandidateK(10))
    expect(result.current.candidatePool).toBeNull()
  })

  it('keeps an active zero-hit ASR gate distinct from an inactive gate', async () => {
    const zeroHitGate = {
      query: 'không có kết quả',
      segment_hit_count: 0,
      candidate_video_count: 0,
      videos: [],
    }
    vi.mocked(retrieveKisCandidates).mockResolvedValueOnce({
      candidate_ids: [],
      candidate_pool_size: 0,
      text_candidate_count: 0,
      ocr_candidate_count: 0,
      asr_gate: zeroHitGate,
      results: [],
    })
    const { result } = renderHook(() => useSearch())

    act(() => {
      result.current.setGlobalTextQuery('dam')
      result.current.setAsrQuery('không có kết quả')
    })
    act(() => result.current.retrieveCandidates())

    await waitFor(() => expect(result.current.candidatePool?.asrGate).toEqual(zeroHitGate))
  })

  it('stores the full Stage 1 pool and reranks that original pool after subquery changes', async () => {
    vi.mocked(retrieveKisCandidates).mockResolvedValueOnce({
      candidate_ids: ['A', 'B', 'C'],
      candidate_pool_size: 3,
      text_candidate_count: 2,
      ocr_candidate_count: 2,
      results: [
        {
          keyframe_id: 'A',
          video_id: 'V1',
          score: 0.9,
          frame_idx: 10,
          timestamp: 0.4,
          thumbnail_path: 'a.jpg',
          thumbnail_url: '/api/keyframes/A/thumbnail',
          image_url: '/api/keyframes/A/image',
          stage1: { source_modalities: ['text'] },
        },
        {
          keyframe_id: 'B',
          video_id: 'V1',
          score: 0.8,
          frame_idx: 20,
          timestamp: 0.8,
          thumbnail_path: 'b.jpg',
          thumbnail_url: '/api/keyframes/B/thumbnail',
          image_url: '/api/keyframes/B/image',
          stage1: { source_modalities: ['text', 'ocr'] },
        },
        {
          keyframe_id: 'C',
          video_id: 'V1',
          score: 1,
          frame_idx: 30,
          timestamp: 1.2,
          thumbnail_path: 'c.jpg',
          thumbnail_url: '/api/keyframes/C/thumbnail',
          image_url: '/api/keyframes/C/image',
          stage1: { source_modalities: ['ocr'] },
        },
      ],
    })
    vi.mocked(rerankKisCandidates)
      .mockResolvedValueOnce({
        candidate_pool_size: 3,
        final_top_k: 1,
        results: [
          {
            keyframe_id: 'B',
            video_id: 'V1',
            score: 1,
            final_score: 1,
            active_modalities: 1,
            modalities: { text: { geometric_mean: 1, score: 1, queries: [] } },
            frame_idx: 20,
            timestamp: 0.8,
            thumbnail_path: 'b.jpg',
            thumbnail_url: '/api/keyframes/B/thumbnail',
            image_url: '/api/keyframes/B/image',
          },
        ],
      })
      .mockResolvedValueOnce({
        candidate_pool_size: 3,
        final_top_k: 1,
        results: [
          {
            keyframe_id: 'C',
            video_id: 'V1',
            score: 1,
            final_score: 1,
            active_modalities: 2,
            modalities: {
              text: { geometric_mean: 1, score: 1, queries: [] },
              ocr: { geometric_mean: 1, score: 1, queries: [] },
            },
            frame_idx: 30,
            timestamp: 1.2,
            thumbnail_path: 'c.jpg',
            thumbnail_url: '/api/keyframes/C/thumbnail',
            image_url: '/api/keyframes/C/image',
          },
        ],
      })
    const { result } = renderHook(() => useSearch())

    act(() => result.current.setGlobalTextQuery('boat'))
    act(() => result.current.retrieveCandidates())
    await waitFor(() =>
      expect(result.current.candidatePool?.candidateIds).toEqual(['A', 'B', 'C']),
    )

    act(() => result.current.addStage2Query())
    const firstId = result.current.stage2Queries[0]!.id
    act(() => result.current.setStage2QueryText(firstId, 'night'))
    act(() => result.current.setFinalTopK(1))
    act(() => result.current.rerankCandidates())
    await waitFor(() => expect(rerankKisCandidates).toHaveBeenCalledTimes(1))

    act(() => result.current.addStage2Query())
    const secondId = result.current.stage2Queries[1]!.id
    act(() => result.current.setStage2QueryType(secondId, 'ocr'))
    act(() => result.current.setStage2QueryText(secondId, 'SAIGON'))
    act(() => result.current.rerankCandidates())
    await waitFor(() => expect(rerankKisCandidates).toHaveBeenCalledTimes(2))

    expect(rerankKisCandidates).toHaveBeenNthCalledWith(
      1,
      {
        candidate_ids: ['A', 'B', 'C'],
        subqueries: [{ type: 'text', text: 'night' }],
        final_top_k: 1,
      },
      expect.any(AbortSignal),
    )
    expect(rerankKisCandidates).toHaveBeenNthCalledWith(
      2,
      {
        candidate_ids: ['A', 'B', 'C'],
        subqueries: [
          { type: 'text', text: 'night' },
          { type: 'ocr', text: 'SAIGON' },
        ],
        final_top_k: 1,
      },
      expect.any(AbortSignal),
    )
    await waitFor(() =>
      expect(result.current.results[0]?.stage1?.source_modalities).toEqual(['ocr']),
    )
  })

  it('does not call Stage 2 rerank when there are no subqueries', async () => {
    vi.mocked(retrieveKisCandidates).mockResolvedValueOnce({
      candidate_ids: ['A'],
      candidate_pool_size: 1,
      text_candidate_count: 1,
      ocr_candidate_count: 0,
      results: [
        {
          keyframe_id: 'A',
          video_id: 'V1',
          score: 0.9,
          frame_idx: 10,
          timestamp: 0.4,
          thumbnail_path: 'a.jpg',
          thumbnail_url: '/api/keyframes/A/thumbnail',
          image_url: '/api/keyframes/A/image',
          stage1: { source_modalities: ['text'] },
        },
      ],
    })
    const { result } = renderHook(() => useSearch())

    act(() => result.current.setGlobalTextQuery('boat'))
    act(() => result.current.retrieveCandidates())
    await waitFor(() => expect(result.current.candidatePool?.candidatePoolSize).toBe(1))
    act(() => result.current.rerankCandidates())

    expect(rerankKisCandidates).not.toHaveBeenCalled()
    expect(result.current.error).toBe(
      'Add at least one Stage 2 constraint to calculate final reranking',
    )
  })

  it('sends ABE pairs as Stage 2 payloads', async () => {
    vi.mocked(retrieveKisCandidates).mockResolvedValueOnce({
      candidate_ids: ['A', 'B'],
      candidate_pool_size: 2,
      text_candidate_count: 2,
      ocr_candidate_count: 0,
      results: [
        {
          keyframe_id: 'A',
          video_id: 'V1',
          score: 0.9,
          frame_idx: 10,
          timestamp: 0.4,
          thumbnail_path: 'a.jpg',
          thumbnail_url: '/api/keyframes/A/thumbnail',
          image_url: '/api/keyframes/A/image',
          stage1: { source_modalities: ['text'] },
        },
        {
          keyframe_id: 'B',
          video_id: 'V1',
          score: 0.8,
          frame_idx: 20,
          timestamp: 0.8,
          thumbnail_path: 'b.jpg',
          thumbnail_url: '/api/keyframes/B/thumbnail',
          image_url: '/api/keyframes/B/image',
          stage1: { source_modalities: ['text'] },
        },
      ],
    })
    const { result } = renderHook(() => useSearch())

    act(() => result.current.setGlobalTextQuery('packet and uniform'))
    act(() => result.current.retrieveCandidates())
    await waitFor(() => expect(result.current.candidatePool?.candidateIds).toEqual(['A', 'B']))

    act(() => result.current.addStage2Query())
    const queryId = result.current.stage2Queries[0]!.id
    act(() => result.current.setStage2QueryType(queryId, 'abe'))
    const pairId = result.current.stage2Queries[0]!.pairs[0]!.id
    act(() => result.current.setAbePairAttribute(queryId, pairId, ' red '))
    act(() => result.current.setAbePairObject(queryId, pairId, ' packet '))
    act(() => result.current.addAbePair(queryId))
    const secondPairId = result.current.stage2Queries[0]!.pairs[1]!.id
    act(() => result.current.setAbePairAttribute(queryId, secondPairId, ' white '))
    act(() => result.current.setAbePairObject(queryId, secondPairId, ' uniform '))
    act(() => result.current.setFinalTopK(1))
    act(() => result.current.rerankCandidates())

    await waitFor(() => expect(rerankKisCandidates).toHaveBeenCalledTimes(1))
    expect(rerankKisCandidates).toHaveBeenCalledWith(
      {
        candidate_ids: ['A', 'B'],
        subqueries: [
          {
            type: 'abe',
            pairs: [
              { attribute: 'red', object: 'packet' },
              { attribute: 'white', object: 'uniform' },
            ],
          },
        ],
        final_top_k: 1,
      },
      expect.any(AbortSignal),
    )
  })
})
