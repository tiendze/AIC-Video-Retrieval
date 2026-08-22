import { act, renderHook, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { rerankKisCandidates, retrieveKisCandidates } from '../api/client'
import { useSearch } from './useSearch'

vi.mock('../api/client', () => ({
  retrieveKisCandidates: vi.fn().mockResolvedValue({
    candidate_ids: [],
    candidate_pool_size: 0,
    text_candidate_count: 0,
    ocr_candidate_count: 0,
    title_candidate_count: 0,
    rrf_k: 60,
    results: [],
  }),
  rerankKisCandidates: vi.fn().mockResolvedValue({
    candidate_pool_size: 0,
    final_top_k: 0,
    results: [],
  }),
  searchByImage: vi.fn().mockResolvedValue({ results: [] }),
}))

describe('useSearch staged KIS flow', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(retrieveKisCandidates).mockResolvedValue({
      candidate_ids: [],
      candidate_pool_size: 0,
      text_candidate_count: 0,
      ocr_candidate_count: 0,
      title_candidate_count: 0,
      rrf_k: 60,
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
    expect(result.current.textQueries).toEqual([''])
    expect(result.current.ocrQuery).toBe('')
    expect(result.current.titleQuery).toBe('')
    expect(result.current.textCandidateK).toBe(300)
    expect(result.current.ocrCandidateK).toBe(100)
    expect(result.current.titleCandidateK).toBe(100)
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

  it('sends Stage 1 Text Q0 and optional OCR with independent candidate counts', () => {
    const { result } = renderHook(() => useSearch())

    act(() => result.current.setGlobalTextQuery('  people on a boat  '))
    act(() => result.current.setTextCandidateK(300))
    act(() => result.current.setOcrQuery(' SAIGON '))
    act(() => result.current.setOcrCandidateK(100))
    act(() => result.current.retrieveCandidates())

    expect(retrieveKisCandidates).toHaveBeenCalledWith(
      {
        text_query: 'people on a boat',
        text_queries: ['people on a boat'],
        text_top_k: 300,
        ocr_query: 'SAIGON',
        ocr_top_k: 100,
        title_query: null,
        title_top_k: 100,
      },
      expect.any(AbortSignal),
    )
  })

  it('stores the full Stage 1 pool and reranks that original pool after subquery changes', async () => {
    vi.mocked(retrieveKisCandidates).mockResolvedValueOnce({
      candidate_ids: ['A', 'B', 'C'],
      candidate_pool_size: 3,
      text_candidate_count: 2,
      ocr_candidate_count: 2,
      title_candidate_count: 0,
      rrf_k: 60,
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
      title_candidate_count: 0,
      rrf_k: 60,
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
      title_candidate_count: 0,
      rrf_k: 60,
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
