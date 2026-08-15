import { act, renderHook, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { ApiError, findImageCandidates, searchByCandidate } from '../api/client'
import type { SearchHit } from '../api/types'
import { useImageCandidates } from './useImageCandidates'

// RED reproducer for QUEST branch 2 (plan §P2.5, PRD §F5): web sample-image
// search for an Out-of-Knowledge entity. The operator must manually pick a
// candidate -- never auto-picked (PRD §F5 acceptance criteria) -- so this
// hook exposes the candidate list for a picker component and only calls
// `onHits` once the operator actually clicks one (or the backend itself
// fell back to a plain search, which needs no picking at all).
vi.mock('../api/client', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../api/client')>()
  return { ...actual, findImageCandidates: vi.fn(), searchByCandidate: vi.fn() }
})

const HIT: SearchHit = {
  keyframe_id: 'kf1',
  video_id: 'video1',
  score: 0.9,
  frame_idx: 10,
  timestamp: 1.5,
  thumbnail_path: '/t.jpg',
  thumbnail_url: '/api/keyframes/kf1/thumbnail',
  image_url: '/api/keyframes/kf1/image',
}

afterEach(() => {
  vi.clearAllMocks()
})

describe('useImageCandidates', () => {
  it('starts with no result and not loading', () => {
    const onHits = vi.fn()
    const { result } = renderHook(() => useImageCandidates(20, onHits))

    expect(result.current.result).toBeNull()
    expect(result.current.isLoading).toBe(false)
    expect(result.current.error).toBeNull()
  })

  it('does not call the API for a blank query', () => {
    const onHits = vi.fn()
    const { result } = renderHook(() => useImageCandidates(20, onHits))

    act(() => result.current.findCandidates('   '))

    expect(findImageCandidates).not.toHaveBeenCalled()
  })

  it('stores the candidate list on success without calling onHits', async () => {
    const onHits = vi.fn()
    vi.mocked(findImageCandidates).mockResolvedValue({
      status: 'candidates',
      candidate_set_id: 'set-1',
      candidates: [{ candidate_id: 'c1', thumbnail_url: 'https://x/t1.jpg', title: 'Labubu' }],
      fallback_hits: [],
    })
    const { result } = renderHook(() => useImageCandidates(20, onHits))

    act(() => result.current.findCandidates('Labubu'))
    await waitFor(() => expect(result.current.isLoading).toBe(false))

    expect(result.current.result?.status).toBe('candidates')
    expect(result.current.result?.candidates).toHaveLength(1)
    expect(onHits).not.toHaveBeenCalled()
    expect(findImageCandidates).toHaveBeenCalledWith('Labubu', 20, expect.anything())
  })

  it('routes fallback_hits straight to onHits without a candidate list', async () => {
    const onHits = vi.fn()
    vi.mocked(findImageCandidates).mockResolvedValue({
      status: 'fallback',
      candidate_set_id: null,
      candidates: [],
      fallback_hits: [HIT],
    })
    const { result } = renderHook(() => useImageCandidates(20, onHits))

    act(() => result.current.findCandidates('Labubu'))
    await waitFor(() => expect(result.current.isLoading).toBe(false))

    expect(onHits).toHaveBeenCalledWith([HIT])
    expect(result.current.result).toBeNull()
  })

  it('sets an error when the candidate search request itself fails', async () => {
    const onHits = vi.fn()
    vi.mocked(findImageCandidates).mockRejectedValue(
      new ApiError(500, { detail: 'boom', error_code: 'internal_error', request_id: 'r1' }),
    )
    const { result } = renderHook(() => useImageCandidates(20, onHits))

    act(() => result.current.findCandidates('Labubu'))
    await waitFor(() => expect(result.current.isLoading).toBe(false))

    expect(result.current.error).toBe('boom')
  })

  it('does nothing when picking a candidate before any candidates were found', () => {
    const onHits = vi.fn()
    const { result } = renderHook(() => useImageCandidates(20, onHits))

    act(() => result.current.pickCandidate('c1'))

    expect(searchByCandidate).not.toHaveBeenCalled()
  })

  it('searches by the picked candidate and routes hits to onHits, then clears the list', async () => {
    const onHits = vi.fn()
    vi.mocked(findImageCandidates).mockResolvedValue({
      status: 'candidates',
      candidate_set_id: 'set-1',
      candidates: [{ candidate_id: 'c1', thumbnail_url: 'https://x/t1.jpg', title: 'Labubu' }],
      fallback_hits: [],
    })
    vi.mocked(searchByCandidate).mockResolvedValue({ status: 'ok', hits: [HIT] })
    const { result } = renderHook(() => useImageCandidates(20, onHits))
    act(() => result.current.findCandidates('Labubu'))
    await waitFor(() => expect(result.current.result).not.toBeNull())

    act(() => result.current.pickCandidate('c1'))
    await waitFor(() => expect(result.current.isLoading).toBe(false))

    expect(searchByCandidate).toHaveBeenCalledWith('set-1', 'c1', 20)
    expect(onHits).toHaveBeenCalledWith([HIT])
    expect(result.current.result).toBeNull()
  })

  it('sets an error when picking a candidate fails', async () => {
    const onHits = vi.fn()
    vi.mocked(findImageCandidates).mockResolvedValue({
      status: 'candidates',
      candidate_set_id: 'set-1',
      candidates: [{ candidate_id: 'c1', thumbnail_url: 'https://x/t1.jpg', title: 'Labubu' }],
      fallback_hits: [],
    })
    vi.mocked(searchByCandidate).mockRejectedValue(
      new ApiError(404, { detail: 'not found', error_code: 'candidate_not_found', request_id: 'r1' }),
    )
    const { result } = renderHook(() => useImageCandidates(20, onHits))
    act(() => result.current.findCandidates('Labubu'))
    await waitFor(() => expect(result.current.result).not.toBeNull())

    act(() => result.current.pickCandidate('c1'))
    await waitFor(() => expect(result.current.isLoading).toBe(false))

    expect(result.current.error).toBe('not found')
  })

  it('clears the candidate list via dismiss', async () => {
    const onHits = vi.fn()
    vi.mocked(findImageCandidates).mockResolvedValue({
      status: 'candidates',
      candidate_set_id: 'set-1',
      candidates: [{ candidate_id: 'c1', thumbnail_url: 'https://x/t1.jpg', title: 'Labubu' }],
      fallback_hits: [],
    })
    const { result } = renderHook(() => useImageCandidates(20, onHits))
    act(() => result.current.findCandidates('Labubu'))
    await waitFor(() => expect(result.current.result).not.toBeNull())

    act(() => result.current.dismiss())

    expect(result.current.result).toBeNull()
  })
})
