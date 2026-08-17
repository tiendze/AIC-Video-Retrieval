import { act, renderHook, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { ApiError, trakeSearch } from '../api/client'
import type { TrakeVideoResultItem } from '../api/types'
import { useTrake } from './useTrake'

// RED reproducer for DANTE/TRAKE UI (plan §P2.2, PRD §F6/US-4): the
// operator types N event descriptions IN TIME ORDER, POST /api/trake
// returns per-video sequences whose keyframes come back in that same
// order -- this hook must never reorder them (CLAUDE.md §3, τ < t).
vi.mock('../api/client', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../api/client')>()
  return { ...actual, trakeSearch: vi.fn() }
})

const RESULT: TrakeVideoResultItem = {
  video_id: 'L21_V001',
  score: 0.8,
  keyframes: [
    {
      keyframe_id: 'L21_V001_f10',
      frame_idx: 10,
      timestamp: 1.0,
      thumbnail_url: '/api/keyframes/L21_V001_f10/thumbnail',
      image_url: '/api/keyframes/L21_V001_f10/image',
    },
    {
      keyframe_id: 'L21_V001_f50',
      frame_idx: 50,
      timestamp: 5.0,
      thumbnail_url: '/api/keyframes/L21_V001_f50/thumbnail',
      image_url: '/api/keyframes/L21_V001_f50/image',
    },
  ],
}

afterEach(() => {
  vi.clearAllMocks()
})

describe('useTrake', () => {
  it('starts with two blank event rows and no results', () => {
    const { result } = renderHook(() => useTrake(10))

    expect(result.current.events).toEqual(['', ''])
    expect(result.current.results).toEqual([])
    expect(result.current.isLoading).toBe(false)
    expect(result.current.error).toBeNull()
    expect(result.current.threshold).toBe('0')
  })

  it('updates only the targeted event row', () => {
    const { result } = renderHook(() => useTrake(10))

    act(() => result.current.setEvent(0, 'người đàn ông mở cửa'))

    expect(result.current.events).toEqual(['người đàn ông mở cửa', ''])
  })

  it('appends a blank row via addEvent', () => {
    const { result } = renderHook(() => useTrake(10))

    act(() => result.current.addEvent())

    expect(result.current.events).toEqual(['', '', ''])
  })

  it('removes the row at the given index via removeEvent', () => {
    const { result } = renderHook(() => useTrake(10))
    act(() => result.current.setEvent(0, 'a'))
    act(() => result.current.setEvent(1, 'b'))
    act(() => result.current.addEvent())
    act(() => result.current.setEvent(2, 'c'))

    act(() => result.current.removeEvent(1))

    expect(result.current.events).toEqual(['a', 'c'])
  })

  it('never removes the last remaining row', () => {
    const { result } = renderHook(() => useTrake(10))
    act(() => result.current.removeEvent(0))
    act(() => result.current.removeEvent(0))

    act(() => result.current.removeEvent(0))

    expect(result.current.events).toHaveLength(1)
  })

  it('does not call the API when every event is blank', () => {
    const { result } = renderHook(() => useTrake(10))

    act(() => result.current.search())

    expect(trakeSearch).not.toHaveBeenCalled()
  })

  it('trims and drops blank rows before submitting, preserving order', () => {
    vi.mocked(trakeSearch).mockReturnValue(new Promise(() => {}))
    const { result } = renderHook(() => useTrake(10))
    act(() => result.current.setEvent(0, '  người đàn ông mở cửa  '))
    act(() => result.current.addEvent())
    act(() => result.current.setEvent(2, 'xe máy chạy qua'))

    act(() => result.current.search())

    expect(trakeSearch).toHaveBeenCalledWith(
      ['người đàn ông mở cửa', 'xe máy chạy qua'],
      10,
      null,
      0,
      expect.anything(),
    )
  })

  it('passes the configured threshold through to the request', () => {
    vi.mocked(trakeSearch).mockReturnValue(new Promise(() => {}))
    const { result } = renderHook(() => useTrake(10))
    act(() => result.current.setEvent(0, 'a'))
    act(() => result.current.setEvent(1, 'b'))
    act(() => result.current.setThreshold('1.25'))

    act(() => result.current.search())

    expect(trakeSearch).toHaveBeenCalledWith(['a', 'b'], 10, null, 1.25, expect.anything())
  })

  it('shows an error instead of searching when threshold is not numeric', () => {
    const { result } = renderHook(() => useTrake(10))
    act(() => result.current.setEvent(0, 'a'))
    act(() => result.current.setThreshold('abc'))

    act(() => result.current.search())

    expect(trakeSearch).not.toHaveBeenCalled()
    expect(result.current.error).toBe('Threshold phải là số hợp lệ')
  })

  it('sets results on a successful search', async () => {
    vi.mocked(trakeSearch).mockResolvedValue({ results: [RESULT] })
    const { result } = renderHook(() => useTrake(10))
    act(() => result.current.setEvent(0, 'a'))
    act(() => result.current.setEvent(1, 'b'))

    act(() => result.current.search())
    await waitFor(() => expect(result.current.isLoading).toBe(false))

    expect(result.current.results).toEqual([RESULT])
    // Order preserved exactly as the backend returned it -- no client-side
    // re-sort of the keyframes array anywhere in this hook.
    expect(result.current.results[0]?.keyframes.map((k) => k.frame_idx)).toEqual([10, 50])
  })

  it('sets an error message when the request fails', async () => {
    vi.mocked(trakeSearch).mockRejectedValue(
      new ApiError(500, { detail: 'boom', error_code: 'internal_error', request_id: 'r1' }),
    )
    const { result } = renderHook(() => useTrake(10))
    act(() => result.current.setEvent(0, 'a'))
    act(() => result.current.setEvent(1, 'b'))

    act(() => result.current.search())
    await waitFor(() => expect(result.current.isLoading).toBe(false))

    expect(result.current.error).toBe('boom')
  })

  it('passes the configured top_k through to the request', () => {
    vi.mocked(trakeSearch).mockReturnValue(new Promise(() => {}))
    const { result } = renderHook(() => useTrake(42))
    act(() => result.current.setEvent(0, 'a'))
    act(() => result.current.setEvent(1, 'b'))

    act(() => result.current.search())

    expect(trakeSearch).toHaveBeenCalledWith(['a', 'b'], 42, null, 0, expect.anything())
  })
})
