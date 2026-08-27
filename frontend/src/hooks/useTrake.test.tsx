import { act, renderHook, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { ApiError, trakeSearch } from '../api/client'
import type { TrakeVideoResultItem } from '../api/types'
import { EMPTY_TRAKE_EVENT, countFilledTrakeEvents, useTrake } from './useTrake'

// TRAKE UI (PRD §F6/US-4): the operator types N event rows IN TIME ORDER, each
// carrying its own Text / OCR / Title / ASR boxes. POST /api/trake returns
// sequences whose keyframes come back in that same order, already laid out in
// submission order -- this hook must never reorder either.
vi.mock('../api/client', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../api/client')>()
  return { ...actual, trakeSearch: vi.fn() }
})

const RESULT: TrakeVideoResultItem = {
  video_id: 'L21_V001',
  score: 0.8,
  min_score: 0.6,
  mean_score: 0.9,
  coverage: 1.0,
  video_score: 2.1,
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

describe('countFilledTrakeEvents', () => {
  it('counts a row with only an OCR box as a real event', () => {
    // An OCR-only or ASR-only row is legitimate: it contributes no semantic
    // axis but still gets one submitted frame like any other event.
    expect(
      countFilledTrakeEvents([
        { ...EMPTY_TRAKE_EVENT, text: 'a' },
        { ...EMPTY_TRAKE_EVENT, ocr: 'ty so 1-0' },
        { ...EMPTY_TRAKE_EVENT },
      ]),
    ).toBe(2)
  })
})

describe('useTrake', () => {
  it('starts with two blank event rows and no results', () => {
    const { result } = renderHook(() => useTrake(100))

    expect(result.current.events).toEqual([EMPTY_TRAKE_EVENT, EMPTY_TRAKE_EVENT])
    expect(result.current.results).toEqual([])
    expect(result.current.isLoading).toBe(false)
    expect(result.current.error).toBeNull()
    expect(result.current.latencyMs).toBeNull()
  })

  it('updates only the targeted box of the targeted row', () => {
    const { result } = renderHook(() => useTrake(100))

    act(() => result.current.setEventField(0, 'ocr', 'ty so 0-0'))

    expect(result.current.events[0]).toEqual({ ...EMPTY_TRAKE_EVENT, ocr: 'ty so 0-0' })
    expect(result.current.events[1]).toEqual(EMPTY_TRAKE_EVENT)
  })

  it('appends a blank row via addEvent', () => {
    const { result } = renderHook(() => useTrake(100))

    act(() => result.current.addEvent())

    expect(result.current.events).toHaveLength(3)
    expect(result.current.events[2]).toEqual(EMPTY_TRAKE_EVENT)
  })

  it('removes the row at the given index via removeEvent', () => {
    const { result } = renderHook(() => useTrake(100))
    act(() => result.current.setEventField(0, 'text', 'a'))
    act(() => result.current.setEventField(1, 'text', 'b'))
    act(() => result.current.addEvent())
    act(() => result.current.setEventField(2, 'text', 'c'))

    act(() => result.current.removeEvent(1))

    expect(result.current.events.map((event) => event.text)).toEqual(['a', 'c'])
  })

  it('never removes the last remaining row', () => {
    const { result } = renderHook(() => useTrake(100))
    act(() => result.current.removeEvent(0))
    act(() => result.current.removeEvent(0))

    act(() => result.current.removeEvent(0))

    expect(result.current.events).toHaveLength(1)
  })

  it('does not call the API when every box of every row is blank', () => {
    const { result } = renderHook(() => useTrake(100))

    act(() => result.current.search())

    expect(trakeSearch).not.toHaveBeenCalled()
  })

  it('sends each row as its own event object with all four channels', () => {
    vi.mocked(trakeSearch).mockReturnValue(new Promise(() => {}))
    const { result } = renderHook(() => useTrake(100))
    act(() => result.current.setEventField(0, 'text', '  cau thu chay len  '))
    act(() => result.current.setEventField(0, 'ocr', 'ty so 0-0'))
    act(() => result.current.setEventField(0, 'title', 'chung ket'))
    act(() => result.current.setEventField(0, 'asr', 'mo man tran dau'))
    act(() => result.current.setEventField(1, 'text', 'sut bong'))

    act(() => result.current.search())

    expect(trakeSearch).toHaveBeenCalledWith(
      {
        events: [
          {
            text_query: 'cau thu chay len',
            ocr_query: 'ty so 0-0',
            title_query: 'chung ket',
            asr_query: 'mo man tran dau',
          },
          { text_query: 'sut bong' },
        ],
        top_k: 100,
        threshold: -100,
      },
      expect.anything(),
    )
  })

  it('omits a blank box rather than sending an empty string', () => {
    // An empty OCR query would be an empty BM25 search, not "no OCR filter".
    vi.mocked(trakeSearch).mockReturnValue(new Promise(() => {}))
    const { result } = renderHook(() => useTrake(100))
    act(() => result.current.setEventField(0, 'text', 'a'))

    act(() => result.current.search())

    const payload = vi.mocked(trakeSearch).mock.calls[0]?.[0]
    expect(payload?.events[0]).toEqual({ text_query: 'a' })
  })

  it('drops fully blank rows before submitting, preserving order', () => {
    vi.mocked(trakeSearch).mockReturnValue(new Promise(() => {}))
    const { result } = renderHook(() => useTrake(100))
    act(() => result.current.setEventField(0, 'text', 'first'))
    act(() => result.current.addEvent())
    act(() => result.current.setEventField(2, 'text', 'third'))

    act(() => result.current.search())

    const payload = vi.mocked(trakeSearch).mock.calls[0]?.[0]
    expect(payload?.events).toEqual([{ text_query: 'first' }, { text_query: 'third' }])
  })

  it('includes a per-event text_top_k when the operator sets one', () => {
    vi.mocked(trakeSearch).mockReturnValue(new Promise(() => {}))
    const { result } = renderHook(() => useTrake(100))
    act(() => result.current.setEventField(0, 'text', 'a'))
    act(() => result.current.setEventField(0, 'textTopK', '250'))

    act(() => result.current.search())

    const payload = vi.mocked(trakeSearch).mock.calls[0]?.[0]
    expect(payload?.events[0]).toEqual({ text_query: 'a', text_top_k: 250 })
  })

  it('omits text_top_k when the operator leaves it blank, letting the backend default apply', () => {
    vi.mocked(trakeSearch).mockReturnValue(new Promise(() => {}))
    const { result } = renderHook(() => useTrake(100))
    act(() => result.current.setEventField(0, 'text', 'a'))

    act(() => result.current.search())

    const payload = vi.mocked(trakeSearch).mock.calls[0]?.[0]
    expect(payload?.events[0]).toEqual({ text_query: 'a' })
  })

  it('clamps an out-of-range text_top_k to the backend bound instead of rejecting it', () => {
    vi.mocked(trakeSearch).mockReturnValue(new Promise(() => {}))
    const { result } = renderHook(() => useTrake(100))
    act(() => result.current.setEventField(0, 'text', 'a'))
    act(() => result.current.setEventField(0, 'textTopK', '5000'))

    act(() => result.current.search())

    const payload = vi.mocked(trakeSearch).mock.calls[0]?.[0]
    expect(payload?.events[0]?.text_top_k).toBe(1000)
  })

  it('submits an OCR-only row as a real event', () => {
    vi.mocked(trakeSearch).mockReturnValue(new Promise(() => {}))
    const { result } = renderHook(() => useTrake(100))
    act(() => result.current.setEventField(0, 'text', 'a'))
    act(() => result.current.setEventField(1, 'ocr', 'ty so 1-0'))

    act(() => result.current.search())

    const payload = vi.mocked(trakeSearch).mock.calls[0]?.[0]
    expect(payload?.events).toEqual([{ text_query: 'a' }, { ocr_query: 'ty so 1-0' }])
  })

  it('passes the configured threshold through to the request', () => {
    vi.mocked(trakeSearch).mockReturnValue(new Promise(() => {}))
    const { result } = renderHook(() => useTrake(100))
    act(() => result.current.setEventField(0, 'text', 'a'))
    act(() => result.current.setThreshold('1.25'))

    act(() => result.current.search())

    expect(vi.mocked(trakeSearch).mock.calls[0]?.[0]?.threshold).toBe(1.25)
  })

  it('passes the configured top_k through to the request', () => {
    vi.mocked(trakeSearch).mockReturnValue(new Promise(() => {}))
    const { result } = renderHook(() => useTrake(42))
    act(() => result.current.setEventField(0, 'text', 'a'))

    act(() => result.current.search())

    expect(vi.mocked(trakeSearch).mock.calls[0]?.[0]?.top_k).toBe(42)
  })

  it('shows an error instead of searching when threshold is not numeric', () => {
    const { result } = renderHook(() => useTrake(100))
    act(() => result.current.setEventField(0, 'text', 'a'))
    act(() => result.current.setThreshold('abc'))

    act(() => result.current.search())

    expect(trakeSearch).not.toHaveBeenCalled()
    expect(result.current.error).toBe('Threshold phải là số hợp lệ')
  })

  it('sets results and pipeline diagnostics on a successful search', async () => {
    vi.mocked(trakeSearch).mockResolvedValue({
      results: [RESULT],
      events: [
        { event_index: 0, kis_candidate_count: 12, kis_video_count: 5, has_text: true },
        { event_index: 1, kis_candidate_count: 9, kis_video_count: 4, has_text: false },
      ],
      candidate_video_count: 7,
      chained_video_count: 3,
      dropped_video_count: 4,
    })
    const { result } = renderHook(() => useTrake(100))
    act(() => result.current.setEventField(0, 'text', 'a'))
    act(() => result.current.setEventField(1, 'ocr', 'b'))

    act(() => result.current.search())
    await waitFor(() => expect(result.current.isLoading).toBe(false))

    expect(result.current.results).toEqual([RESULT])
    expect(result.current.candidateVideoCount).toBe(7)
    expect(result.current.droppedVideoCount).toBe(4)
    expect(result.current.eventDiagnostics).toHaveLength(2)
    expect(result.current.latencyMs).toBeGreaterThanOrEqual(0)
    // Order preserved exactly as the backend returned it -- the response is
    // already in submission order, so any re-sort here would undo the
    // round-robin slot allocation.
    expect(result.current.results[0]?.keyframes.map((k) => k.frame_idx)).toEqual([10, 50])
  })

  it('tolerates a response without the optional diagnostics block', async () => {
    vi.mocked(trakeSearch).mockResolvedValue({ results: [RESULT] })
    const { result } = renderHook(() => useTrake(100))
    act(() => result.current.setEventField(0, 'text', 'a'))

    act(() => result.current.search())
    await waitFor(() => expect(result.current.isLoading).toBe(false))

    expect(result.current.eventDiagnostics).toEqual([])
    expect(result.current.candidateVideoCount).toBeNull()
  })

  it('sets an error message when the request fails', async () => {
    vi.mocked(trakeSearch).mockRejectedValue(
      new ApiError(500, { detail: 'boom', error_code: 'internal_error', request_id: 'r1' }),
    )
    const { result } = renderHook(() => useTrake(100))
    act(() => result.current.setEventField(0, 'text', 'a'))

    act(() => result.current.search())
    await waitFor(() => expect(result.current.isLoading).toBe(false))

    expect(result.current.error).toBe('boom')
  })
})
