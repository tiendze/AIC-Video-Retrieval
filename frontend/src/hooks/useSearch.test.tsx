import { act, renderHook } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { useSearch } from './useSearch'

// RED reproducer for Semantic mode's supplementary OCR conditions (plan:
// "OCR Queries · optional" rows alongside the primary Semantic Query).
// `ocrQueries` lives in the same hook as `query`/`mode`/`topK` so the
// eventual backend wiring (POST /api/search ocr_queries param) is a change
// to this one hook's runSearch call, not a cross-component rewire.
// Rows are keyed by a stable `id` (not array index) -- mirrors
// useTrake.ts's `events` array shape but swaps index-addressing for
// id-addressing since OcrQueryList needs a stable React key.
describe('useSearch OCR queries', () => {
  it('starts with a single blank OCR row', () => {
    const { result } = renderHook(() => useSearch())

    expect(result.current.ocrQueries).toHaveLength(1)
    expect(result.current.ocrQueries[0]?.text).toBe('')
  })

  it('updates only the targeted row via setOcrQuery', () => {
    const { result } = renderHook(() => useSearch())
    const firstId = result.current.ocrQueries[0]!.id

    act(() => result.current.addOcrQuery())
    act(() => result.current.setOcrQuery(firstId, 'VINMART'))

    expect(result.current.ocrQueries.map((q) => q.text)).toEqual(['VINMART', ''])
  })

  it('appends a blank row with a new unique id via addOcrQuery', () => {
    const { result } = renderHook(() => useSearch())
    const firstId = result.current.ocrQueries[0]!.id

    act(() => result.current.addOcrQuery())

    expect(result.current.ocrQueries).toHaveLength(2)
    expect(result.current.ocrQueries[1]?.id).not.toBe(firstId)
    expect(result.current.ocrQueries[1]?.text).toBe('')
  })

  it('removes the row with the given id via removeOcrQuery', () => {
    const { result } = renderHook(() => useSearch())
    const firstId = result.current.ocrQueries[0]!.id
    act(() => result.current.addOcrQuery())
    act(() => result.current.setOcrQuery(firstId, 'a'))
    const secondId = result.current.ocrQueries[1]!.id
    act(() => result.current.setOcrQuery(secondId, 'b'))

    act(() => result.current.removeOcrQuery(firstId))

    expect(result.current.ocrQueries).toEqual([{ id: secondId, text: 'b' }])
  })

  it('never removes the last remaining row', () => {
    const { result } = renderHook(() => useSearch())
    const firstId = result.current.ocrQueries[0]!.id

    act(() => result.current.removeOcrQuery(firstId))

    expect(result.current.ocrQueries).toHaveLength(1)
  })
})
