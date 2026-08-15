import { act, renderHook, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { ApiError, rewriteQuery } from '../api/client'
import { useQueryRewrite } from './useQueryRewrite'

// RED reproducer for QUEST branch 1 (plan §P2.5, PRD §F5): the hook wraps
// POST /api/quest/rewrite-query. Per chủ dự án's decision (2026-08-15) it
// never searches by itself -- it only exposes both original_query and
// rewritten_query for a panel to display, and the operator decides which to
// actually search with. A backend `status: "fallback"` (Gemini
// error/timeout) is a normal 200 response, not a hook-level error --
// original_query is always usable regardless.
vi.mock('../api/client', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../api/client')>()
  return { ...actual, rewriteQuery: vi.fn() }
})

afterEach(() => {
  vi.clearAllMocks()
})

describe('useQueryRewrite', () => {
  it('starts with no result and not loading', () => {
    const { result } = renderHook(() => useQueryRewrite())

    expect(result.current.result).toBeNull()
    expect(result.current.isLoading).toBe(false)
    expect(result.current.error).toBeNull()
  })

  it('does not call the API for a blank query', () => {
    const { result } = renderHook(() => useQueryRewrite())

    act(() => result.current.rewrite('   '))

    expect(rewriteQuery).not.toHaveBeenCalled()
  })

  it('sets the rewritten result on success', async () => {
    vi.mocked(rewriteQuery).mockResolvedValue({
      status: 'rewritten',
      original_query: 'Labubu',
      rewritten_query: 'một con búp bê tai thỏ màu hồng',
    })
    const { result } = renderHook(() => useQueryRewrite())

    act(() => result.current.rewrite('Labubu'))

    expect(result.current.isLoading).toBe(true)
    await waitFor(() => expect(result.current.isLoading).toBe(false))
    expect(result.current.result).toEqual({
      status: 'rewritten',
      original_query: 'Labubu',
      rewritten_query: 'một con búp bê tai thỏ màu hồng',
    })
    expect(result.current.error).toBeNull()
    expect(rewriteQuery).toHaveBeenCalledWith('Labubu', expect.anything())
  })

  it('surfaces a backend fallback status as a normal result, not an error', async () => {
    vi.mocked(rewriteQuery).mockResolvedValue({
      status: 'fallback',
      original_query: 'Labubu',
      rewritten_query: null,
    })
    const { result } = renderHook(() => useQueryRewrite())

    act(() => result.current.rewrite('Labubu'))
    await waitFor(() => expect(result.current.isLoading).toBe(false))

    expect(result.current.result?.status).toBe('fallback')
    expect(result.current.result?.original_query).toBe('Labubu')
    expect(result.current.error).toBeNull()
  })

  it('sets an error message when the request itself fails', async () => {
    vi.mocked(rewriteQuery).mockRejectedValue(
      new ApiError(500, { detail: 'boom', error_code: 'internal_error', request_id: 'r1' }),
    )
    const { result } = renderHook(() => useQueryRewrite())

    act(() => result.current.rewrite('Labubu'))
    await waitFor(() => expect(result.current.isLoading).toBe(false))

    expect(result.current.error).toBe('boom')
    expect(result.current.result).toBeNull()
  })

  it('clears the result via dismiss', async () => {
    vi.mocked(rewriteQuery).mockResolvedValue({
      status: 'rewritten',
      original_query: 'Labubu',
      rewritten_query: 'x',
    })
    const { result } = renderHook(() => useQueryRewrite())
    act(() => result.current.rewrite('Labubu'))
    await waitFor(() => expect(result.current.result).not.toBeNull())

    act(() => result.current.dismiss())

    expect(result.current.result).toBeNull()
  })
})
