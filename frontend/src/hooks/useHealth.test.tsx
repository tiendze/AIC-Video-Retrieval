import { renderHook, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { getHealth } from '../api/client'
import { useHealth } from './useHealth'

// RED reproducer for plan §P1.8 (PRD T5): the frontend needs to know when
// Elasticsearch is down so it can show a degraded-mode banner and disable
// OCR row selection -- GET /health (app/api/health.py) already reports
// elasticsearch_ok, this hook just polls it.
vi.mock('../api/client', () => ({
  getHealth: vi.fn(),
}))

const OK_STATUS = {
  status: 'ok' as const,
  milvus_ok: true,
  elasticsearch_ok: true,
  mongodb_ok: true,
  asr_status: 'ready' as const,
  asr_ready: true,
  asr_segment_count: 873,
}
const DEGRADED_STATUS = {
  status: 'degraded' as const,
  milvus_ok: true,
  elasticsearch_ok: false,
  mongodb_ok: true,
  asr_status: 'unreachable' as const,
  asr_ready: false,
  asr_segment_count: null,
}

afterEach(() => {
  vi.clearAllMocks()
  vi.useRealTimers()
})

describe('useHealth', () => {
  it('returns null before the first health check resolves', () => {
    vi.mocked(getHealth).mockReturnValue(new Promise(() => {})) // never resolves

    const { result } = renderHook(() => useHealth())

    expect(result.current).toBeNull()
  })

  it('returns the health status once GET /health resolves', async () => {
    vi.mocked(getHealth).mockResolvedValue(OK_STATUS)

    const { result } = renderHook(() => useHealth())

    await waitFor(() => expect(result.current).toEqual(OK_STATUS))
  })

  it('treats a fetch failure as degraded rather than throwing', async () => {
    vi.mocked(getHealth).mockRejectedValue(new Error('network down'))

    const { result } = renderHook(() => useHealth())

    await waitFor(() => expect(result.current?.elasticsearch_ok).toBe(false))
    expect(result.current?.status).toBe('degraded')
    expect(result.current?.asr_status).toBe('unreachable')
    expect(result.current?.asr_ready).toBe(false)
    expect(result.current?.asr_segment_count).toBeNull()
  })

  it('polls again after the interval elapses', async () => {
    vi.useFakeTimers()
    vi.mocked(getHealth).mockResolvedValue(DEGRADED_STATUS)

    renderHook(() => useHealth())
    await vi.waitFor(() => expect(getHealth).toHaveBeenCalledTimes(1))

    await vi.advanceTimersByTimeAsync(30_000)

    expect(getHealth).toHaveBeenCalledTimes(2)
  })
})
