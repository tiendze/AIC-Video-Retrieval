import { useEffect, useState } from 'react'
import { getHealth } from '../api/client'
import type { HealthStatus } from '../api/types'

// plan §P1.8, PRD T5: polls GET /health so the UI can show a degraded-mode
// banner and disable OCR query-row selection when Elasticsearch is
// unreachable, without a person having to submit a search first to find out.
const POLL_INTERVAL_MS = 30_000

const UNREACHABLE_STATUS: HealthStatus = {
  status: 'degraded',
  milvus_ok: false,
  elasticsearch_ok: false,
  mongodb_ok: false,
  asr_status: 'unreachable',
  asr_ready: false,
  asr_segment_count: null,
}

// `null` = not checked yet -- callers should treat this as "unknown", not
// "degraded", so the banner doesn't flash on first render.
export function useHealth(): HealthStatus | null {
  const [health, setHealth] = useState<HealthStatus | null>(null)

  useEffect(() => {
    let cancelled = false

    function check(): void {
      getHealth()
        .then((result) => {
          if (!cancelled) setHealth(result)
        })
        .catch(() => {
          // GET /health itself unreachable (backend fully down, not just
          // ES) -- degraded is the honest answer, not a thrown error that
          // would crash the polling loop.
          if (!cancelled) setHealth(UNREACHABLE_STATUS)
        })
    }

    check()
    const intervalId = setInterval(check, POLL_INTERVAL_MS)
    return () => {
      cancelled = true
      clearInterval(intervalId)
    }
  }, [])

  return health
}
