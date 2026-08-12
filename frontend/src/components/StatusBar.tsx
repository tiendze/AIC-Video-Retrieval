import styles from './StatusBar.module.css'

export interface StatusBarProps {
  resultCount: number
  latencyMs: number | null
  isLoading: boolean
  error: string | null
}

// Latency + result count now; a degraded-mode banner (ES down -> semantic
// only) is P1.8 scope, not this one (plan §P1.6 "Ngoài phạm vi").
export function StatusBar({ resultCount, latencyMs, isLoading, error }: StatusBarProps) {
  return (
    <div className={styles.bar}>
      {isLoading && <span>Searching…</span>}
      {!isLoading && error && <span className={styles.error}>{error}</span>}
      {!isLoading && !error && (
        <span>
          {resultCount} result{resultCount === 1 ? '' : 's'}
          {latencyMs !== null && ` · ${Math.round(latencyMs)}ms`}
        </span>
      )}
    </div>
  )
}
