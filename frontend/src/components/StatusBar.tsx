import styles from './StatusBar.module.css'

export interface StatusBarProps {
  resultCount: number
  latencyMs: number | null
  isLoading: boolean
  error: string | null
  // plan §P1.8, PRD T5: set when GET /health reports elasticsearch_ok=false
  // -- null means "not degraded" (or not checked yet), never an empty
  // string, so App.tsx has one unambiguous way to say "nothing to show".
  degradedMessage: string | null
}

// Latency + result count, plus the degraded-mode banner named as P1.8 scope
// back in plan §P1.6 ("Ngoài phạm vi"). The banner is independent of the
// per-search loading/error state below it -- it reflects backend health,
// not the last search.
export function StatusBar({ resultCount, latencyMs, isLoading, error, degradedMessage }: StatusBarProps) {
  return (
    <div className={styles.bar}>
      {degradedMessage && <div className={styles.degraded}>{degradedMessage}</div>}
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
