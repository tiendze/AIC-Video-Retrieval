import type { QueryRewriteResult } from '../api/types'
import styles from './QueryRewritePanel.module.css'

export interface QueryRewritePanelProps {
  result: QueryRewriteResult
  onSearch: (text: string) => void
  onDismiss: () => void
}

// QUEST branch 1 UI (plan §P2.5, PRD §F5 acceptance criteria: "Luôn hiển
// thị truy vấn gốc VÀ truy vấn đã viết lại"). Never searches on its own --
// per chủ dự án's decision (2026-08-15), the operator reads both and picks
// which to search with (CLAUDE.md §3: interactive system).
export function QueryRewritePanel({ result, onSearch, onDismiss }: QueryRewritePanelProps) {
  return (
    <div className={styles.panel} role="region" aria-label="AI query rewrite">
      <button type="button" className={styles.close} onClick={onDismiss} aria-label="Close">
        ✕
      </button>
      <div className={styles.row}>
        <span className={styles.label}>Gốc:</span>
        <span className={styles.text}>{result.original_query}</span>
        <button
          type="button"
          className={styles.useButton}
          onClick={() => onSearch(result.original_query)}
        >
          Dùng câu gốc
        </button>
      </div>
      {result.rewritten_query ? (
        <div className={styles.row}>
          <span className={styles.label}>AI viết lại:</span>
          <span className={styles.text}>{result.rewritten_query}</span>
          <button
            type="button"
            className={styles.useButton}
            onClick={() => onSearch(result.rewritten_query as string)}
          >
            Dùng câu này
          </button>
        </div>
      ) : (
        <div className={styles.fallback}>
          Không thể viết lại truy vấn lúc này — vẫn dùng được câu gốc.
        </div>
      )}
    </div>
  )
}
