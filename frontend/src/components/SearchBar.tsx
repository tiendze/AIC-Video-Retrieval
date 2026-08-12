import { forwardRef } from 'react'
import styles from './SearchBar.module.css'

export interface SearchBarProps {
  query: string
  onQueryChange: (query: string) => void
  onSubmit: () => void
}

// PRD §F9 (A): "ô nhập + nút AI enhancer". The enhancer is QUEST's query
// rewrite (P2 backend, plan §P1.5 decision: shown disabled with a badge
// rather than omitted, so the layout doesn't get rearranged when P2 wires
// the real handler in).
export const SearchBar = forwardRef<HTMLInputElement, SearchBarProps>(function SearchBar(
  { query, onQueryChange, onSubmit },
  ref,
) {
  return (
    <form
      className={styles.form}
      onSubmit={(event) => {
        event.preventDefault()
        onSubmit()
      }}
    >
      <input
        ref={ref}
        aria-label="query"
        className={styles.input}
        type="text"
        placeholder="Tìm theo mô tả cảnh (tiếng Việt)…"
        value={query}
        onChange={(event) => onQueryChange(event.target.value)}
      />
      <button type="submit" className={styles.submit}>
        Search
      </button>
      <button type="button" className={styles.enhancer} disabled title="AI enhancer -- sắp có ở P2">
        ✨ AI enhancer <span className={styles.badge}>P2</span>
      </button>
    </form>
  )
})
