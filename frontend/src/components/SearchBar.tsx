import { forwardRef } from 'react'
import { OcrQueryList, type OcrQueryRow } from './OcrQueryList'
import styles from './SearchBar.module.css'

export interface SearchBarOcrProps {
  queries: OcrQueryRow[]
  onChange: (id: string, value: string) => void
  onAdd: () => void
  onRemove: (id: string) => void
}

export interface SearchBarProps {
  query: string
  onQueryChange: (query: string) => void
  onSubmit: () => void
  // QUEST branch 1 (plan §P2.5, PRD §F5) -- calls POST
  // /api/quest/rewrite-query with the current query; wired in App.tsx via
  // useQueryRewrite. Live since 2026-08-15 (was a disabled P2 placeholder).
  onEnhance: (query: string) => void
  enhanceLoading?: boolean
  // Semantic mode's supplementary OCR conditions -- App.tsx only passes
  // this while activeMode === 'semantic'; omitted for OCR mode, which
  // keeps rendering exactly the single-input form it always has.
  ocr?: SearchBarOcrProps
}

// PRD §F9 (A): "ô nhập + nút AI enhancer".
export const SearchBar = forwardRef<HTMLInputElement, SearchBarProps>(function SearchBar(
  { query, onQueryChange, onSubmit, onEnhance, enhanceLoading = false, ocr },
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
      {ocr && <span className={styles.label}>Semantic Query</span>}
      <div className={styles.queryRow}>
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
        <button
          type="button"
          className={styles.enhancer}
          disabled={!query.trim() || enhanceLoading}
          title="Viết lại truy vấn bằng AI (QUEST nhánh 1)"
          onClick={() => onEnhance(query)}
        >
          ✨ AI enhancer{enhanceLoading ? '…' : ''}
        </button>
      </div>
      {ocr && (
        <OcrQueryList
          queries={ocr.queries}
          onChange={ocr.onChange}
          onAdd={ocr.onAdd}
          onRemove={ocr.onRemove}
        />
      )}
    </form>
  )
})
