import { useEffect, useRef } from 'react'
import { MAX_OCR_QUERIES } from '../api/constants'
import styles from './OcrQueryList.module.css'

export interface OcrQueryRow {
  id: string
  text: string
}

export interface OcrQueryListProps {
  queries: OcrQueryRow[]
  onChange: (id: string, value: string) => void
  onAdd: () => void
  onRemove: (id: string) => void
}

// Semantic mode's supplementary OCR conditions (plan: 1 primary Semantic
// Query + N optional OCR rows). Rendered as plain <div> rows inside
// SearchBar's own <form>, not a nested form -- nesting <form> elements is
// invalid HTML and would break the existing Enter-to-submit behavior.
// Rows are keyed by a stable `id` (not array index) so removing a middle
// row never re-associates the DOM node (and its focus/caret) of a
// surviving row with the wrong query.
export function OcrQueryList({ queries, onChange, onAdd, onRemove }: OcrQueryListProps) {
  const inputRefs = useRef(new Map<string, HTMLInputElement>())
  const previousIds = useRef<string[]>(queries.map((q) => q.id))

  useEffect(() => {
    const prevIds = previousIds.current
    const addedId = queries.map((q) => q.id).find((id) => !prevIds.includes(id))
    if (addedId) {
      inputRefs.current.get(addedId)?.focus()
    }
    previousIds.current = queries.map((q) => q.id)
  }, [queries])

  const canAdd = queries.length < MAX_OCR_QUERIES

  return (
    <div className={styles.container}>
      <span className={styles.label}>OCR Queries · optional</span>
      {queries.map((row, index) => (
        <div className={styles.row} key={row.id}>
          <input
            ref={(el) => {
              if (el) inputRefs.current.set(row.id, el)
              else inputRefs.current.delete(row.id)
            }}
            aria-label={`ocr query ${index + 1}`}
            className={styles.input}
            type="text"
            placeholder="Nhập nội dung OCR…"
            value={row.text}
            onChange={(event) => onChange(row.id, event.target.value)}
          />
          {queries.length > 1 && (
            <button
              type="button"
              className={styles.remove}
              aria-label={`xoá ocr query ${index + 1}`}
              title="Xoá OCR query này"
              onClick={() => onRemove(row.id)}
            >
              ×
            </button>
          )}
        </div>
      ))}
      <button type="button" className={styles.addButton} disabled={!canAdd} onClick={onAdd}>
        + Add OCR query
      </button>
    </div>
  )
}
