import { useState } from 'react'
import { MAX_TOP_K, MIN_TOP_K } from '../api/constants'
import styles from './TopKControl.module.css'

export interface TopKControlProps {
  topK: number
  onChange: (topK: number) => void
}

function clamp(value: number): number {
  return Math.min(MAX_TOP_K, Math.max(MIN_TOP_K, value))
}

// A `value={topK}` bound straight to the parent's committed number would
// fight the user mid-type -- e.g. clamping to "1" the instant they type the
// leading "1" of "100". Keeps its own draft string instead, and only
// clamps + emits a new number on blur/Enter, matching backend max_top_k
// (configs/search/search.yaml, api/constants.ts).
export function TopKControl({ topK, onChange }: TopKControlProps) {
  const [draft, setDraft] = useState(String(topK))

  function commit(): void {
    const raw = Number(draft)
    const clamped = Number.isFinite(raw) && draft.trim() !== '' ? clamp(Math.trunc(raw)) : topK
    setDraft(String(clamped))
    if (clamped !== topK) onChange(clamped)
  }

  return (
    <label className={styles.label}>
      Top K
      <input
        aria-label="top k"
        type="number"
        min={MIN_TOP_K}
        max={MAX_TOP_K}
        className={styles.input}
        value={draft}
        onChange={(event) => setDraft(event.target.value)}
        onBlur={commit}
        onKeyDown={(event) => {
          if (event.key === 'Enter') commit()
        }}
      />
    </label>
  )
}
