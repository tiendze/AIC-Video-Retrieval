import type { SearchMode } from '../api/types'
import styles from './ModeSelector.module.css'

export interface ModeSelectorProps {
  mode: SearchMode
  onChange: (mode: SearchMode) => void
}

// PRD §F9 mode list: Semantic / OCR / DANTE. Semantic+OCR are live (P1);
// DANTE needs P2's sequence-search backend, so it's shown per plan §P1.5's
// decision ("hiện nhưng disabled + badge") rather than hidden -- the
// layout is settled once, P2 only has to wire the handler in.
const MODES: { value: SearchMode; label: string; shortcutKey: string }[] = [
  { value: 'semantic', label: 'Semantic', shortcutKey: '1' },
  { value: 'ocr', label: 'OCR', shortcutKey: '2' },
]

export function ModeSelector({ mode, onChange }: ModeSelectorProps) {
  return (
    <div className={styles.group} role="radiogroup" aria-label="Search mode">
      {MODES.map((m) => (
        <button
          key={m.value}
          type="button"
          role="radio"
          aria-checked={mode === m.value}
          className={mode === m.value ? styles.active : styles.button}
          onClick={() => onChange(m.value)}
        >
          {m.label} <span className={styles.hint}>{m.shortcutKey}</span>
        </button>
      ))}
      <button type="button" className={styles.disabled} disabled title="Sắp có ở P2">
        DANTE <span className={styles.badge}>P2</span>
      </button>
    </div>
  )
}
