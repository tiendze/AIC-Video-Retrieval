import type { SearchMode } from '../api/types'
import styles from './ModeSelector.module.css'

export interface ModeSelectorProps {
  mode: SearchMode
  onChange: (mode: SearchMode) => void
  // plan §P1.8, PRD T5: true when GET /health reports elasticsearch_ok=false
  // -- OCR would just 503 (app/services/search_service.py's
  // "ocr_unavailable"), so disable picking it rather than let someone find
  // out via a failed search.
  ocrDisabled?: boolean
}

// PRD §F9 mode list: Semantic / OCR / DANTE. Semantic+OCR are live (P1);
// DANTE needs P2's sequence-search backend, so it's shown per plan §P1.5's
// decision ("hiện nhưng disabled + badge") rather than hidden -- the
// layout is settled once, P2 only has to wire the handler in.
const MODES: { value: SearchMode; label: string; shortcutKey: string }[] = [
  { value: 'semantic', label: 'Semantic', shortcutKey: '1' },
  { value: 'ocr', label: 'OCR', shortcutKey: '2' },
]

export function ModeSelector({ mode, onChange, ocrDisabled = false }: ModeSelectorProps) {
  return (
    <div className={styles.group} role="radiogroup" aria-label="Search mode">
      {MODES.map((m) => {
        const disabled = m.value === 'ocr' && ocrDisabled
        return (
          <button
            key={m.value}
            type="button"
            role="radio"
            aria-checked={mode === m.value}
            disabled={disabled}
            title={disabled ? 'Elasticsearch không khả dụng' : undefined}
            className={
              disabled ? styles.disabled : mode === m.value ? styles.active : styles.button
            }
            onClick={() => onChange(m.value)}
          >
            {m.label} <span className={styles.hint}>{m.shortcutKey}</span>
          </button>
        )
      })}
      <button type="button" className={styles.disabled} disabled title="Sắp có ở P2">
        DANTE <span className={styles.badge}>P2</span>
      </button>
    </div>
  )
}
