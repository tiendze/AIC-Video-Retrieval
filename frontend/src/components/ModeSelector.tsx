import type { AppMode } from '../api/types'
import styles from './ModeSelector.module.css'

export interface ModeSelectorProps {
  mode: AppMode
  onChange: (mode: AppMode) => void
  // plan §P1.8, PRD T5: true when GET /health reports elasticsearch_ok=false
  // -- OCR would just 503 (app/services/search_service.py's
  // "ocr_unavailable"), so disable picking it rather than let someone find
  // out via a failed search.
  ocrDisabled?: boolean
}

// PRD §F9 mode list: Semantic / OCR / DANTE. All three are live as of
// 2026-08-15 -- DANTE was the P2 disabled placeholder (plan §P1.5's
// decision: "hiện nhưng disabled + badge") until POST /api/trake existed
// to wire it to.
const MODES: { value: AppMode; label: string; shortcutKey: string }[] = [
  { value: 'semantic', label: 'Semantic', shortcutKey: '1' },
  { value: 'ocr', label: 'OCR', shortcutKey: '2' },
  { value: 'dante', label: 'DANTE', shortcutKey: '3' },
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
    </div>
  )
}
