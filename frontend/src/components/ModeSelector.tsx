import type { AppMode } from '../api/types'
import styles from './ModeSelector.module.css'

export interface ModeSelectorProps {
  mode: AppMode
  onChange: (mode: AppMode) => void
}

const MODES: { value: AppMode; label: string; shortcutKey: string }[] = [
  { value: 'kis', label: 'KIS', shortcutKey: '1' },
  { value: 'dante', label: 'TRAKE', shortcutKey: '2' },
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
    </div>
  )
}
