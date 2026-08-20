import type { TrakeVideoResultItem } from '../api/types'
import {
  buildTrakeFinalRanking,
  suggestedFilename,
  validateCsvFilename,
  validateTrakeEventCount,
  type SelectedTrakeSequence,
} from '../utils/submissionExport'
import styles from './SelectionPanel.module.css'

export interface TrakeSelectionPanelProps {
  results: TrakeVideoResultItem[]
  selection: SelectedTrakeSequence[]
  exportName: string
  eventCount: number
  onExportNameChange: (value: string) => void
  onMoveSelection: (index: number, direction: -1 | 1) => void
  onRemoveSelection: (key: string) => void
  onClearSelection: () => void
  onExport: () => void
}

export function TrakeSelectionPanel({
  results,
  selection,
  exportName,
  eventCount,
  onExportNameChange,
  onMoveSelection,
  onRemoveSelection,
  onClearSelection,
  onExport,
}: TrakeSelectionPanelProps) {
  const validation = validateCsvFilename(exportName, 'trake')
  const finalRows = buildTrakeFinalRanking(results, selection)
  const eventCountError = validateTrakeEventCount(finalRows, eventCount)
  const disableReason =
    !validation.isValid
      ? 'Fix the export filename before exporting.'
      : finalRows.length === 0
        ? 'No TRAKE rows available to export.'
        : eventCountError

  return (
    <aside className={styles.panel} aria-label="TRAKE selection export">
      <div className={styles.panelTitle}>
        <span>TRAKE Selection</span>
        <span>{selection.length}</span>
      </div>

      <section className={styles.section}>
        <label className={styles.label} htmlFor="trake-export-name">
          Query export name
        </label>
        <input
          id="trake-export-name"
          className={`${styles.input} ${validation.isValid ? '' : styles.inputInvalid}`}
          value={exportName}
          onChange={(event) => onExportNameChange(event.target.value)}
        />
        {validation.message && (
          <div className={styles.warning}>
            {validation.message}{' '}
            <button
              type="button"
              className={styles.compactButton}
              onClick={() => onExportNameChange(suggestedFilename(exportName, 'trake'))}
            >
              Fix
            </button>
          </div>
        )}
        <div className={styles.filename}>Will export: {validation.filename}</div>
      </section>

      <section className={styles.section}>
        <div className={styles.sectionHeader}>
          <span>Selected sequences</span>
          <span>{selection.length}</span>
        </div>
        <div className={styles.hint}>Final TRAKE export will contain {finalRows.length} row(s).</div>
        {selection.length === 0 ? (
          <div className={styles.empty}>Selected sequences appear here.</div>
        ) : (
          <>
            <div className={styles.actions}>
              <button type="button" className={styles.compactButton} onClick={onClearSelection}>
                Clear selected
              </button>
            </div>
            <div className={styles.selectedList}>
              {selection.map((item, index) => (
                <div
                  key={item.key}
                  className={`${styles.item} ${
                    item.result.keyframes.length !== eventCount ? styles.itemIncomplete : ''
                  }`}
                >
                  <div className={styles.ordinal}>{index + 1}</div>
                  <div className={styles.itemBody}>
                    <div className={styles.meta}>
                      <strong>{item.result.video_id}</strong>
                      <div className={styles.sequenceFrames}>
                        {item.result.keyframes.map((keyframe) => keyframe.frame_idx).join(' → ')}
                      </div>
                      <div className={styles.secondary}>Score {item.result.score.toFixed(4)}</div>
                      <div className={styles.secondary}>Original #{item.originalRank + 1}</div>
                    </div>
                    <div className={styles.actions}>
                      <button
                        type="button"
                        className={styles.compactButton}
                        disabled={index === 0}
                        onClick={() => onMoveSelection(index, -1)}
                        aria-label={`Move TRAKE sequence ${index + 1} up`}
                      >
                        ↑
                      </button>
                      <button
                        type="button"
                        className={styles.compactButton}
                        disabled={index === selection.length - 1}
                        onClick={() => onMoveSelection(index, 1)}
                        aria-label={`Move TRAKE sequence ${index + 1} down`}
                      >
                        ↓
                      </button>
                      <button
                        type="button"
                        className={styles.compactButton}
                        onClick={() => onRemoveSelection(item.key)}
                      >
                        Remove
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </>
        )}
      </section>

      <div className={styles.footer}>
        {disableReason && <div className={styles.warning}>{disableReason}</div>}
        <button
          type="button"
          className={styles.exportButton}
          disabled={disableReason !== null}
          onClick={onExport}
        >
          Export TRAKE CSV
        </button>
      </div>
    </aside>
  )
}
