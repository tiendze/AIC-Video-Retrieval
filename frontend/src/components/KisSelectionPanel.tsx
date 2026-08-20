import type { ReactNode } from 'react'
import type { SearchHit } from '../api/types'
import {
  buildKisFinalRanking,
  buildVqaFinalRows,
  suggestedFilename,
  validateCsvFilename,
  type SelectedSearchItem,
  type SelectedVqaItem,
} from '../utils/submissionExport'
import styles from './SelectionPanel.module.css'

export type KisSelectionMode = 'kis' | 'qa'

export interface KisSelectionPanelProps {
  mode: KisSelectionMode
  onModeChange: (mode: KisSelectionMode) => void
  results: SearchHit[]
  kisSelection: SelectedSearchItem[]
  vqaSelection: SelectedVqaItem[]
  kisExportName: string
  vqaExportName: string
  onKisExportNameChange: (value: string) => void
  onVqaExportNameChange: (value: string) => void
  onMoveKisSelection: (index: number, direction: -1 | 1) => void
  onMoveVqaSelection: (index: number, direction: -1 | 1) => void
  onRemoveKisSelection: (key: string) => void
  onRemoveVqaSelection: (key: string) => void
  onClearKisSelection: () => void
  onClearVqaSelection: () => void
  onClearVqaAnswers: () => void
  onVqaAnswerChange: (key: string, answer: string) => void
  onExportKis: () => void
  onExportVqa: () => void
}

export function KisSelectionPanel({
  mode,
  onModeChange,
  results,
  kisSelection,
  vqaSelection,
  kisExportName,
  vqaExportName,
  onKisExportNameChange,
  onVqaExportNameChange,
  onMoveKisSelection,
  onMoveVqaSelection,
  onRemoveKisSelection,
  onRemoveVqaSelection,
  onClearKisSelection,
  onClearVqaSelection,
  onClearVqaAnswers,
  onVqaAnswerChange,
  onExportKis,
  onExportVqa,
}: KisSelectionPanelProps) {
  const activeName = mode === 'kis' ? kisExportName : vqaExportName
  const validation = validateCsvFilename(activeName, mode)
  const finalKisRows = buildKisFinalRanking(results, kisSelection)
  const finalVqaRows = buildVqaFinalRows(vqaSelection)
  const missingAnswers = vqaSelection.filter((item) => item.answer.length === 0)
  const overLimitAnswers = vqaSelection.filter((item) => item.answer.length > 100)
  const disableReason =
    mode === 'kis'
      ? kisDisableReason(validation.isValid, finalKisRows.length)
      : vqaDisableReason(
          validation.isValid,
          vqaSelection.length,
          finalVqaRows.length,
          missingAnswers.length,
          overLimitAnswers.length,
        )

  return (
    <div>
      <section className={styles.section}>
        <div className={styles.sectionHeader}>
          <span>Task</span>
        </div>
        <div className={styles.taskToggle} role="radiogroup" aria-label="submission task">
          <button
            type="button"
            className={`${styles.taskButton} ${mode === 'kis' ? styles.taskButtonActive : ''}`}
            aria-pressed={mode === 'kis'}
            onClick={() => onModeChange('kis')}
          >
            KIS
          </button>
          <button
            type="button"
            className={`${styles.taskButton} ${mode === 'qa' ? styles.taskButtonActive : ''}`}
            aria-pressed={mode === 'qa'}
            onClick={() => onModeChange('qa')}
          >
            VQA
          </button>
        </div>
      </section>

      <section className={styles.section}>
        <label className={styles.label} htmlFor="kis-export-name">
          Query export name
        </label>
        <input
          id="kis-export-name"
          className={`${styles.input} ${validation.isValid ? '' : styles.inputInvalid}`}
          value={activeName}
          onChange={(event) =>
            mode === 'kis'
              ? onKisExportNameChange(event.target.value)
              : onVqaExportNameChange(event.target.value)
          }
        />
        {validation.message && (
          <div className={styles.warning}>
            {validation.message}{' '}
            <button
              type="button"
              className={styles.compactButton}
              onClick={() =>
                mode === 'kis'
                  ? onKisExportNameChange(suggestedFilename(activeName, 'kis'))
                  : onVqaExportNameChange(suggestedFilename(activeName, 'qa'))
              }
            >
              Fix
            </button>
          </div>
        )}
        <div className={styles.filename}>Will export: {validation.filename}</div>
      </section>

      {mode === 'kis' ? (
        <KisSelectedList
          selection={kisSelection}
          finalCount={finalKisRows.length}
          onMove={onMoveKisSelection}
          onRemove={onRemoveKisSelection}
          onClear={onClearKisSelection}
        />
      ) : (
        <VqaSelectedList
          selection={vqaSelection}
          onMove={onMoveVqaSelection}
          onRemove={onRemoveVqaSelection}
          onClear={onClearVqaSelection}
          onClearAnswers={onClearVqaAnswers}
          onAnswerChange={onVqaAnswerChange}
        />
      )}

      <div className={styles.footer}>
        {disableReason && <div className={styles.warning}>{disableReason}</div>}
        <button
          type="button"
          className={styles.exportButton}
          disabled={disableReason !== null}
          onClick={mode === 'kis' ? onExportKis : onExportVqa}
        >
          {mode === 'kis' ? 'Export KIS CSV' : 'Export VQA CSV'}
        </button>
      </div>
    </div>
  )
}

function KisSelectedList({
  selection,
  finalCount,
  onMove,
  onRemove,
  onClear,
}: {
  selection: SelectedSearchItem[]
  finalCount: number
  onMove: (index: number, direction: -1 | 1) => void
  onRemove: (key: string) => void
  onClear: () => void
}) {
  return (
    <section className={styles.section}>
      <div className={styles.sectionHeader}>
        <span>Selected Results</span>
        <span>{selection.length}</span>
      </div>
      <div className={styles.hint}>Final KIS export will contain {finalCount} row(s).</div>
      {selection.length === 0 ? (
        <div className={styles.empty}>Selected frames appear here.</div>
      ) : (
        <>
          <div className={styles.actions}>
            <button type="button" className={styles.compactButton} onClick={onClear}>
              Clear selected
            </button>
          </div>
          <div className={styles.selectedList}>
            {selection.map((item, index) => (
              <FrameItem
                key={item.key}
                item={item}
                index={index}
                total={selection.length}
                onMove={onMove}
                onRemove={onRemove}
              />
            ))}
          </div>
        </>
      )}
    </section>
  )
}

function VqaSelectedList({
  selection,
  onMove,
  onRemove,
  onClear,
  onClearAnswers,
  onAnswerChange,
}: {
  selection: SelectedVqaItem[]
  onMove: (index: number, direction: -1 | 1) => void
  onRemove: (key: string) => void
  onClear: () => void
  onClearAnswers: () => void
  onAnswerChange: (key: string, answer: string) => void
}) {
  return (
    <section className={styles.section}>
      <div className={styles.sectionHeader}>
        <span>Selected VQA Results</span>
        <span>{selection.length}</span>
      </div>
      {selection.length === 0 ? (
        <div className={styles.empty}>Selected VQA frames appear here.</div>
      ) : (
        <>
          <div className={styles.actions}>
            <button type="button" className={styles.compactButton} onClick={onClear}>
              Clear selected
            </button>
            <button type="button" className={styles.compactButton} onClick={onClearAnswers}>
              Clear answers
            </button>
          </div>
          <div className={styles.selectedList}>
            {selection.map((item, index) => (
              <FrameItem
                key={item.key}
                item={item}
                index={index}
                total={selection.length}
                onMove={onMove}
                onRemove={onRemove}
              >
                <div className={styles.answerBlock}>
                  <label className={styles.label} htmlFor={`vqa-answer-${item.key}`}>
                    Answer
                  </label>
                  <input
                    id={`vqa-answer-${item.key}`}
                    className={`${styles.answerInput} ${
                      item.answer.length === 0 || item.answer.length > 100
                        ? styles.inputInvalid
                        : ''
                    }`}
                    value={item.answer}
                    onChange={(event) => onAnswerChange(item.key, event.target.value)}
                    aria-invalid={item.answer.length === 0 || item.answer.length > 100}
                  />
                  <div className={styles.answerMeta}>
                    <span>{item.answer.length === 0 ? 'Required' : 'Independent answer'}</span>
                    <span className={item.answer.length > 100 ? styles.answerOver : ''}>
                      {item.answer.length}/100
                    </span>
                  </div>
                </div>
              </FrameItem>
            ))}
          </div>
        </>
      )}
    </section>
  )
}

function FrameItem({
  item,
  index,
  total,
  onMove,
  onRemove,
  children,
}: {
  item: SelectedSearchItem
  index: number
  total: number
  onMove: (index: number, direction: -1 | 1) => void
  onRemove: (key: string) => void
  children?: ReactNode
}) {
  const answer = isVqaItem(item) ? item.answer : null
  const incomplete = answer !== null && (answer.length === 0 || answer.length > 100)
  return (
    <div className={`${styles.item} ${incomplete ? styles.itemIncomplete : ''}`}>
      <div className={styles.ordinal}>{index + 1}</div>
      <div className={styles.itemBody}>
        <div className={styles.frameRow}>
          <img className={styles.thumb} src={item.hit.thumbnail_url} alt="" loading="lazy" />
          <div className={styles.meta}>
            <strong>{item.hit.video_id}</strong>
            <div className={styles.secondary}>frame {item.hit.frame_idx}</div>
            <div className={styles.secondary}>Original #{item.originalRank + 1}</div>
          </div>
        </div>
        {children}
        <div className={styles.actions}>
          <button
            type="button"
            className={styles.compactButton}
            disabled={index === 0}
            onClick={() => onMove(index, -1)}
            aria-label={`Move ${item.hit.video_id} frame ${item.hit.frame_idx} up`}
          >
            ↑
          </button>
          <button
            type="button"
            className={styles.compactButton}
            disabled={index === total - 1}
            onClick={() => onMove(index, 1)}
            aria-label={`Move ${item.hit.video_id} frame ${item.hit.frame_idx} down`}
          >
            ↓
          </button>
          <button
            type="button"
            className={styles.compactButton}
            onClick={() => onRemove(item.key)}
          >
            Remove
          </button>
        </div>
      </div>
    </div>
  )
}

function isVqaItem(item: SelectedSearchItem): item is SelectedVqaItem {
  return 'answer' in item && typeof item.answer === 'string'
}

function kisDisableReason(isFilenameValid: boolean, finalRowCount: number): string | null {
  if (!isFilenameValid) return 'Fix the export filename before exporting.'
  if (finalRowCount === 0) return 'No KIS rows available to export.'
  return null
}

function vqaDisableReason(
  isFilenameValid: boolean,
  selectedCount: number,
  answeredRows: number,
  missingAnswers: number,
  overLimitAnswers: number,
): string | null {
  if (!isFilenameValid) return 'Fix the export filename before exporting.'
  if (selectedCount === 0) return 'Select at least one VQA frame.'
  if (missingAnswers > 0) return `${missingAnswers} selected frame(s) need an answer.`
  if (overLimitAnswers > 0) return `${overLimitAnswers} answer(s) exceed 100 characters.`
  if (answeredRows === 0) return 'No answered VQA rows available to export.'
  return null
}
