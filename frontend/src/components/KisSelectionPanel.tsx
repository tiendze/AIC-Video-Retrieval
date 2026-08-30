import { useState, type DragEvent, type ReactNode } from 'react'
import type { SearchHit } from '../api/types'
import {
  buildKisSubmissionPlan,
  buildVqaFinalRows,
  createManualKisSelection,
  createManualVqaSelection,
  kisSweepReachWarning,
  suggestedFilename,
  validateCsvFilename,
  type KisSubmissionPlan,
  type KisSweepDirection,
  type SelectedKisItem,
  type SelectedSearchItem,
  type SelectedVqaItem,
} from '../utils/submissionExport'
import styles from './SelectionPanel.module.css'

export type KisSelectionMode = 'kis' | 'qa'

export interface KisSelectionPanelProps {
  mode: KisSelectionMode
  onModeChange: (mode: KisSelectionMode) => void
  results: SearchHit[]
  kisSelection: SelectedKisItem[]
  vqaSelection: SelectedVqaItem[]
  kisExportName: string
  vqaExportName: string
  onKisExportNameChange: (value: string) => void
  onVqaExportNameChange: (value: string) => void
  onReorderKisSelection: (fromIndex: number, toIndex: number) => void
  onReorderVqaSelection: (fromIndex: number, toIndex: number) => void
  onRemoveKisSelection: (key: string) => void
  onAddManualKisSelection: (item: SelectedKisItem) => void
  onUpdateKisSelection: (key: string, patch: Partial<SelectedKisItem>) => void
  onAddManualVqaSelection: (item: SelectedVqaItem) => void
  onUpdateVqaSelection: (key: string, patch: Partial<SelectedVqaItem>) => void
  onRemoveVqaSelection: (key: string) => void
  onClearKisSelection: () => void
  onClearVqaSelection: () => void
  onClearVqaAnswers: () => void
  onVqaAnswerChange: (key: string, answer: string) => void
  kisFrameSweepStep: number
  onKisFrameSweepStepChange: (value: number) => void
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
  onReorderKisSelection,
  onReorderVqaSelection,
  onRemoveKisSelection,
  onAddManualKisSelection,
  onUpdateKisSelection,
  onAddManualVqaSelection,
  onUpdateVqaSelection,
  onRemoveVqaSelection,
  onClearKisSelection,
  onClearVqaSelection,
  onClearVqaAnswers,
  onVqaAnswerChange,
  kisFrameSweepStep,
  onKisFrameSweepStepChange,
  onExportKis,
  onExportVqa,
}: KisSelectionPanelProps) {
  const activeName = mode === 'kis' ? kisExportName : vqaExportName
  const validation = validateCsvFilename(activeName, mode)
  const finalKisPlan = buildKisSubmissionPlan(kisSelection, results)
  const finalVqaRows = buildVqaFinalRows(vqaSelection)
  const missingAnswers = vqaSelection.filter((item) => item.answer.length === 0)
  const overLimitAnswers = vqaSelection.filter((item) => item.answer.length > 100)
  const invalidKisSweeps = countInvalidSweepConfigs(kisSelection)
  const invalidVqaSweeps = countInvalidSweepConfigs(vqaSelection)
  const disableReason =
    mode === 'kis'
      ? kisDisableReason(
          validation.isValid,
          kisSelection.length,
          finalKisPlan.rows.length,
          invalidKisSweeps,
        )
      : vqaDisableReason(
          validation.isValid,
          vqaSelection.length,
          finalVqaRows.length,
          missingAnswers.length,
          overLimitAnswers.length,
          invalidVqaSweeps,
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
        {activeName.trim().length > 0 ? (
          <div className={styles.filename}>Will export: {validation.filename}</div>
        ) : (
          <div className={styles.hint}>Enter this first to unlock retrieval inputs.</div>
        )}
      </section>

      {mode === 'kis' ? (
        <KisSelectedList
          selection={kisSelection}
          finalCount={results.length}
          submissionPlan={finalKisPlan}
          frameSweepStep={kisFrameSweepStep}
          onFrameSweepStepChange={onKisFrameSweepStepChange}
          onAddManualSelection={onAddManualKisSelection}
          onUpdateSelection={onUpdateKisSelection}
          onReorder={onReorderKisSelection}
          onRemove={onRemoveKisSelection}
          onClear={onClearKisSelection}
        />
      ) : (
        <VqaSelectedList
          selection={vqaSelection}
          finalRows={finalVqaRows.length}
          frameSweepStep={kisFrameSweepStep}
          onFrameSweepStepChange={onKisFrameSweepStepChange}
          onAddManualSelection={onAddManualVqaSelection}
          onUpdateSelection={onUpdateVqaSelection}
          onReorder={onReorderVqaSelection}
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
  submissionPlan,
  frameSweepStep,
  onFrameSweepStepChange,
  onAddManualSelection,
  onUpdateSelection,
  onReorder,
  onRemove,
  onClear,
}: {
  selection: SelectedKisItem[]
  finalCount: number
  submissionPlan: KisSubmissionPlan
  frameSweepStep: number
  onFrameSweepStepChange: (value: number) => void
  onAddManualSelection: (item: SelectedKisItem) => void
  onUpdateSelection: (key: string, patch: Partial<SelectedKisItem>) => void
  onReorder: (fromIndex: number, toIndex: number) => void
  onRemove: (key: string) => void
  onClear: () => void
}) {
  const [manualVideoId, setManualVideoId] = useState('')
  const [manualFrameIdx, setManualFrameIdx] = useState('')
  const [manualSweepDirection, setManualSweepDirection] = useState<KisSweepDirection>('none')
  const [manualTargetFrameIdx, setManualTargetFrameIdx] = useState('')
  const [manualError, setManualError] = useState<string | null>(null)
  const manualCenter = /^\d+$/.test(manualFrameIdx.trim()) ? Number(manualFrameIdx) : null
  const manualTarget =
    /^\d+$/.test(manualTargetFrameIdx.trim()) && manualSweepDirection !== 'none'
      ? Number(manualTargetFrameIdx)
      : null
  const manualWarning =
    manualCenter === null
      ? null
      : kisSweepReachWarning(
          manualCenter,
          manualSweepDirection,
          frameSweepStep,
          manualTarget,
        )

  function addManualFrame(): void {
    const validation = validateManualFrameInput(
      manualVideoId,
      manualFrameIdx,
      manualSweepDirection,
      manualTargetFrameIdx,
      frameSweepStep,
    )
    if (validation) {
      setManualError(validation)
      return
    }
    onAddManualSelection(
      createManualKisSelection(
        manualVideoId,
        Number(manualFrameIdx),
        manualSweepDirection,
        frameSweepStep,
        manualTarget,
      ),
    )
    setManualVideoId('')
    setManualFrameIdx('')
    setManualSweepDirection('none')
    setManualTargetFrameIdx('')
    setManualError(null)
  }

  function skipManualFrame(): void {
    setManualVideoId('')
    setManualFrameIdx('')
    setManualSweepDirection('none')
    setManualTargetFrameIdx('')
    setManualError(null)
  }

  return (
    <section className={styles.section}>
      <div className={styles.sectionHeader}>
        <span>Selected Results</span>
        <span>{selection.length}</span>
      </div>
      <div className={styles.hint}>
        Final image ranking has {finalCount} frame(s); CSV writes {submissionPlan.rows.length} row(s).
      </div>
      <div className={styles.choiceGrid}>
        <div className={styles.choiceBlock}>
          <strong>Retrieval frame</strong>
          <span>Chọn frame bằng nút Select KIS ở kết quả retrieval.</span>
        </div>
        <div className={styles.choiceBlock}>
          <strong>Manual frame</strong>
          <span>Nhập video và keyframe nếu muốn chọn trực tiếp.</span>
        </div>
      </div>
      <div className={styles.manualGrid}>
        <label className={styles.label}>
          Video ID
          <input
            className={styles.input}
            value={manualVideoId}
            placeholder="L30_V046"
            onChange={(event) => setManualVideoId(event.target.value)}
          />
        </label>
        <label className={styles.label}>
          Frame Index
          <input
            className={styles.input}
            type="number"
            min={0}
            value={manualFrameIdx}
            placeholder="3255"
            onChange={(event) => setManualFrameIdx(event.target.value)}
          />
        </label>
      </div>
      <SweepEditor
        direction={manualSweepDirection}
        targetFrameIdx={manualTargetFrameIdx}
        step={frameSweepStep}
        onDirectionChange={(direction) => {
          setManualSweepDirection(direction)
          if (direction === 'none') setManualTargetFrameIdx('')
        }}
        onTargetFrameIdxChange={setManualTargetFrameIdx}
        onStepChange={onFrameSweepStepChange}
      />
      {manualError && <div className={styles.warning}>{manualError}</div>}
      {!manualError && manualWarning && <div className={styles.warning}>{manualWarning}</div>}
      <div className={styles.actions}>
        <button type="button" className={styles.compactButton} onClick={addManualFrame}>
          Add Frame
        </button>
        <button type="button" className={styles.compactButton} onClick={skipManualFrame}>
          Skip
        </button>
      </div>
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
                onUpdateKisSelection={onUpdateSelection}
                onReorder={onReorder}
                onRemove={onRemove}
              />
            ))}
          </div>
          <SubmissionPreview plan={submissionPlan} />
        </>
      )}
    </section>
  )
}

function VqaSelectedList({
  selection,
  finalRows,
  frameSweepStep,
  onFrameSweepStepChange,
  onAddManualSelection,
  onUpdateSelection,
  onReorder,
  onRemove,
  onClear,
  onClearAnswers,
  onAnswerChange,
}: {
  selection: SelectedVqaItem[]
  finalRows: number
  frameSweepStep: number
  onFrameSweepStepChange: (value: number) => void
  onAddManualSelection: (item: SelectedVqaItem) => void
  onUpdateSelection: (key: string, patch: Partial<SelectedVqaItem>) => void
  onReorder: (fromIndex: number, toIndex: number) => void
  onRemove: (key: string) => void
  onClear: () => void
  onClearAnswers: () => void
  onAnswerChange: (key: string, answer: string) => void
}) {
  const [manualVideoId, setManualVideoId] = useState('')
  const [manualFrameIdx, setManualFrameIdx] = useState('')
  const [manualAnswer, setManualAnswer] = useState('')
  const [manualSweepDirection, setManualSweepDirection] = useState<KisSweepDirection>('none')
  const [manualTargetFrameIdx, setManualTargetFrameIdx] = useState('')
  const [manualError, setManualError] = useState<string | null>(null)
  const manualAnswerValid = manualAnswer.length > 0 && manualAnswer.length <= 100
  const manualFrameValid = /^\d+$/.test(manualFrameIdx.trim())
  const manualTarget =
    /^\d+$/.test(manualTargetFrameIdx.trim()) && manualSweepDirection !== 'none'
      ? Number(manualTargetFrameIdx)
      : null
  const manualWarning =
    manualFrameValid && manualAnswerValid
      ? kisSweepReachWarning(
          Number(manualFrameIdx),
          manualSweepDirection,
          frameSweepStep,
          manualTarget,
        )
      : null

  function addManualFrame(): void {
    const validation = validateManualVqaFrameInput(
      manualVideoId,
      manualFrameIdx,
      manualAnswer,
      manualSweepDirection,
      manualTargetFrameIdx,
      frameSweepStep,
    )
    if (validation) {
      setManualError(validation)
      return
    }
    onAddManualSelection(
      createManualVqaSelection(
        manualVideoId,
        Number(manualFrameIdx),
        manualAnswer,
        manualSweepDirection,
        frameSweepStep,
        manualTarget,
      ),
    )
    setManualVideoId('')
    setManualFrameIdx('')
    setManualAnswer('')
    setManualSweepDirection('none')
    setManualTargetFrameIdx('')
    setManualError(null)
  }

  function clearManualFrame(): void {
    setManualVideoId('')
    setManualFrameIdx('')
    setManualAnswer('')
    setManualSweepDirection('none')
    setManualTargetFrameIdx('')
    setManualError(null)
  }

  return (
    <section className={styles.section}>
      <div className={styles.sectionHeader}>
        <span>Selected VQA Results</span>
        <span>{selection.length}</span>
      </div>
      <div className={styles.hint}>CSV writes {finalRows} answered row(s).</div>
      <div className={styles.choiceGrid}>
        <div className={styles.choiceBlock}>
          <strong>Retrieval frame</strong>
          <span>Chọn frame bằng nút Select VQA rồi nhập answer trên item.</span>
        </div>
        <div className={styles.choiceBlock}>
          <strong>Manual frame</strong>
          <span>Nhập video, frame_idx và answer trước khi chọn khoảng.</span>
        </div>
      </div>
      <div className={styles.manualGrid}>
        <label className={styles.label}>
          Video ID
          <input
            className={styles.input}
            value={manualVideoId}
            placeholder="L30_V046"
            onChange={(event) => setManualVideoId(event.target.value)}
          />
        </label>
        <label className={styles.label}>
          Frame Index
          <input
            className={styles.input}
            type="number"
            min={0}
            value={manualFrameIdx}
            placeholder="3255"
            onChange={(event) => setManualFrameIdx(event.target.value)}
          />
        </label>
      </div>
      <div className={styles.answerBlock}>
        <label className={styles.label} htmlFor="manual-vqa-answer">
          Answer
        </label>
        <input
          id="manual-vqa-answer"
          className={`${styles.answerInput} ${
            manualAnswer.length > 100 ? styles.inputInvalid : ''
          }`}
          value={manualAnswer}
          onChange={(event) => setManualAnswer(event.target.value)}
        />
        <div className={styles.answerMeta}>
          <span>{manualAnswer.length === 0 ? 'Required before range' : 'Manual answer'}</span>
          <span className={manualAnswer.length > 100 ? styles.answerOver : ''}>
            {manualAnswer.length}/100
          </span>
        </div>
      </div>
      {manualFrameValid && manualAnswerValid ? (
        <SweepEditor
          direction={manualSweepDirection}
          targetFrameIdx={manualTargetFrameIdx}
          step={frameSweepStep}
          onDirectionChange={(direction) => {
            setManualSweepDirection(direction)
            if (direction === 'none') setManualTargetFrameIdx('')
          }}
          onTargetFrameIdxChange={setManualTargetFrameIdx}
          onStepChange={onFrameSweepStepChange}
        />
      ) : (
        <div className={styles.hint}>Fill frame_idx and answer before choosing a range.</div>
      )}
      {manualError && <div className={styles.warning}>{manualError}</div>}
      {!manualError && manualWarning && <div className={styles.warning}>{manualWarning}</div>}
      <div className={styles.actions}>
        <button type="button" className={styles.compactButton} onClick={addManualFrame}>
          Add Frame
        </button>
        <button type="button" className={styles.compactButton} onClick={clearManualFrame}>
          Clear manual
        </button>
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
                onUpdateVqaSelection={onUpdateSelection}
                onReorder={onReorder}
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
                    <span>{item.answer.length === 0 ? 'Required before range' : 'Independent answer'}</span>
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
  onUpdateKisSelection,
  onUpdateVqaSelection,
  onReorder,
  onRemove,
  children,
}: {
  item: SelectedSearchItem
  index: number
  total: number
  onUpdateKisSelection?: (key: string, patch: Partial<SelectedKisItem>) => void
  onUpdateVqaSelection?: (key: string, patch: Partial<SelectedVqaItem>) => void
  onReorder: (fromIndex: number, toIndex: number) => void
  onRemove: (key: string) => void
  children?: ReactNode
}) {
  const [dropPosition, setDropPosition] = useState<'before' | 'after' | null>(null)
  const answer = isVqaItem(item) ? item.answer : null
  const incomplete = answer !== null && (answer.length === 0 || answer.length > 100)
  const sweepItem = isKisItem(item) || isVqaItem(item) ? item : null
  const answerAllowsSweep = !isVqaItem(item) || (item.answer.length > 0 && item.answer.length <= 100)
  const sweepWarning = sweepItem
    ? kisSweepReachWarning(
        sweepItem.hit.frame_idx,
        sweepItem.sweepDirection,
        sweepItem.step,
        sweepItem.targetFrameIdx,
      )
    : null
  function handleDragStart(event: DragEvent<HTMLDivElement>): void {
    event.dataTransfer.effectAllowed = 'move'
    event.dataTransfer.setData('text/plain', String(index))
  }

  function handleDrop(event: DragEvent<HTMLDivElement>): void {
    event.preventDefault()
    const fromIndex = Number(event.dataTransfer.getData('text/plain'))
    if (!Number.isInteger(fromIndex)) return
    const insertionIndex = dropPosition === 'after' ? index + 1 : index
    setDropPosition(null)
    onReorder(fromIndex, insertionIndex)
  }

  return (
    <div
      className={[
        styles.item,
        incomplete ? styles.itemIncomplete : '',
        dropPosition === 'before' ? styles.itemDropBefore : '',
        dropPosition === 'after' ? styles.itemDropAfter : '',
      ]
        .filter(Boolean)
        .join(' ')}
      draggable={total > 1}
      onDragStart={handleDragStart}
      onDragOver={(event) => {
        event.preventDefault()
        event.dataTransfer.dropEffect = 'move'
        const rect = event.currentTarget.getBoundingClientRect()
        const nextPosition = event.clientY < rect.top + rect.height / 2 ? 'before' : 'after'
        setDropPosition(nextPosition)
      }}
      onDragLeave={() => setDropPosition(null)}
      onDragEnd={() => setDropPosition(null)}
      onDrop={handleDrop}
    >
      <div className={styles.ordinal} title="Drag to reorder">
        {index + 1}
      </div>
      <div className={styles.itemBody}>
        <div className={styles.frameRow}>
          {item.hit.thumbnail_url ? (
            <img className={styles.thumb} src={item.hit.thumbnail_url} alt="" loading="lazy" />
          ) : (
            <div className={styles.thumbPlaceholder} aria-hidden="true">
              Manual
            </div>
          )}
          <div className={styles.meta}>
            <strong>{item.hit.video_id}</strong>
            <div className={styles.secondary}>frame {item.hit.frame_idx}</div>
            {sweepItem && (
              <div className={styles.secondary}>
                {formatSource(sweepItem.source)} · {formatSelectionSweep(sweepItem)}
              </div>
            )}
            {item.originalRank >= 0 && (
              <div className={styles.secondary}>Original #{item.originalRank + 1}</div>
            )}
          </div>
        </div>
        {isKisItem(item) && onUpdateKisSelection && (
          <SweepEditor
            direction={item.sweepDirection}
            targetFrameIdx={item.targetFrameIdx === null ? '' : String(item.targetFrameIdx)}
            step={item.step}
            compact
            onDirectionChange={(direction) =>
              onUpdateKisSelection(item.key, {
                sweepDirection: direction,
                targetFrameIdx: direction === 'none' ? null : item.targetFrameIdx,
              })
            }
            onTargetFrameIdxChange={(value) =>
              onUpdateKisSelection(item.key, {
                targetFrameIdx: /^\d+$/.test(value.trim()) ? Number(value) : null,
              })
            }
            onStepChange={(value) => onUpdateKisSelection(item.key, { step: value })}
          />
        )}
        {children}
        {isVqaItem(item) && onUpdateVqaSelection && (
          answerAllowsSweep ? (
            <SweepEditor
              direction={item.sweepDirection}
              targetFrameIdx={item.targetFrameIdx === null ? '' : String(item.targetFrameIdx)}
              step={item.step}
              compact
              onDirectionChange={(direction) =>
                onUpdateVqaSelection(item.key, {
                  sweepDirection: direction,
                  targetFrameIdx: direction === 'none' ? null : item.targetFrameIdx,
                })
              }
              onTargetFrameIdxChange={(value) =>
                onUpdateVqaSelection(item.key, {
                  targetFrameIdx: /^\d+$/.test(value.trim()) ? Number(value) : null,
                })
              }
              onStepChange={(value) => onUpdateVqaSelection(item.key, { step: value })}
            />
          ) : (
            <div className={styles.hint}>Fill answer before choosing a range.</div>
          )
        )}
        {sweepWarning && <div className={styles.warning}>{sweepWarning}</div>}
        <div className={styles.actions}>
          {total > 1 && <span className={styles.dragHint}>Drag to reorder</span>}
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

function isKisItem(item: SelectedSearchItem): item is SelectedKisItem {
  return 'source' in item && 'step' in item && 'sweepDirection' in item
}

function SweepEditor({
  direction,
  targetFrameIdx,
  step,
  compact = false,
  onDirectionChange,
  onTargetFrameIdxChange,
  onStepChange,
}: {
  direction: KisSweepDirection
  targetFrameIdx: string
  step: number
  compact?: boolean
  onDirectionChange: (direction: KisSweepDirection) => void
  onTargetFrameIdxChange: (value: string) => void
  onStepChange: (value: number) => void
}) {
  return (
    <div className={compact ? styles.sweepEditorCompact : styles.sweepEditor}>
      <div className={styles.directionToggle} role="radiogroup" aria-label="frame sweep direction">
        {(['none', 'left', 'right'] as const).map((option) => (
          <button
            key={option}
            type="button"
            className={`${styles.taskButton} ${direction === option ? styles.taskButtonActive : ''}`}
            aria-pressed={direction === option}
            onClick={() => onDirectionChange(option)}
          >
            {option === 'none' ? 'None' : option === 'left' ? 'Left' : 'Right'}
          </button>
        ))}
      </div>
      {direction !== 'none' && (
        <div className={styles.sweepGrid}>
          <label className={styles.label}>
            Target frame_idx
            <input
              className={styles.input}
              type="number"
              min={0}
              value={targetFrameIdx}
              onChange={(event) => onTargetFrameIdxChange(event.target.value)}
            />
          </label>
          <label className={styles.label}>
            Step
            <input
              className={styles.input}
              type="number"
              min={1}
              value={step}
              onChange={(event) => onStepChange(Number(event.currentTarget.value))}
            />
          </label>
        </div>
      )}
    </div>
  )
}

function SubmissionPreview({ plan }: { plan: KisSubmissionPlan }) {
  if (plan.rows.length === 0) return null
  return (
    <div className={styles.preview}>
      <div className={styles.sectionHeader}>
        <span>Submission order</span>
        <span>{plan.rows.length}</span>
      </div>
      {plan.groups.map((group, index) => {
        const { item } = group
        return (
          <div className={styles.previewRow} key={item.key}>
            <span>{index + 1}</span>
            <strong>{item.source.toUpperCase()}</strong>
            <span>
              {item.hit.video_id} · frame {item.hit.frame_idx}
              {group.sweepStart !== null &&
                group.sweepEnd !== null &&
                ` · sweep ${group.sweepStart}-${group.sweepEnd} · step ${item.step}`}
            </span>
            <em>{group.emittedCount}</em>
          </div>
        )
      })}
      <div className={styles.previewFooter}>Ranking fallback: {plan.rankingFallbackCount} frames</div>
    </div>
  )
}

function validateManualFrameInput(
  videoId: string,
  frameIdx: string,
  direction: KisSweepDirection,
  targetFrameIdx: string,
  step: number,
): string | null {
  if (videoId.trim().length === 0) return 'Enter a video ID.'
  if (!/^\d+$/.test(frameIdx.trim())) return 'Frame Index must be a non-negative integer.'
  if (direction === 'none') return null
  if (!/^\d+$/.test(targetFrameIdx.trim())) {
    return 'Target frame_idx must be a non-negative integer.'
  }
  if (!Number.isFinite(step) || step <= 0) return 'Step must be greater than zero.'
  const warning = kisSweepReachWarning(
    Number(frameIdx),
    direction,
    step,
    Number(targetFrameIdx),
  )
  if (warning?.startsWith('Target frame')) return warning
  return null
}

function validateManualVqaFrameInput(
  videoId: string,
  frameIdx: string,
  answer: string,
  direction: KisSweepDirection,
  targetFrameIdx: string,
  step: number,
): string | null {
  if (videoId.trim().length === 0) return 'Enter a video ID.'
  if (!/^\d+$/.test(frameIdx.trim())) return 'Frame Index must be a non-negative integer.'
  if (answer.length === 0) return 'Enter an answer before adding this VQA frame.'
  if (answer.length > 100) return 'Answer must be 100 characters or fewer.'
  if (direction === 'none') return null
  if (!/^\d+$/.test(targetFrameIdx.trim())) {
    return 'Target frame_idx must be a non-negative integer.'
  }
  if (!Number.isFinite(step) || step <= 0) return 'Step must be greater than zero.'
  const warning = kisSweepReachWarning(
    Number(frameIdx),
    direction,
    step,
    Number(targetFrameIdx),
  )
  if (warning?.startsWith('Target frame')) return warning
  return null
}

function formatSource(source: SelectedKisItem['source']): string {
  return source === 'manual' ? 'Manual' : 'Visual'
}

function formatSelectionSweep(
  item: Pick<SelectedKisItem | SelectedVqaItem, 'sweepDirection' | 'targetFrameIdx' | 'step'>,
): string {
  if (item.sweepDirection === 'none') return 'sweep none'
  const target = item.targetFrameIdx === null ? '?' : item.targetFrameIdx
  return `sweep ${item.sweepDirection} to ${target} · step ${item.step}`
}

function kisDisableReason(
  isFilenameValid: boolean,
  selectedCount: number,
  finalRowCount: number,
  invalidSweepCount: number,
): string | null {
  if (!isFilenameValid) return 'Fix the export filename before exporting.'
  if (selectedCount === 0) return 'Select a primary KIS frame before exporting.'
  if (invalidSweepCount > 0) return `${invalidSweepCount} selected frame(s) need a valid sweep target.`
  if (finalRowCount === 0) return 'No KIS rows available to export.'
  return null
}

function vqaDisableReason(
  isFilenameValid: boolean,
  selectedCount: number,
  answeredRows: number,
  missingAnswers: number,
  overLimitAnswers: number,
  invalidSweepCount: number,
): string | null {
  if (!isFilenameValid) return 'Fix the export filename before exporting.'
  if (selectedCount === 0) return 'Select at least one VQA frame.'
  if (missingAnswers > 0) return `${missingAnswers} selected frame(s) need an answer.`
  if (overLimitAnswers > 0) return `${overLimitAnswers} answer(s) exceed 100 characters.`
  if (invalidSweepCount > 0) return `${invalidSweepCount} selected frame(s) need a valid sweep target.`
  if (answeredRows === 0) return 'No answered VQA rows available to export.'
  return null
}

function countInvalidSweepConfigs(
  items: Array<Pick<SelectedKisItem | SelectedVqaItem, 'hit' | 'sweepDirection' | 'step' | 'targetFrameIdx'>>,
): number {
  return items.filter((item) => {
    if (item.sweepDirection === 'none') return false
    if (item.targetFrameIdx === null) return true
    const warning = kisSweepReachWarning(
      item.hit.frame_idx,
      item.sweepDirection,
      item.step,
      item.targetFrameIdx,
    )
    return warning?.startsWith('Target frame') ?? false
  }).length
}
