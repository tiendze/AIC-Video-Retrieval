import { useState, type DragEvent } from 'react'
import type { TrakeVideoResultItem } from '../api/types'
import {
  buildTrakeSubmissionPlan,
  buildTrakeUniformShiftDeltas,
  createManualTrakeSelection,
  serializeTrakeCsv,
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
  onAddManualSelection: (item: SelectedTrakeSequence) => void
  onUpdateSelection: (key: string, patch: Partial<SelectedTrakeSequence>) => void
  onReorderSelection: (fromIndex: number, toIndex: number) => void
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
  onAddManualSelection,
  onUpdateSelection,
  onReorderSelection,
  onRemoveSelection,
  onClearSelection,
  onExport,
}: TrakeSelectionPanelProps) {
  const validation = validateCsvFilename(exportName, 'trake')
  const plan = buildTrakeSubmissionPlan(selection, results)
  const finalRows = plan.rows
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
          <span>Selected Results</span>
          <span>{selection.length}</span>
        </div>
        <div className={styles.choiceGrid}>
          <div className={styles.choiceBlock}>
            <strong>Retrieval sequence</strong>
            <span>Chọn sequence bằng nút Select ở kết quả TRAKE.</span>
          </div>
          <div className={styles.choiceBlock}>
            <strong>Manual sequence</strong>
            <span>Nhập video và keyframes đúng thứ tự để thêm thẳng vào CSV.</span>
          </div>
        </div>
        <ManualTrakeSequenceForm
          eventCount={eventCount}
          onAddManualSelection={onAddManualSelection}
        />
        <div className={styles.hint}>
          Final TRAKE export: {finalRows.length} row(s) - {plan.centerCount} original,{' '}
          {plan.variantCount} margin, {plan.rankingFallbackCount} fallback.
        </div>
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
                <TrakeSequenceItem
                  key={item.key}
                  item={item}
                  index={index}
                  total={selection.length}
                  isPrimary={index === 0}
                  eventCount={eventCount}
                  onReorder={onReorderSelection}
                  onUpdate={onUpdateSelection}
                  onRemove={onRemoveSelection}
                />
              ))}
            </div>
            <TrakeSubmissionPreview plan={plan} />
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

function ManualTrakeSequenceForm({
  eventCount,
  onAddManualSelection,
}: {
  eventCount: number
  onAddManualSelection: (item: SelectedTrakeSequence) => void
}) {
  const [videoId, setVideoId] = useState('')
  const [keyframesText, setKeyframesText] = useState('')
  const [error, setError] = useState<string | null>(null)

  const parsedFrames = parseKeyframesInput(keyframesText)
  const eventHint =
    eventCount > 0
      ? `${eventCount} event frame${eventCount === 1 ? '' : 's'} required.`
      : 'Manual row sets the event count if retrieval is empty.'

  function addManualSequence(): void {
    const validation = validateManualTrakeSequence(videoId, parsedFrames, eventCount)
    if (validation) {
      setError(validation)
      return
    }
    onAddManualSelection(createManualTrakeSelection(videoId, parsedFrames.frames))
    setVideoId('')
    setKeyframesText('')
    setError(null)
  }

  function clearManualSequence(): void {
    setVideoId('')
    setKeyframesText('')
    setError(null)
  }

  return (
    <>
      <div className={styles.manualGrid}>
        <label className={styles.label}>
          Video ID
          <input
            className={styles.input}
            value={videoId}
            placeholder="L30_V046"
            onChange={(event) => setVideoId(event.target.value)}
          />
        </label>
        <label className={styles.label}>
          Keyframes
          <input
            className={styles.input}
            value={keyframesText}
            placeholder="894, 969, 1200"
            onChange={(event) => setKeyframesText(event.target.value)}
          />
        </label>
      </div>
      <div className={styles.hint}>
        {eventHint} Parsed: {parsedFrames.frames.length} frame(s).
      </div>
      {error && <div className={styles.warning}>{error}</div>}
      <div className={styles.actions}>
        <button type="button" className={styles.compactButton} onClick={addManualSequence}>
          Add Sequence
        </button>
        <button type="button" className={styles.compactButton} onClick={clearManualSequence}>
          Clear manual
        </button>
      </div>
    </>
  )
}

function TrakeSequenceItem({
  item,
  index,
  total,
  isPrimary,
  eventCount,
  onReorder,
  onUpdate,
  onRemove,
}: {
  item: SelectedTrakeSequence
  index: number
  total: number
  isPrimary: boolean
  eventCount: number
  onReorder: (fromIndex: number, toIndex: number) => void
  onUpdate: (key: string, patch: Partial<SelectedTrakeSequence>) => void
  onRemove: (key: string) => void
}) {
  const [dropPosition, setDropPosition] = useState<'before' | 'after' | null>(null)
  const incomplete = eventCount > 0 && item.result.keyframes.length !== eventCount

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
        <div className={styles.meta}>
          <strong>{item.result.video_id}</strong>
          <div className={styles.sequenceFrames}>
            {item.result.keyframes.map((keyframe) => keyframe.frame_idx).join(', ')}
          </div>
          <div className={styles.secondary}>
            {item.source === 'manual' ? 'Manual' : 'Retrieval'} sequence
            {item.originalRank >= 0 ? ` - Original #${item.originalRank + 1}` : ''}
          </div>
          {incomplete && (
            <div className={styles.warning}>
              Needs {eventCount} frame{eventCount === 1 ? '' : 's'} for this query.
            </div>
          )}
        </div>
        <div className={styles.trakeMiniStrip}>
          {item.result.keyframes.map((keyframe, keyframeIndex) => (
            <div key={`${keyframe.keyframe_id}:${keyframeIndex}`} className={styles.trakeMiniFrame}>
              {keyframe.thumbnail_url ? (
                <img src={keyframe.thumbnail_url} alt="" loading="lazy" />
              ) : (
                <div className={styles.thumbPlaceholder} aria-hidden="true">
                  E{keyframeIndex + 1}
                </div>
              )}
              <span>f{keyframe.frame_idx}</span>
            </div>
          ))}
        </div>
        {isPrimary ? (
          <TrakeMarginEditor item={item} onUpdate={onUpdate} />
        ) : (
          <div className={styles.hint}>Not used for CSV export - only sequence #1 is submitted.</div>
        )}
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

function TrakeMarginEditor({
  item,
  onUpdate,
}: {
  item: SelectedTrakeSequence
  onUpdate: (key: string, patch: Partial<SelectedTrakeSequence>) => void
}) {
  const step = Math.max(1, Math.trunc(item.step || 1))
  const margin = item.margin
  const deltas = buildTrakeUniformShiftDeltas({ step, margin })
  const forwardReach = Math.max(0, ...deltas.filter((delta) => delta > 0))
  const backwardReach = Math.min(0, ...deltas.filter((delta) => delta < 0))

  function setStep(value: string): void {
    onUpdate(item.key, { step: Math.max(1, Math.trunc(Number(value) || 1)) })
  }

  function setMargin(value: string): void {
    const parsed = Number(value)
    const nextMargin = value.trim() === '' || !Number.isFinite(parsed) ? null : Math.max(0, Math.trunc(parsed))
    onUpdate(item.key, { margin: nextMargin })
  }

  return (
    <div className={styles.trakeStepEditor}>
      <div className={styles.sectionHeader}>
        <span>Margin</span>
        <span>{deltas.length} row(s)</span>
      </div>
      <div className={styles.trakeStepControls}>
        <label className={styles.label}>
          Margin step (frames)
          <input
            className={styles.input}
            type="number"
            min={1}
            step={1}
            value={step}
            onChange={(event) => setStep(event.target.value)}
          />
        </label>
        <label className={styles.label}>
          Max margin (frames)
          <input
            className={styles.input}
            type="number"
            min={0}
            step={1}
            value={margin ?? ''}
            placeholder="fill to 100 rows"
            onChange={(event) => setMargin(event.target.value)}
          />
        </label>
      </div>
      <div className={styles.hint}>
        {deltas.length} row(s) - backward to {backwardReach}, forward to +{forwardReach}.
      </div>
    </div>
  )
}

function TrakeSubmissionPreview({
  plan,
}: {
  plan: ReturnType<typeof buildTrakeSubmissionPlan>
}) {
  const rows = plan.rows
  if (rows.length === 0) return null
  return (
    <div className={styles.preview}>
      <div className={styles.sectionHeader}>
        <span>Submission order</span>
        <span>{rows.length}</span>
      </div>
      {plan.entries.slice(0, 8).map((entry, index) => (
        <div className={styles.previewRow} key={`${entry.row.video_id}:${index}`}>
          <span>{index + 1}</span>
          <strong>{entry.row.video_id}</strong>
          <span>
            {entry.label}: {entry.row.keyframes.map((keyframe) => keyframe.frame_idx).join(' -> ')}
          </span>
          <em>{entry.row.keyframes.length}</em>
        </div>
      ))}
      <div className={styles.previewFooter}>
        {rows.length > 8 ? `Showing first 8 rows. ` : ''}
        {serializeTrakeCsv(rows).split('\n').length} CSV row(s) ready.
      </div>
    </div>
  )
}

function parseKeyframesInput(value: string): { frames: number[]; hasInvalidToken: boolean } {
  const tokens = value
    .split(/[,\s]+/)
    .map((token) => token.trim())
    .filter(Boolean)
  const frames: number[] = []
  let hasInvalidToken = false
  for (const token of tokens) {
    if (!/^\d+$/.test(token)) {
      hasInvalidToken = true
      continue
    }
    frames.push(Number(token))
  }
  return { frames, hasInvalidToken }
}

function validateManualTrakeSequence(
  videoId: string,
  parsedFrames: { frames: number[]; hasInvalidToken: boolean },
  eventCount: number,
): string | null {
  if (videoId.trim().length === 0) return 'Enter a video ID.'
  if (parsedFrames.hasInvalidToken) return 'Keyframes must be non-negative integers.'
  if (parsedFrames.frames.length === 0) return 'Enter at least one keyframe.'
  if (eventCount > 0 && parsedFrames.frames.length !== eventCount) {
    return `Enter exactly ${eventCount} keyframe${eventCount === 1 ? '' : 's'} for this TRAKE query.`
  }
  return null
}
