import {
  TRAKE_DEFAULT_EVENT_TEXT_TOP_K,
  TRAKE_MAX_EVENT_TEXT_TOP_K,
  TRAKE_MIN_EVENT_TEXT_TOP_K,
} from '../api/constants'
import type { TrakeEventDiagnostics } from '../api/types'
import type { TrakeEventDraft, TrakeEventField } from '../hooks/useTrake'
import { QueryInput } from './QueryInput'
import styles from './EventSequenceInput.module.css'
import searchStyles from './SearchBar.module.css'

export interface EventSequenceInputProps {
  events: TrakeEventDraft[]
  onEventFieldChange: (index: number, field: TrakeEventField, value: string) => void
  onAddEvent: () => void
  onRemoveEvent: (index: number) => void
  threshold: string
  onThresholdChange: (value: string) => void
  onSubmit: () => void
  isLoading?: boolean
  asrDisabled?: boolean
  asrReadinessMessage?: string | null
  eventDiagnostics?: TrakeEventDiagnostics[]
  candidateVideoCount?: number | null
  droppedVideoCount?: number | null
}

interface ChannelSpec {
  field: TrakeEventField
  label: string
  placeholder: string
  hint: string
}

// The four channels are per-event because TRAKE runs one KIS Stage 1 call per
// event (app/services/trake_service.py). Text drives the semantic axis; OCR
// pins a moment inside the video; Title and ASR gate WHICH videos are
// candidates for that event.
const CHANNELS: ChannelSpec[] = [
  {
    field: 'text',
    label: 'Text',
    placeholder: 'Mô tả khoảnh khắc...',
    hint: 'Kênh ngữ nghĩa. Bỏ trống được, nhưng khi đó event này chỉ dựa vào OCR/ASR.',
  },
  {
    field: 'ocr',
    label: 'OCR',
    placeholder: 'Chữ xuất hiện trên hình...',
    hint: 'BM25 trên chữ trong khung hình — dùng để khoá đúng khoảnh khắc.',
  },
  {
    field: 'title',
    label: 'Title',
    placeholder: 'Từ khoá trong tiêu đề video...',
    hint: 'Lọc ở mức video theo tiêu đề.',
  },
  {
    field: 'asr',
    label: 'ASR',
    placeholder: 'Lời nói trong video...',
    hint: 'Cổng lọc video theo lời thoại.',
  },
]

// Text is the semantic axis and carries the longest queries, so it gets its own
// full-width row; OCR / Title / ASR are short pins/gates and share one 3-column
// row below it. The order still IS the array index -- this only changes layout.
const TEXT_CHANNEL: ChannelSpec = CHANNELS[0]
const SECONDARY_CHANNELS: ChannelSpec[] = CHANNELS.slice(1)

// TRAKE input UI (PRD §F6/US-4): "tôi nhập N mô tả sự kiện theo thứ tự". Each
// event is its own fixed-position block, never reorderable by drag -- the order
// IS the array index, which is what POST /api/trake and its response both key
// off of.
export function EventSequenceInput({
  events,
  onEventFieldChange,
  onAddEvent,
  onRemoveEvent,
  threshold,
  onThresholdChange,
  onSubmit,
  isLoading = false,
  asrDisabled = false,
  asrReadinessMessage = null,
  eventDiagnostics = [],
  candidateVideoCount = null,
  droppedVideoCount = null,
}: EventSequenceInputProps) {
  return (
    <form
      className={searchStyles.flow}
      onSubmit={(event) => {
        event.preventDefault()
        onSubmit()
      }}
    >
      <section className={searchStyles.stage}>
        <div className={searchStyles.stageHeader}>
          <span>Ordered Events</span>
          <strong>{events.length} event rows</strong>
        </div>
        <div className={styles.eventList}>
          {events.map((event, index) => {
            const diagnostics = eventDiagnostics.find((item) => item.event_index === index)
            return (
              <fieldset className={styles.eventBlock} key={index}>
                <legend className={styles.eventLegend}>
                  <span className={styles.eventLabel}>Event {index + 1}</span>
                  {events.length > 1 && (
                    <button
                      type="button"
                      className={styles.eventRemoveButton}
                      aria-label={`remove event ${index + 1}`}
                      title="Remove event"
                      onClick={() => onRemoveEvent(index)}
                    >
                      x
                    </button>
                  )}
                </legend>
                <div className={styles.textRow}>
                  <span className={styles.channelLabel}>{TEXT_CHANNEL.label}</span>
                  <QueryInput
                    aria-label={`event ${index + 1} ${TEXT_CHANNEL.label.toLowerCase()}`}
                    title={TEXT_CHANNEL.hint}
                    type="text"
                    placeholder={TEXT_CHANNEL.placeholder}
                    value={event[TEXT_CHANNEL.field]}
                    onValueChange={(nextValue) =>
                      onEventFieldChange(index, TEXT_CHANNEL.field, nextValue)
                    }
                  />
                  <label
                    className={styles.topKField}
                    title="Số keyframe tối đa lấy về cho kênh Text của event này (mặc định 1000)."
                  >
                    <span className={styles.topKLabel}>K</span>
                    <input
                      aria-label={`event ${index + 1} text top k`}
                      className={styles.topKInput}
                      type="number"
                      inputMode="numeric"
                      min={TRAKE_MIN_EVENT_TEXT_TOP_K}
                      max={TRAKE_MAX_EVENT_TEXT_TOP_K}
                      placeholder={String(TRAKE_DEFAULT_EVENT_TEXT_TOP_K)}
                      value={event.textTopK}
                      onChange={(changeEvent) =>
                        onEventFieldChange(index, 'textTopK', changeEvent.target.value)
                      }
                    />
                  </label>
                </div>
                <div className={styles.secondaryRow}>
                  {SECONDARY_CHANNELS.map((channel) => {
                    const disabled = channel.field === 'asr' && asrDisabled
                    return (
                      <label className={styles.secondaryField} key={channel.field}>
                        <span className={styles.secondaryLabel}>{channel.label}</span>
                        <QueryInput
                          aria-label={`event ${index + 1} ${channel.label.toLowerCase()}`}
                          title={channel.hint}
                          type="text"
                          placeholder={channel.placeholder}
                          value={event[channel.field]}
                          disabled={disabled}
                          onValueChange={(nextValue) =>
                            onEventFieldChange(index, channel.field, nextValue)
                          }
                        />
                      </label>
                    )
                  })}
                </div>
                {diagnostics && (
                  <div className={styles.eventStats}>
                    {diagnostics.kis_candidate_count} keyframes /{' '}
                    {diagnostics.kis_video_count} videos
                  </div>
                )}
              </fieldset>
            )
          })}
        </div>
        {asrDisabled && asrReadinessMessage && (
          <p className={searchStyles.asrReadiness}>{asrReadinessMessage}</p>
        )}
        {candidateVideoCount !== null && (
          <p className={searchStyles.poolBreakdown}>
            {candidateVideoCount} video ứng viên
            {droppedVideoCount !== null && ` · ${droppedVideoCount} bị loại (không nối được chuỗi)`}
          </p>
        )}
        <div className={searchStyles.actions}>
          <label className={searchStyles.fieldCompact}>
            <span>Threshold</span>
            <input
              aria-label="threshold"
              className={searchStyles.numberInput}
              type="number"
              step="0.1"
              value={threshold}
              onChange={(event) => onThresholdChange(event.target.value)}
            />
          </label>
          <button type="button" className={searchStyles.addButton} onClick={onAddEvent}>
            + Add event
          </button>
          <button type="submit" className={searchStyles.submit} disabled={isLoading}>
            Search TRAKE{isLoading ? '...' : ''}
          </button>
        </div>
      </section>
    </form>
  )
}
