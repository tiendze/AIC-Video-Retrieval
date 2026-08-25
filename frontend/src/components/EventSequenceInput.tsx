import { QueryInput } from './QueryInput'
import styles from './EventSequenceInput.module.css'
import searchStyles from './SearchBar.module.css'

export interface EventSequenceInputProps {
  events: string[]
  onEventChange: (index: number, value: string) => void
  onAddEvent: () => void
  onRemoveEvent: (index: number) => void
  threshold: string
  onThresholdChange: (value: string) => void
  onSubmit: () => void
  isLoading?: boolean
}

// DANTE/TRAKE input UI (plan §P2.2, PRD §F6/US-4): "tôi nhập N mô tả sự
// kiện theo thứ tự" -- each row is its own labeled textbox in fixed
// position order, never reorderable by drag (order = array index, matches
// what POST /api/trake and its response both key off of).
export function EventSequenceInput({
  events,
  onEventChange,
  onAddEvent,
  onRemoveEvent,
  threshold,
  onThresholdChange,
  onSubmit,
  isLoading = false,
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
        <div className={searchStyles.queryList}>
          {events.map((value, index) => (
            <div className={searchStyles.queryRow} key={index}>
              <span className={`${searchStyles.typeSelect} ${styles.eventLabel}`}>
                Event {index + 1}
              </span>
              <QueryInput
                aria-label={`event ${index + 1}`}
                type="text"
                placeholder="Describe event..."
                value={value}
                onValueChange={(nextValue) => onEventChange(index, nextValue)}
              />
              {events.length > 1 && (
                <button
                  type="button"
                  className={`${searchStyles.remove} ${searchStyles.removeQuery}`}
                  aria-label={`remove event ${index + 1}`}
                  title="Remove event"
                  onClick={() => onRemoveEvent(index)}
                >
                  x
                </button>
              )}
            </div>
          ))}
        </div>
        <div className={searchStyles.actions}>
          <label className={searchStyles.fieldCompact}>
            <span>Threshold</span>
            <input
              aria-label="threshold"
              className={searchStyles.numberInput}
              type="number"
              step="0.01"
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
