import styles from './EventSequenceInput.module.css'

export interface EventSequenceInputProps {
  events: string[]
  onEventChange: (index: number, value: string) => void
  onAddEvent: () => void
  onRemoveEvent: (index: number) => void
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
  onSubmit,
  isLoading = false,
}: EventSequenceInputProps) {
  return (
    <form
      className={styles.form}
      onSubmit={(event) => {
        event.preventDefault()
        onSubmit()
      }}
    >
      {events.map((value, index) => (
        <div className={styles.row} key={index}>
          <span className={styles.index}>{index + 1}</span>
          <input
            aria-label={`event ${index + 1}`}
            className={styles.input}
            type="text"
            placeholder="Mô tả sự kiện (tiếng Việt)…"
            value={value}
            onChange={(event) => onEventChange(index, event.target.value)}
          />
          {events.length > 1 && (
            <button
              type="button"
              className={styles.remove}
              aria-label={`xoá sự kiện ${index + 1}`}
              title="Xoá sự kiện này"
              onClick={() => onRemoveEvent(index)}
            >
              ✕
            </button>
          )}
        </div>
      ))}
      <div className={styles.actions}>
        <button type="button" className={styles.addButton} onClick={onAddEvent}>
          + Thêm sự kiện
        </button>
        <button type="submit" className={styles.submit} disabled={isLoading}>
          Tìm chuỗi{isLoading ? '…' : ''}
        </button>
      </div>
    </form>
  )
}
