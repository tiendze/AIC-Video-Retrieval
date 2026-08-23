import type { SearchHit } from '../api/types'
import styles from './ResultCard.module.css'

export interface ResultCardProps {
  hit: SearchHit
  selected?: boolean
  manuallySelected?: boolean
  manualSelectionLabel?: string
  onSelect: (hit: SearchHit) => void
  onFindSimilar: (hit: SearchHit) => void
  onToggleManualSelection?: (hit: SearchHit) => void
  onOpenDetail?: (hit: SearchHit) => void
}

// PRD §F9 / plan §P1.6: video_id + timestamp must be visible on the
// thumbnail itself, without hovering -- an operator under contest time
// pressure should never have to mouse over 100 tiles to read one.
export function ResultCard({
  hit,
  selected = false,
  manuallySelected = false,
  manualSelectionLabel = 'Select',
  onSelect,
  onFindSimilar,
  onToggleManualSelection,
  onOpenDetail,
}: ResultCardProps) {
  const score = hit.final_score ?? hit.score

  return (
    <button
      type="button"
      className={`${styles.card} ${selected ? styles.selected : ''} ${
        manuallySelected ? styles.manualSelectedCard : ''
      }`}
      aria-label={`${hit.video_id} at ${hit.timestamp}s`}
      onClick={() => onSelect(hit)}
    >
      <img
        className={styles.thumbnail}
        src={hit.thumbnail_url}
        alt={`${hit.video_id} @ ${hit.timestamp}s`}
        loading="lazy"
      />
      <span className={styles.label}>
        <span className={styles.videoId}>{hit.video_id}</span>
        <span>{hit.timestamp}s</span>
      </span>
      {hit.asr_evidence && (
        <span className={styles.asrBadge}>ASR #{hit.asr_evidence.video_rank}</span>
      )}
      {hit.final_score !== undefined && (
        <span className={styles.scoreBadge}>Final {score.toFixed(3)}</span>
      )}
      {onOpenDetail && (
        <span
          role="button"
          tabIndex={0}
          className={styles.openDetail}
          onClick={(event) => {
            event.stopPropagation()
            onOpenDetail(hit)
          }}
          onKeyDown={(event) => {
            if (event.key === 'Enter' || event.key === ' ') {
              event.stopPropagation()
              onOpenDetail(hit)
            }
          }}
        >
          Details
        </span>
      )}
      {onToggleManualSelection && (
        <span
          role="button"
          tabIndex={0}
          className={`${styles.manualSelect} ${
            manuallySelected ? styles.manualSelectActive : ''
          }`}
          onClick={(event) => {
            event.stopPropagation()
            onToggleManualSelection(hit)
          }}
          onKeyDown={(event) => {
            if (event.key === 'Enter' || event.key === ' ') {
              event.stopPropagation()
              onToggleManualSelection(hit)
            }
          }}
        >
          {manuallySelected ? '✓ Selected' : manualSelectionLabel}
        </span>
      )}
      <span
        role="button"
        tabIndex={0}
        className={styles.findSimilar}
        onClick={(event) => {
          event.stopPropagation()
          onFindSimilar(hit)
        }}
        onKeyDown={(event) => {
          if (event.key === 'Enter' || event.key === ' ') {
            event.stopPropagation()
            onFindSimilar(hit)
          }
        }}
      >
        Find Similar
      </span>
    </button>
  )
}
