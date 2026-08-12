import type { SearchHit } from '../api/types'
import styles from './ResultCard.module.css'

export interface ResultCardProps {
  hit: SearchHit
  onSelect: (hit: SearchHit) => void
  onFindSimilar: (hit: SearchHit) => void
}

// PRD §F9 / plan §P1.6: video_id + timestamp must be visible on the
// thumbnail itself, without hovering -- an operator under contest time
// pressure should never have to mouse over 100 tiles to read one.
export function ResultCard({ hit, onSelect, onFindSimilar }: ResultCardProps) {
  return (
    <button
      type="button"
      className={styles.card}
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
