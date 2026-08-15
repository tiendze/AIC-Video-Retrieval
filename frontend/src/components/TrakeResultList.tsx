import type { TrakeVideoResultItem } from '../api/types'
import styles from './TrakeResultList.module.css'

export interface TrakeResultListProps {
  results: TrakeVideoResultItem[]
}

// DANTE/TRAKE results UI (plan §P2.2, PRD §F6/US-4 + F9(C)). Each result
// is one video's *ordered* keyframe sequence -- rendered as its own strip,
// never flattened into the regular per-keyframe grid, and never re-sorted
// (CLAUDE.md §3, DANTE's τ < t invariant). The trailing line spells out
// the literal AIC submission shape for TRAKE, `<video_id>, <frame_id₁>,
// …, <frame_idₙ>` (references.md §1.2), so the operator can visually
// verify order/content before it ever reaches an export step.
export function TrakeResultList({ results }: TrakeResultListProps) {
  if (results.length === 0) {
    return <div className={styles.empty}>No results yet -- try a search.</div>
  }

  return (
    <div className={styles.list}>
      {results.map((result) => (
        <div key={result.video_id} className={styles.card}>
          <div className={styles.header}>
            <span className={styles.videoId}>{result.video_id}</span>
            <span className={styles.score}>score {result.score.toFixed(3)}</span>
          </div>
          <div className={styles.strip}>
            {result.keyframes.map((keyframe) => (
              <div key={keyframe.keyframe_id} className={styles.keyframe}>
                <img
                  src={keyframe.thumbnail_url}
                  alt={`frame ${keyframe.frame_idx}`}
                  loading="lazy"
                />
                <span className={styles.frameIdx}>{keyframe.frame_idx}</span>
              </div>
            ))}
          </div>
          <div className={styles.submission}>
            {result.video_id}, {result.keyframes.map((k) => k.frame_idx).join(', ')}
          </div>
        </div>
      ))}
    </div>
  )
}
