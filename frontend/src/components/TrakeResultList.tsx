import type { TrakeVideoResultItem } from '../api/types'
import styles from './TrakeResultList.module.css'

export interface TrakeResultListProps {
  results: TrakeVideoResultItem[]
}

interface TrakeVideoGroup {
  videoId: string
  sequences: TrakeVideoResultItem[]
}

function groupByVideo(results: TrakeVideoResultItem[]): TrakeVideoGroup[] {
  const byVideo = new Map<string, TrakeVideoResultItem[]>()
  for (const result of results) {
    const existing = byVideo.get(result.video_id)
    if (existing) {
      existing.push(result)
    } else {
      byVideo.set(result.video_id, [result])
    }
  }
  return Array.from(byVideo, ([videoId, sequences]) => ({ videoId, sequences }))
}

// DANTE/TRAKE results UI (plan §P2.2, PRD §F6/US-4 + F9(C)). Each result
// is one ordered keyframe sequence -- rendered as its own strip, never
// flattened into the regular per-keyframe grid, and never re-sorted
// (CLAUDE.md §3, DANTE's τ < t invariant). Multiple top-k sequences from
// the same video are grouped after ranking, preserving backend result order.
export function TrakeResultList({ results }: TrakeResultListProps) {
  if (results.length === 0) {
    return <div className={styles.empty}>No results yet -- try a search.</div>
  }

  const groups = groupByVideo(results)

  return (
    <div className={styles.list}>
      {groups.map((group) => (
        <div key={group.videoId} className={styles.card}>
          <div className={styles.header}>
            <span className={styles.videoId}>{group.videoId}</span>
            <span className={styles.sequenceCount}>
              {group.sequences.length} sequence{group.sequences.length === 1 ? '' : 's'}
            </span>
          </div>
          <div className={styles.sequences}>
            {group.sequences.map((result, index) => (
              <div key={result.keyframes.map((keyframe) => keyframe.keyframe_id).join('-')}>
                {index > 0 && <div className={styles.separator} role="separator" />}
                <div className={styles.sequenceHeader}>
                  <span className={styles.sequenceLabel}>Sequence {index + 1}</span>
                  <span className={styles.score}>Score {result.score.toFixed(4)}</span>
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
        </div>
      ))}
    </div>
  )
}
