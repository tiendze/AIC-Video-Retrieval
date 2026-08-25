import type { TrakeVideoResultItem } from '../api/types'
import { trakeSequenceKey } from '../utils/submissionExport'
import styles from './TrakeResultList.module.css'

export interface TrakeResultListProps {
  results: TrakeVideoResultItem[]
  selectedSequenceKeys?: Set<string>
  onToggleSequence?: (result: TrakeVideoResultItem, originalRank: number) => void
  onOpenDetail?: (result: TrakeVideoResultItem, keyframeIndex: number) => void
}

interface RankedTrakeResult {
  result: TrakeVideoResultItem
  originalRank: number
}

interface TrakeVideoGroup {
  videoId: string
  sequences: RankedTrakeResult[]
}

function groupByVideo(results: TrakeVideoResultItem[]): TrakeVideoGroup[] {
  const byVideo = new Map<string, RankedTrakeResult[]>()
  results.forEach((result, originalRank) => {
    const rankedResult = { result, originalRank }
    const existing = byVideo.get(result.video_id)
    if (existing) {
      existing.push(rankedResult)
    } else {
      byVideo.set(result.video_id, [rankedResult])
    }
  })
  return Array.from(byVideo, ([videoId, sequences]) => ({ videoId, sequences }))
}

export function formatTrakeTimestamp(seconds: number): string {
  const safeSeconds = Number.isFinite(seconds) ? Math.max(0, seconds) : 0
  const hours = Math.floor(safeSeconds / 3600)
  const minutes = Math.floor((safeSeconds % 3600) / 60)
  const remainingSeconds = safeSeconds - hours * 3600 - minutes * 60
  const secondsText = remainingSeconds.toFixed(1).padStart(4, '0')
  if (hours > 0) {
    return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${secondsText}`
  }
  return `${String(minutes).padStart(2, '0')}:${secondsText}`
}

// DANTE/TRAKE results UI (plan §P2.2, PRD §F6/US-4 + F9(C)). Each result
// is one ordered keyframe sequence -- rendered as its own strip, never
// flattened into the regular per-keyframe grid, and never re-sorted
// (CLAUDE.md §3, DANTE's τ < t invariant). Multiple top-k sequences from
// the same video are grouped after ranking, preserving backend result order.
export function TrakeResultList({
  results,
  selectedSequenceKeys,
  onToggleSequence,
  onOpenDetail,
}: TrakeResultListProps) {
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
            {group.sequences.map(({ result, originalRank }, index) => {
              const sequenceKey = trakeSequenceKey(result)
              const sequenceSelected = selectedSequenceKeys?.has(sequenceKey) ?? false
              return (
                <div
                  key={sequenceKey}
                  className={sequenceSelected ? styles.sequenceSelected : ''}
                >
                  {index > 0 && <div className={styles.separator} role="separator" />}
                  <div className={styles.sequenceHeader}>
                    <span className={styles.sequenceLabel}>Sequence {index + 1}</span>
                    <div className={styles.sequenceActions}>
                      {onToggleSequence && (
                        <button
                          type="button"
                          className={`${styles.selectSequence} ${
                            sequenceSelected ? styles.selectSequenceActive : ''
                          }`}
                          onClick={() => onToggleSequence(result, originalRank)}
                        >
                          {sequenceSelected ? '✓ Selected' : 'Select'}
                        </button>
                      )}
                      <span className={styles.score}>Score {result.score.toFixed(4)}</span>
                    </div>
                  </div>
                  <div className={styles.strip}>
                    {result.keyframes.map((keyframe, keyframeIndex) => (
                      <button
                        type="button"
                        key={keyframe.keyframe_id}
                        className={styles.keyframe}
                        onClick={() => onOpenDetail?.(result, keyframeIndex)}
                        aria-label={`Open event ${keyframeIndex + 1} frame ${keyframe.frame_idx}`}
                      >
                        <img
                          src={keyframe.thumbnail_url}
                          alt={`event ${keyframeIndex + 1} at ${formatTrakeTimestamp(
                            keyframe.timestamp,
                          )}`}
                          loading="lazy"
                        />
                        <span className={styles.timestamp}>
                          Event {keyframeIndex + 1} - {formatTrakeTimestamp(keyframe.timestamp)}
                        </span>
                        <span className={styles.frameIdx}>frame {keyframe.frame_idx}</span>
                      </button>
                    ))}
                  </div>
                  <div className={styles.submission}>
                    {result.video_id}, {result.keyframes.map((k) => k.frame_idx).join(', ')}
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      ))}
    </div>
  )
}
