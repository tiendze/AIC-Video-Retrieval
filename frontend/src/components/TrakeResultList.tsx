import { useMemo, useState } from 'react'
import type { TrakeVideoResultItem } from '../api/types'
import { trakeSequenceKey, trakeVisualSelectionKey } from '../utils/submissionExport'
import { formatTrakeTimestamp } from '../utils/trakeFormat'
import styles from './TrakeResultList.module.css'

export interface TrakeResultListProps {
  results: TrakeVideoResultItem[]
  // Per-video top-k sequences (before the submission cap), keyed by video_id.
  // The "Details" button expands a video's card into these.
  sequencesByVideo?: Record<string, TrakeVideoResultItem[]>
  selectedSequenceKeys?: Set<string>
  onToggleSequence?: (
    result: TrakeVideoResultItem,
    originalRank: number,
    selectionKey: string,
  ) => void
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

// DANTE/TRAKE results UI (plan §P2.2, PRD §F6/US-4 + F9(C)). Each result
// is one ordered keyframe sequence -- rendered as its own strip, never
// flattened into the regular per-keyframe grid, and never re-sorted
// (CLAUDE.md §3, DANTE's τ < t invariant). Multiple top-k sequences from
// the same video are grouped after ranking, preserving backend result order.
export function TrakeResultList({
  results,
  sequencesByVideo,
  selectedSequenceKeys,
  onToggleSequence,
  onOpenDetail,
}: TrakeResultListProps) {
  const [expandedVideos, setExpandedVideos] = useState<Set<string>>(() => new Set())

  const rankByKey = useMemo(() => {
    const map = new Map<string, number[]>()
    results.forEach((result, index) => {
      const key = trakeSequenceKey(result)
      const ranks = map.get(key)
      if (ranks) ranks.push(index)
      else map.set(key, [index])
    })
    return map
  }, [results])

  if (results.length === 0) {
    return <div className={styles.empty}>No results yet -- try a search.</div>
  }

  const groups = groupByVideo(results)

  function toggleExpanded(videoId: string): void {
    setExpandedVideos((prev) => {
      const next = new Set(prev)
      if (next.has(videoId)) next.delete(videoId)
      else next.add(videoId)
      return next
    })
  }

  return (
    <div className={styles.list}>
      {groups.map((group) => {
        const expanded = expandedVideos.has(group.videoId)
        const detailSequences = sequencesByVideo?.[group.videoId] ?? []
        const hasMore = detailSequences.length > group.sequences.length
        // Expanded view shows the video's full top-k; collapsed shows only the
        // sequence(s) that made the submission list. Detail rows reuse their
        // rank from `results` when present, else -1 (display only).
        const rendered: RankedTrakeResult[] = (() => {
          if (!expanded || detailSequences.length === 0) return group.sequences
          const seenBySequenceKey = new Map<string, number>()
          return detailSequences.map((result) => {
            const key = trakeSequenceKey(result)
            const occurrence = seenBySequenceKey.get(key) ?? 0
            seenBySequenceKey.set(key, occurrence + 1)
            return {
              result,
              originalRank: rankByKey.get(key)?.[occurrence] ?? -1,
            }
          })
        })()
        return (
          <div key={group.videoId} className={styles.card}>
            <div className={styles.header}>
              <span className={styles.videoId}>{group.videoId}</span>
              <div className={styles.headerActions}>
                {hasMore && (
                  <button
                    type="button"
                    className={styles.detailsButton}
                    onClick={() => toggleExpanded(group.videoId)}
                    aria-expanded={expanded}
                  >
                    {expanded ? 'Hide' : `Details (${detailSequences.length})`}
                  </button>
                )}
                <span className={styles.sequenceCount}>
                  {rendered.length} sequence{rendered.length === 1 ? '' : 's'}
                </span>
              </div>
            </div>
            <div className={styles.sequences}>
              {rendered.map(({ result, originalRank }, index) => {
              const sequenceKey = trakeVisualSelectionKey(result, originalRank, `${group.videoId}:${index}`)
              const renderKey = sequenceKey
              const sequenceSelected = selectedSequenceKeys?.has(sequenceKey) ?? false
              return (
                <div
                  key={renderKey}
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
                          onClick={() => onToggleSequence(result, originalRank, sequenceKey)}
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
                        {(keyframe.semantic_z != null ||
                          keyframe.local_score != null ||
                          keyframe.kis_rank != null) && (
                          <span
                            className={styles.diagnostics}
                            title="z: điểm ngữ nghĩa (semantic_z) · local: khớp phrase-to-patch (local_z, chỉ hiện khi video này được B2b rerank) · kis: hạng KIS trả về cho frame này"
                          >
                            {keyframe.semantic_z != null && `z ${keyframe.semantic_z.toFixed(2)}`}
                            {/* local_z is always a number from the backend (0.0 default) --
                                gate the display on local_score, which is null unless B2b
                                actually reranked this frame's video. */}
                            {keyframe.local_score != null &&
                              keyframe.local_z != null &&
                              ` · local ${keyframe.local_z.toFixed(2)}`}
                            {keyframe.kis_rank != null && ` · kis #${keyframe.kis_rank + 1}`}
                          </span>
                        )}
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
        )
      })}
    </div>
  )
}
