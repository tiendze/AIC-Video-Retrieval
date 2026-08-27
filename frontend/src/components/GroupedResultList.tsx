import type { SearchHit } from '../api/types'
import { buildVideoGroups, type VideoGroupSortMode } from '../utils/retrievalView'
import { ResultCard } from './ResultCard'
import styles from './GroupedResultList.module.css'

export interface GroupedResultListProps {
  hits: SearchHit[]
  rankSourceHits?: SearchHit[]
  sortMode: VideoGroupSortMode
  rrfK: number
  selectedKeyframeId?: string | null
  manuallySelectedKeys?: Set<string>
  manualSelectionLabel?: string
  onSelect: (hit: SearchHit) => void
  onFindSimilar: (hit: SearchHit) => void
  onToggleManualSelection?: (hit: SearchHit) => void
  onOpenDetail?: (hit: SearchHit) => void
  onExcludeVideo?: (videoId: string) => void
}

export function GroupedResultList({
  hits,
  rankSourceHits = hits,
  sortMode,
  rrfK,
  selectedKeyframeId = null,
  manuallySelectedKeys,
  manualSelectionLabel,
  onSelect,
  onFindSimilar,
  onToggleManualSelection,
  onOpenDetail,
  onExcludeVideo,
}: GroupedResultListProps) {
  const groups = buildVideoGroups(hits, sortMode, rrfK, rankSourceHits)
  if (groups.length === 0) {
    return <div className={styles.empty}>No results yet -- try a search.</div>
  }

  return (
    <div className={styles.groups}>
      {groups.map((group) => (
        <section className={styles.group} key={group.videoId}>
          <header className={styles.header}>
            <strong>#{group.finalVideoRank} · {group.videoId}</strong>
            {group.ocrDetected && <span className={styles.badge}>OCR</span>}
            {group.titleOcrDetected && <span className={styles.badge}>TITLE OCR</span>}
            {group.bestLocalRank !== null && (
              <span>
                Best local #{group.bestLocalRank}
                {group.bestLocalScore !== null ? ` · ${group.bestLocalScore.toFixed(3)}` : ''}
              </span>
            )}
            <span>
              Best global {group.bestGlobalRank !== null ? `#${group.bestGlobalRank}` : '-'}
            </span>
            <span>{group.matchCount} matches</span>
            {sortMode === 'rrf' && <span>RRF {group.rrfScore.toFixed(4)}</span>}
            {onExcludeVideo && (
              <button
                type="button"
                className={styles.exclude}
                onClick={() => onExcludeVideo(group.videoId)}
              >
                Exclude
              </button>
            )}
          </header>
          <div className={styles.frames}>
            {group.frames.map(({ hit, rank, localRank, globalRank }) => (
              <div className={styles.cell} key={hit.keyframe_id}>
                <ResultCard
                  hit={hit}
                  rank={rank}
                  localRank={localRank}
                  globalRank={globalRank}
                  selected={hit.keyframe_id === selectedKeyframeId}
                  manuallySelected={manuallySelectedKeys?.has(`${hit.video_id}:${hit.frame_idx}`)}
                  manualSelectionLabel={manualSelectionLabel}
                  onSelect={onSelect}
                  onFindSimilar={onFindSimilar}
                  onToggleManualSelection={onToggleManualSelection}
                  onOpenDetail={onOpenDetail}
                />
              </div>
            ))}
          </div>
        </section>
      ))}
    </div>
  )
}
