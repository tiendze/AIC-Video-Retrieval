import type { SearchHit } from '../api/types'
import { buildVideoGroups, type VideoGroupSortMode } from '../utils/retrievalView'
import { ResultCard } from './ResultCard'
import styles from './GroupedResultList.module.css'

export interface GroupedResultListProps {
  hits: SearchHit[]
  sortMode: VideoGroupSortMode
  rrfK: number
  selectedKeyframeId?: string | null
  manuallySelectedKeys?: Set<string>
  manualSelectionLabel?: string
  onSelect: (hit: SearchHit) => void
  onFindSimilar: (hit: SearchHit) => void
  onToggleManualSelection?: (hit: SearchHit) => void
  onOpenDetail?: (hit: SearchHit) => void
}

export function GroupedResultList({
  hits,
  sortMode,
  rrfK,
  selectedKeyframeId = null,
  manuallySelectedKeys,
  manualSelectionLabel,
  onSelect,
  onFindSimilar,
  onToggleManualSelection,
  onOpenDetail,
}: GroupedResultListProps) {
  const groups = buildVideoGroups(hits, sortMode, rrfK)
  if (groups.length === 0) {
    return <div className={styles.empty}>No results yet -- try a search.</div>
  }

  return (
    <div className={styles.groups}>
      {groups.map((group) => (
        <section className={styles.group} key={group.videoId}>
          <header className={styles.header}>
            <strong>{group.videoId}</strong>
            <span>Best rank #{group.bestRank}</span>
            <span>{group.matchCount} matches</span>
            {sortMode === 'rrf' && <span>RRF {group.rrfScore.toFixed(4)}</span>}
          </header>
          <div className={styles.frames}>
            {group.frames.map(({ hit, rank }) => (
              <div className={styles.cell} key={hit.keyframe_id}>
                <ResultCard
                  hit={hit}
                  rank={rank}
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
