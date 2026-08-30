import { useMemo, useRef, type RefObject } from 'react'
import { useVirtualizer } from '@tanstack/react-virtual'
import type { SearchHit } from '../api/types'
import { rankedHits } from '../utils/retrievalView'
import { ResultCard } from './ResultCard'
import styles from './ResultGrid.module.css'

export interface ResultGridProps {
  hits: SearchHit[]
  rankSourceHits?: SearchHit[]
  selectedKeyframeId?: string | null
  manuallySelectedKeys?: Set<string>
  manualSelectionLabel?: string
  scrollElementRef?: RefObject<HTMLDivElement | null>
  onSelect: (hit: SearchHit) => void
  onFindSimilar: (hit: SearchHit) => void
  onToggleManualSelection?: (hit: SearchHit) => void
  onOpenDetail?: (hit: SearchHit) => void
}

const MIN_CARD_WIDTH_PX = 180
const CARD_HEIGHT_PX = 160
const GAP_PX = 4
const MAX_DESKTOP_COLUMNS = 5

// plan §P1.6: "Virtualized rendering cho lưới hàng nghìn thumbnail" --
// max_top_k=100 caps a *live* grid (AIC submission rule, references.md
// §1.2), but the grid component itself must not assume that cap (plan
// §P1.6 finding 2: docs/code contradiction resolved by keeping max_top_k
// and verifying virtualization against a synthetic 1,000-item fixture
// instead). A row virtualizer (not per-card) keeps the DOM bounded to
// roughly one viewport's worth of rows regardless of hit count.
export function ResultGrid({
  hits,
  rankSourceHits = hits,
  selectedKeyframeId = null,
  manuallySelectedKeys,
  manualSelectionLabel,
  scrollElementRef,
  onSelect,
  onFindSimilar,
  onToggleManualSelection,
  onOpenDetail,
}: ResultGridProps) {
  const scrollRef = useRef<HTMLDivElement>(null)
  const usesExternalScroll = scrollElementRef !== undefined
  const scrollMargin = usesExternalScroll ? (scrollRef.current?.offsetTop ?? 0) : 0
  const ranked = useMemo(() => rankedHits(hits, rankSourceHits), [hits, rankSourceHits])

  const containerWidth = scrollRef.current?.clientWidth ?? MIN_CARD_WIDTH_PX
  const columns = Math.min(
    MAX_DESKTOP_COLUMNS,
    Math.max(1, Math.floor((containerWidth + GAP_PX) / (MIN_CARD_WIDTH_PX + GAP_PX))),
  )
  const rowCount = Math.ceil(hits.length / columns)
  const cellWidth = `calc((100% - ${GAP_PX * (columns - 1)}px) / ${columns})`

  const rowVirtualizer = useVirtualizer({
    count: rowCount,
    getScrollElement: () => scrollElementRef?.current ?? scrollRef.current,
    estimateSize: () => CARD_HEIGHT_PX + GAP_PX,
    scrollMargin,
    overscan: 3,
  })

  if (hits.length === 0) {
    return <div className={styles.empty}>No results yet -- try a search.</div>
  }

  return (
    <div
      ref={scrollRef}
      data-testid="result-grid-scroll-container"
      className={`${styles.scrollContainer} ${usesExternalScroll ? styles.externalScrollContainer : ''}`}
    >
      <div
        style={{ height: rowVirtualizer.getTotalSize(), position: 'relative', width: '100%' }}
      >
        {rowVirtualizer.getVirtualItems().map((virtualRow) => {
          const rowStart = virtualRow.index * columns
          const rowHits = ranked.slice(rowStart, rowStart + columns)
          return (
            <div
              key={virtualRow.key}
              data-testid="result-grid-row"
              className={styles.row}
              style={{
                position: 'absolute',
                top: 0,
                left: 0,
                width: '100%',
                height: virtualRow.size,
                transform: `translateY(${virtualRow.start - scrollMargin}px)`,
              }}
            >
              {rowHits.map(({ hit, rank, localRank, globalRank }, columnIndex) => (
                <div
                  key={hit.keyframe_id}
                  className={styles.cell}
                  style={{ width: cellWidth, height: CARD_HEIGHT_PX }}
                >
                  <ResultCard
                    hit={hit}
                    rank={rank ?? rowStart + columnIndex + 1}
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
          )
        })}
      </div>
    </div>
  )
}
