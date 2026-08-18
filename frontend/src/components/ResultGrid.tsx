import { useRef, type RefObject } from 'react'
import { useVirtualizer } from '@tanstack/react-virtual'
import type { SearchHit } from '../api/types'
import { ResultCard } from './ResultCard'
import styles from './ResultGrid.module.css'

export interface ResultGridProps {
  hits: SearchHit[]
  selectedKeyframeId?: string | null
  scrollElementRef?: RefObject<HTMLDivElement | null>
  onSelect: (hit: SearchHit) => void
  onFindSimilar: (hit: SearchHit) => void
  onOpenDetail?: (hit: SearchHit) => void
}

const CARD_SIZE_PX = 160
const GAP_PX = 4

// plan §P1.6: "Virtualized rendering cho lưới hàng nghìn thumbnail" --
// max_top_k=100 caps a *live* grid (AIC submission rule, references.md
// §1.2), but the grid component itself must not assume that cap (plan
// §P1.6 finding 2: docs/code contradiction resolved by keeping max_top_k
// and verifying virtualization against a synthetic 1,000-item fixture
// instead). A row virtualizer (not per-card) keeps the DOM bounded to
// roughly one viewport's worth of rows regardless of hit count.
export function ResultGrid({
  hits,
  selectedKeyframeId = null,
  scrollElementRef,
  onSelect,
  onFindSimilar,
  onOpenDetail,
}: ResultGridProps) {
  const scrollRef = useRef<HTMLDivElement>(null)
  const usesExternalScroll = scrollElementRef !== undefined
  const scrollMargin = usesExternalScroll ? (scrollRef.current?.offsetTop ?? 0) : 0

  const columns = Math.max(
    1,
    Math.floor((scrollRef.current?.clientWidth ?? CARD_SIZE_PX) / (CARD_SIZE_PX + GAP_PX)),
  )
  const rowCount = Math.ceil(hits.length / columns)

  const rowVirtualizer = useVirtualizer({
    count: rowCount,
    getScrollElement: () => scrollElementRef?.current ?? scrollRef.current,
    estimateSize: () => CARD_SIZE_PX + GAP_PX,
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
          const rowHits = hits.slice(rowStart, rowStart + columns)
          return (
            <div
              key={virtualRow.key}
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
              {rowHits.map((hit) => (
                <div
                  key={hit.keyframe_id}
                  className={styles.cell}
                  style={{ width: CARD_SIZE_PX, height: CARD_SIZE_PX }}
                >
                  <ResultCard
                    hit={hit}
                    selected={hit.keyframe_id === selectedKeyframeId}
                    onSelect={onSelect}
                    onFindSimilar={onFindSimilar}
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
