import type {
  RetrievalViewMode,
  VideoGroupSortMode,
} from '../utils/retrievalView'
import styles from './ResultBrowserControls.module.css'

export interface ResultBrowserControlsProps {
  viewMode: RetrievalViewMode
  onViewModeChange: (mode: RetrievalViewMode) => void
  groupSortMode: VideoGroupSortMode
  onGroupSortModeChange: (mode: VideoGroupSortMode) => void
  videoFilter: string
  onVideoFilterChange: (value: string) => void
  filteredCount: number
  totalCount: number
}

export function ResultBrowserControls({
  viewMode,
  onViewModeChange,
  groupSortMode,
  onGroupSortModeChange,
  videoFilter,
  onVideoFilterChange,
  filteredCount,
  totalCount,
}: ResultBrowserControlsProps) {
  return (
    <div className={styles.controls}>
      <div className={styles.segment} role="radiogroup" aria-label="result view">
        <button
          type="button"
          className={viewMode === 'ranked' ? styles.active : ''}
          aria-pressed={viewMode === 'ranked'}
          onClick={() => onViewModeChange('ranked')}
        >
          Ranked
        </button>
        <button
          type="button"
          className={viewMode === 'grouped' ? styles.active : ''}
          aria-pressed={viewMode === 'grouped'}
          onClick={() => onViewModeChange('grouped')}
        >
          Group by Video
        </button>
      </div>
      {viewMode === 'grouped' && (
        <div className={styles.segment} role="radiogroup" aria-label="video sort">
          <button
            type="button"
            className={groupSortMode === 'best' ? styles.active : ''}
            aria-pressed={groupSortMode === 'best'}
            onClick={() => onGroupSortModeChange('best')}
          >
            Best Frame
          </button>
          <button
            type="button"
            className={groupSortMode === 'rrf' ? styles.active : ''}
            aria-pressed={groupSortMode === 'rrf'}
            onClick={() => onGroupSortModeChange('rrf')}
          >
            RRF
          </button>
        </div>
      )}
      <label className={styles.filter}>
        <span>Video ID filter</span>
        <input
          aria-label="video id filter"
          value={videoFilter}
          placeholder="L25_V076 L22_V002"
          onChange={(event) => onVideoFilterChange(event.currentTarget.value)}
        />
      </label>
      {videoFilter.trim() && (
        <button type="button" className={styles.clear} onClick={() => onVideoFilterChange('')}>
          Clear
        </button>
      )}
      <span className={styles.count}>
        {filteredCount}/{totalCount}
      </span>
    </div>
  )
}
