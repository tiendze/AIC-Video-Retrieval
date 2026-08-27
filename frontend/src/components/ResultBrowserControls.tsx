import type {
  ImageRankSortMode,
  RetrievalViewMode,
  VideoGroupSortMode,
} from '../utils/retrievalView'
import styles from './ResultBrowserControls.module.css'

export interface ResultBrowserControlsProps {
  viewMode: RetrievalViewMode
  onViewModeChange: (mode: RetrievalViewMode) => void
  imageSortMode: ImageRankSortMode
  onImageSortModeChange: (mode: ImageRankSortMode) => void
  groupSortMode: VideoGroupSortMode
  onGroupSortModeChange: (mode: VideoGroupSortMode) => void
  hasLocalRanking: boolean
  videoFilter: string
  onVideoFilterChange: (value: string) => void
  excludedCount: number
  onClearExcludedVideos?: () => void
  filteredCount: number
  totalCount: number
}

export function ResultBrowserControls({
  viewMode,
  onViewModeChange,
  imageSortMode,
  onImageSortModeChange,
  groupSortMode,
  onGroupSortModeChange,
  hasLocalRanking,
  videoFilter,
  onVideoFilterChange,
  excludedCount,
  onClearExcludedVideos,
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
          Images
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
      {viewMode === 'ranked' && (
        <div className={styles.segment} role="radiogroup" aria-label="image sort">
          <button
            type="button"
            className={imageSortMode === 'local' ? styles.active : ''}
            aria-pressed={imageSortMode === 'local'}
            disabled={!hasLocalRanking}
            title={
              hasLocalRanking
                ? 'Sort images by phrase-to-patch local score'
                : 'Local rank appears after a local phrase search'
            }
            onClick={() => onImageSortModeChange('local')}
          >
            Local Rank
          </button>
          <button
            type="button"
            className={imageSortMode === 'global' ? styles.active : ''}
            aria-pressed={imageSortMode === 'global'}
            onClick={() => onImageSortModeChange('global')}
          >
            Global Rank
          </button>
        </div>
      )}
      {viewMode === 'grouped' && (
        <div className={styles.segment} role="radiogroup" aria-label="video sort">
          <button
            type="button"
            className={groupSortMode === 'local' ? styles.active : ''}
            aria-pressed={groupSortMode === 'local'}
            disabled={!hasLocalRanking}
            title={
              hasLocalRanking
                ? 'Sort videos by their best local phrase frame'
                : 'Local video sort appears after a local phrase search'
            }
            onClick={() => onGroupSortModeChange('local')}
          >
            Best Local
          </button>
          <button
            type="button"
            className={groupSortMode === 'global' ? styles.active : ''}
            aria-pressed={groupSortMode === 'global'}
            onClick={() => onGroupSortModeChange('global')}
          >
            Best Global
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
      {excludedCount > 0 && (
        <button type="button" className={styles.clear} onClick={onClearExcludedVideos}>
          Clear excluded ({excludedCount})
        </button>
      )}
      <span className={styles.count}>
        {filteredCount}/{totalCount}
      </span>
    </div>
  )
}
