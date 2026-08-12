import styles from './FilterPanel.module.css'

// PRD §F9 (D): "bộ lọc metadata". Deliberately left disabled -- plan §P1.5
// decision: SearchRequest has no `filters` field yet and no Milvus filter
// expression is wired (CLAUDE.md §4: filtering must run in the engine, not
// post-hoc), so an enabled control here would be UI theater. Kept as a
// layout placeholder so P1.6's UI shape doesn't need to be rearranged when
// filters are actually implemented.
export function FilterPanel() {
  return (
    <div className={styles.panel} title="Bộ lọc metadata -- chưa triển khai">
      <span>Filters (author / batch / video_id)</span>
      <span className={styles.badge}>chưa có</span>
    </div>
  )
}
