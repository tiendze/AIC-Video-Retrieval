import styles from './WebImageSearch.module.css'

export interface WebImageSearchProps {
  query: string
  onSearch: (query: string) => void
  disabled?: boolean
}

// QUEST branch 2 trigger (plan §P2.5, PRD §F5): kicks off POST
// /api/quest/image-candidates for the operator's current query, next to
// ImageUpload's local-file flow -- one gives a self-supplied sample image,
// this one gives a web-found one.
export function WebImageSearch({ query, onSearch, disabled = false }: WebImageSearchProps) {
  return (
    <button
      type="button"
      className={styles.button}
      disabled={!query.trim() || disabled}
      title="Tìm ảnh mẫu của thực thể lạ trên web (QUEST nhánh 2)"
      onClick={() => onSearch(query)}
    >
      🌐 Tìm ảnh mẫu trên web{disabled ? '…' : ''}
    </button>
  )
}
