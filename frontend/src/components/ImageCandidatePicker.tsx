import type { ImageCandidate } from '../api/types'
import styles from './ImageCandidatePicker.module.css'

export interface ImageCandidatePickerProps {
  candidates: ImageCandidate[]
  onPick: (candidateId: string) => void
  onDismiss: () => void
}

// QUEST branch 2 UI (plan §P2.5, PRD §F5 acceptance criteria: "Nhánh 2 cho
// phép người dùng CHỌN THỦ CÔNG ảnh mẫu, không tự động chọn ảnh đầu tiên").
// Every candidate renders as its own button -- nothing here auto-selects.
export function ImageCandidatePicker({ candidates, onPick, onDismiss }: ImageCandidatePickerProps) {
  return (
    <div className={styles.overlay} role="dialog" aria-label="Chọn ảnh mẫu">
      <div className={styles.panel}>
        <button type="button" className={styles.close} onClick={onDismiss} aria-label="Close">
          ✕
        </button>
        <p className={styles.hint}>Chọn ảnh giống thực thể bạn đang tìm nhất:</p>
        <div className={styles.grid}>
          {candidates.map((candidate) => (
            <button
              key={candidate.candidate_id}
              type="button"
              className={styles.candidate}
              title={candidate.title}
              onClick={() => onPick(candidate.candidate_id)}
            >
              <img src={candidate.thumbnail_url} alt={candidate.title} loading="lazy" />
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}
