import { useRef, useState } from 'react'
import styles from './ImageUpload.module.css'

export interface ImageUploadProps {
  onUpload: (file: File) => void
  disabled?: boolean
}

// Mirrors the backend's own guard (app/schemas/request/search.py's
// ALLOWED_UPLOAD_MIME_TYPES / MAX_UPLOAD_SIZE_BYTES) so a bad file is
// rejected instantly client-side instead of round-tripping to the API --
// the backend still re-validates independently, this is UX only.
const ALLOWED_MIME_TYPES = new Set(['image/jpeg', 'image/png', 'image/webp'])
const MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024

export function ImageUpload({ onUpload, disabled = false }: ImageUploadProps) {
  const [error, setError] = useState<string | null>(null)
  const [isDragOver, setIsDragOver] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  function validateAndUpload(file: File): void {
    if (disabled) return
    if (!ALLOWED_MIME_TYPES.has(file.type)) {
      setError(`Unsupported image type: ${file.type}`)
      return
    }
    if (file.size > MAX_UPLOAD_SIZE_BYTES) {
      setError('Image exceeds the 10 MiB limit')
      return
    }
    setError(null)
    onUpload(file)
  }

  return (
    <div
      className={isDragOver ? styles.dropZoneActive : styles.dropZone}
      onDragOver={(event) => {
        event.preventDefault()
        setIsDragOver(true)
      }}
      onDragLeave={() => setIsDragOver(false)}
      onDrop={(event) => {
        event.preventDefault()
        setIsDragOver(false)
        if (disabled) return
        const file = event.dataTransfer.files[0]
        if (file) validateAndUpload(file)
      }}
      onClick={() => {
        if (!disabled) inputRef.current?.click()
      }}
      aria-disabled={disabled}
    >
      <input
        ref={inputRef}
        type="file"
        aria-label="Upload query image"
        accept="image/jpeg,image/png,image/webp"
        className={styles.hiddenInput}
        disabled={disabled}
        onChange={(event) => {
          const file = event.target.files?.[0]
          if (file) validateAndUpload(file)
          event.target.value = ''
        }}
      />
      <span>Kéo thả ảnh vào đây hoặc bấm để chọn (image-to-image search)</span>
      {error && <span className={styles.error}>{error}</span>}
    </div>
  )
}
