import {
  memo,
  type ChangeEvent,
  type DragEvent,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react'
import { getViewFileFrameResolution } from '../api/client'
import type { SearchHit, ViewFileFrameResolution } from '../api/types'
import {
  parseViewFileCsv,
  type ParsedViewFileCsv,
  type ViewFileCsvRow,
} from '../utils/viewFileCsv'
import { createZipBlob, downloadBlob, uniqueZipEntryNames } from '../utils/zipExport'
import { KeyframeDetail, type KeyframeDetailMedia } from './KeyframeDetail'
import styles from './ViewFileWorkspace.module.css'

const YOUTUBE_NOT_AVAILABLE = 'Youtube Not Available'
const TIMESTAMP_UNAVAILABLE = 'Unable to resolve video timestamp'

interface UploadedViewFile {
  id: string
  name: string
  uploadedAt: number
  rawText: string
  rangeStart: string
  rangeEnd: string
  rowOrder: string[]
  parsed: ParsedViewFileCsv | null
  error: string | null
}

interface RankedViewFileRow {
  row: ViewFileCsvRow
  rank: number
}

interface ViewFilePreview {
  hit: SearchHit
  media: KeyframeDetailMedia
}

const resolutionCache = new Map<string, ViewFileFrameResolution>()

export function ViewFileWorkspace() {
  const inputRef = useRef<HTMLInputElement>(null)
  const nextFileIdRef = useRef(1)
  const draggedFileIdRef = useRef<string | null>(null)
  const [files, setFiles] = useState<UploadedViewFile[]>([])
  const [activeFileId, setActiveFileId] = useState<string | null>(null)
  const [dragActive, setDragActive] = useState(false)
  const [manualOrderTouched, setManualOrderTouched] = useState(false)
  const [fileDropTarget, setFileDropTarget] = useState<{
    id: string
    position: 'before' | 'after'
  } | null>(null)
  const [uploadError, setUploadError] = useState<string | null>(null)
  const [preview, setPreview] = useState<ViewFilePreview | null>(null)
  const activeFile = files.find((file) => file.id === activeFileId) ?? null

  async function addFiles(fileList: FileList | File[]): Promise<void> {
    const incomingFiles = Array.from(fileList)
    if (incomingFiles.length === 0) return

    const csvFiles = incomingFiles.filter((file) => file.name.toLowerCase().endsWith('.csv'))
    if (csvFiles.length !== incomingFiles.length) {
      setUploadError('Only .csv files are accepted.')
    } else {
      setUploadError(null)
    }
    if (csvFiles.length === 0) return

    const parsedFiles = await Promise.all(
      csvFiles.map(async (file) => {
        const text = await file.text()
        const result = parseViewFileCsv(file.name, text)
        return {
          id: `view-file-${Date.now()}-${nextFileIdRef.current++}`,
          name: file.name,
          uploadedAt: Date.now(),
          rawText: text,
          rangeStart: '',
          rangeEnd: '',
          rowOrder: result.ok ? result.parsed.rows.map((row) => row.id) : [],
          parsed: result.ok ? result.parsed : null,
          error: result.ok ? null : result.error,
        }
      }),
    )

    const firstIncomingId = [...parsedFiles].sort(compareUploadedViewFiles)[0]?.id ?? null
    setFiles((prev) => {
      const next = [...prev, ...parsedFiles]
      return manualOrderTouched ? next : next.sort(compareUploadedViewFiles)
    })
    setActiveFileId((current) => current ?? firstIncomingId)
  }

  function handleInputChange(event: ChangeEvent<HTMLInputElement>): void {
    void addFiles(event.currentTarget.files ?? [])
    event.currentTarget.value = ''
  }

  function handleDrop(event: DragEvent<HTMLDivElement>): void {
    event.preventDefault()
    setDragActive(false)
    void addFiles(event.dataTransfer.files)
  }

  function removeFile(fileId: string): void {
    setFiles((prev) => {
      const next = prev.filter((file) => file.id !== fileId)
      if (activeFileId === fileId) {
        setActiveFileId(next[0]?.id ?? null)
      }
      return next
    })
  }

  function updateFileRange(
    fileId: string,
    field: 'rangeStart' | 'rangeEnd',
    value: string,
  ): void {
    setFiles((prev) =>
      prev.map((file) => (file.id === fileId ? { ...file, [field]: value } : file)),
    )
  }

  function reorderFile(fromId: string, targetId: string, position: 'before' | 'after'): void {
    if (fromId === targetId) return
    setFiles((prev) => {
      const fromIndex = prev.findIndex((file) => file.id === fromId)
      if (fromIndex < 0) return prev
      const moving = prev[fromIndex]
      const withoutMoving = prev.filter((file) => file.id !== fromId)
      const targetIndex = withoutMoving.findIndex((file) => file.id === targetId)
      if (targetIndex < 0) return prev
      const insertionIndex = position === 'after' ? targetIndex + 1 : targetIndex
      const next = [...withoutMoving]
      next.splice(insertionIndex, 0, moving)
      return next
    })
    setManualOrderTouched(true)
  }

  function reorderFileRow(
    fileId: string,
    fromRowId: string,
    targetRowId: string,
    position: 'before' | 'after',
  ): void {
    if (fromRowId === targetRowId) return
    setFiles((prev) =>
      prev.map((file) => {
        if (file.id !== fileId || !file.parsed) return file
        const currentOrder =
          file.rowOrder.length === file.parsed.rows.length
            ? file.rowOrder
            : file.parsed.rows.map((row) => row.id)
        const fromIndex = currentOrder.indexOf(fromRowId)
        if (fromIndex < 0) return file
        const withoutMoving = currentOrder.filter((rowId) => rowId !== fromRowId)
        const targetIndex = withoutMoving.indexOf(targetRowId)
        if (targetIndex < 0) return file
        const insertionIndex = position === 'after' ? targetIndex + 1 : targetIndex
        const nextOrder = [...withoutMoving]
        nextOrder.splice(insertionIndex, 0, fromRowId)
        return { ...file, rowOrder: nextOrder }
      }),
    )
  }

  function exportZip(): void {
    if (files.length === 0) return
    const entryNames = uniqueZipEntryNames(files.map((file) => file.name)).map(
      (name) => `submissions/${name}`,
    )
    const manifest = {
      generated_at: new Date().toISOString(),
      file_count: files.length,
      files: files.map((file, index) => ({
        order: index + 1,
        id: file.id,
        filename: file.name,
        zip_entry: entryNames[index],
        detected_type: file.parsed?.kind.toUpperCase() ?? null,
        rows: file.parsed?.rows.length ?? 0,
        valid: file.error === null && file.parsed !== null,
        error: file.error,
        range_start: parseRangeBoundary(file.rangeStart),
        range_end: parseRangeBoundary(file.rangeEnd),
        range_valid: !isRangeInvalid(file),
        frame_order: file.rowOrder,
      })),
    }
    const blob = createZipBlob([
      ...files.map((file, index) => ({
        name: entryNames[index],
        content: currentCsvContentForFile(file),
      })),
      {
        name: 'submissions/view_file_manifest.json',
        content: JSON.stringify(manifest, null, 2),
      },
    ])
    downloadBlob('view-file-export.zip', blob)
  }

  useEffect(() => {
    if (!preview) return undefined
    function onKeyDown(event: KeyboardEvent): void {
      if (event.key === 'Escape') setPreview(null)
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [preview])

  const openPreview = useCallback((row: ViewFileCsvRow, rank: number, resolution: ViewFileFrameResolution | null) => {
    setPreview(createPreview(row, rank, resolution))
  }, [])

  const validCount = files.filter((file) => file.parsed !== null).length

  return (
    <section className={styles.workspace} aria-label="View File workspace">
      <header className={styles.header}>
        <div>
          <p className={styles.eyebrow}>View File</p>
          <h1 className={styles.title}>CSV Workspace</h1>
        </div>
        <div className={styles.headerStats}>
          <span>{files.length} files</span>
          <span>{validCount} ready</span>
          <button
            type="button"
            className={styles.secondaryButton}
            disabled={files.length === 0}
            onClick={exportZip}
          >
            Export ZIP
          </button>
        </div>
      </header>

      <div
        className={`${styles.dropZone} ${dragActive ? styles.dropZoneActive : ''}`}
        onDragEnter={(event) => {
          event.preventDefault()
          setDragActive(true)
        }}
        onDragOver={(event) => event.preventDefault()}
        onDragLeave={() => setDragActive(false)}
        onDrop={handleDrop}
      >
        <input
          ref={inputRef}
          className={styles.fileInput}
          type="file"
          aria-label="CSV file upload"
          accept=".csv,text/csv"
          multiple
          onChange={handleInputChange}
        />
        <div>
          <strong>Drop CSV files here</strong>
          <span>KIS and VQA submissions can stay open together.</span>
        </div>
        <button
          type="button"
          className={styles.primaryButton}
          onClick={() => inputRef.current?.click()}
        >
          Choose Files
        </button>
      </div>

      {uploadError && (
        <div className={styles.alert} role="alert">
          {uploadError}
        </div>
      )}

      <div className={styles.body}>
        <aside className={styles.fileRail} aria-label="Uploaded CSV files">
          <div className={styles.fileRailHeader}>
            <span>Files</span>
            <button
              type="button"
              className={styles.linkButton}
              disabled={files.length === 0}
              onClick={() => {
                setFiles([])
                setActiveFileId(null)
                setPreview(null)
              }}
            >
              Clear All
            </button>
          </div>
          {files.length === 0 ? (
            <p className={styles.empty}>No CSV loaded.</p>
          ) : (
            <div className={styles.fileList}>
              {files.map((file) => (
                <div
                  key={file.id}
                  data-testid={`view-file-item-${file.name}`}
                  data-file-id={file.id}
                  className={`${styles.fileItem} ${
                    activeFileId === file.id ? styles.fileItemActive : ''
                  } ${file.error ? styles.fileItemInvalid : ''} ${
                    fileDropTarget?.id === file.id && fileDropTarget.position === 'before'
                      ? styles.fileItemDropBefore
                      : ''
                  } ${
                    fileDropTarget?.id === file.id && fileDropTarget.position === 'after'
                      ? styles.fileItemDropAfter
                      : ''
                  }`}
                  onDragOver={(event) => {
                    event.preventDefault()
                    event.dataTransfer.dropEffect = 'move'
                    const rect = event.currentTarget.getBoundingClientRect()
                    const position =
                      event.clientY < rect.top + rect.height / 2 ? 'before' : 'after'
                    setFileDropTarget({ id: file.id, position })
                  }}
                  onDragLeave={() => setFileDropTarget(null)}
                  onDrop={(event) => {
                    event.preventDefault()
                    const fromId =
                      event.dataTransfer.getData('text/plain') || draggedFileIdRef.current
                    const position = fileDropTarget?.position ?? 'before'
                    draggedFileIdRef.current = null
                    setFileDropTarget(null)
                    if (fromId) reorderFile(fromId, file.id, position)
                  }}
                >
                  <span
                    className={styles.dragHandle}
                    role="button"
                    tabIndex={files.length > 1 ? 0 : -1}
                    draggable={files.length > 1}
                    aria-label={`Drag ${file.name}`}
                    title="Drag to reorder"
                    onDragStart={(event) => {
                      draggedFileIdRef.current = file.id
                      event.dataTransfer.effectAllowed = 'move'
                      event.dataTransfer.setData('text/plain', file.id)
                    }}
                    onDragEnd={() => {
                      draggedFileIdRef.current = null
                      setFileDropTarget(null)
                    }}
                  >
                    ||
                  </span>
                  <button
                    type="button"
                    className={styles.fileSelect}
                    aria-label={`Select ${file.name}`}
                    onClick={() => setActiveFileId(file.id)}
                  >
                    <span className={styles.fileName} data-testid="view-file-name">
                      {file.name}
                    </span>
                    <span className={styles.fileMeta}>
                      {file.error
                        ? 'Invalid CSV'
                        : `${file.parsed?.kind.toUpperCase()} · ${
                            file.parsed?.rows.length ?? 0
                          } rows`}
                    </span>
                    {formatRangeSummary(file) && (
                      <span className={styles.fileRange}>{formatRangeSummary(file)}</span>
                    )}
                  </button>
                  <button
                    type="button"
                    className={styles.removeFile}
                    aria-label={`Remove ${file.name}`}
                    onClick={() => {
                      removeFile(file.id)
                    }}
                  >
                    x
                  </button>
                </div>
              ))}
            </div>
          )}
        </aside>

        <main className={styles.viewer}>
          {!activeFile ? (
            <div className={styles.placeholder}>Upload a CSV to inspect frames.</div>
          ) : activeFile.error || !activeFile.parsed ? (
            <div className={styles.invalidState} role="alert">
              <strong>{activeFile.name}</strong>
              <span>{activeFile.error}</span>
            </div>
          ) : (
            <CsvPreview
              file={activeFile}
              onRangeChange={updateFileRange}
              onReorderRow={reorderFileRow}
              onOpenFrame={openPreview}
            />
          )}
        </main>
      </div>

      {preview && (
        <KeyframeDetail
          hit={preview.hit}
          mediaOverride={preview.media}
          highlightedFrames={[]}
          onClose={() => setPreview(null)}
        />
      )}
    </section>
  )
}

function CsvPreview({
  file,
  onRangeChange,
  onReorderRow,
  onOpenFrame,
}: {
  file: UploadedViewFile
  onRangeChange: (
    fileId: string,
    field: 'rangeStart' | 'rangeEnd',
    value: string,
  ) => void
  onReorderRow: (
    fileId: string,
    fromRowId: string,
    targetRowId: string,
    position: 'before' | 'after',
  ) => void
  onOpenFrame: (
    row: ViewFileCsvRow,
    rank: number,
    resolution: ViewFileFrameResolution | null,
  ) => void
}) {
  const parsed = file.parsed
  const orderedRows = useMemo(
    () => (parsed ? orderRowsByIds(parsed.rows, file.rowOrder) : []),
    [file.rowOrder, parsed],
  )
  const rankedRows = useMemo(
    () => orderedRows.map((row, index) => ({ row, rank: index + 1 })),
    [orderedRows],
  )
  const answerGroups = useMemo(
    () => (parsed ? groupRowsByAnswer(rankedRows) : []),
    [parsed, rankedRows],
  )
  if (!parsed) return null
  const rangeInvalid = isRangeInvalid(file)

  return (
    <section className={styles.preview} aria-label={`${file.name} preview`}>
      <div className={styles.previewHeader}>
        <div>
          <span className={styles.kindBadge}>{parsed.kind.toUpperCase()}</span>
          <h2>{file.name}</h2>
        </div>
        <span className={styles.previewMeta}>
          {parsed.rows.length} rows · loaded {formatTime(file.uploadedAt)}
        </span>
      </div>
      <div
        className={`${styles.rangeEditor} ${rangeInvalid ? styles.rangeEditorInvalid : ''}`}
        aria-label="Frame range note"
      >
        <span>Frame range note</span>
        <label>
          Start
          <input
            type="number"
            aria-label="Frame range start"
            value={file.rangeStart}
            onChange={(event) => onRangeChange(file.id, 'rangeStart', event.target.value)}
          />
        </label>
        <label>
          End
          <input
            type="number"
            aria-label="Frame range end"
            value={file.rangeEnd}
            onChange={(event) => onRangeChange(file.id, 'rangeEnd', event.target.value)}
          />
        </label>
        {rangeInvalid && (
          <strong role="alert">Start must be less than or equal to end.</strong>
        )}
      </div>

      {parsed.kind === 'kis' ? (
        <FrameGrid
          fileId={file.id}
          rows={rankedRows}
          onOpenFrame={onOpenFrame}
          onReorderRow={onReorderRow}
        />
      ) : (
        <div className={styles.answerStack}>
          {answerGroups.map((group) => (
            <section key={group.answer} className={styles.answerGroup}>
              <header className={styles.answerHeader}>
                <h3>{group.answer}</h3>
                <span>{group.rows.length} rows</span>
              </header>
              <FrameGrid
                fileId={file.id}
                rows={group.rows}
                onOpenFrame={onOpenFrame}
                onReorderRow={onReorderRow}
              />
            </section>
          ))}
        </div>
      )}
    </section>
  )
}

const FrameGrid = memo(function FrameGrid({
  fileId,
  rows,
  onOpenFrame,
  onReorderRow,
}: {
  fileId: string
  rows: RankedViewFileRow[]
  onOpenFrame: (
    row: ViewFileCsvRow,
    rank: number,
    resolution: ViewFileFrameResolution | null,
  ) => void
  onReorderRow: (
    fileId: string,
    fromRowId: string,
    targetRowId: string,
    position: 'before' | 'after',
  ) => void
}) {
  const draggedRowIdRef = useRef<string | null>(null)
  const [rowDropTarget, setRowDropTarget] = useState<{
    id: string
    position: 'before' | 'after'
  } | null>(null)

  return (
    <div className={styles.frameGrid}>
      {rows.map(({ row, rank }) => {
        const className = [
          styles.frameItem,
          rowDropTarget?.id === row.id && rowDropTarget.position === 'before'
            ? styles.frameItemDropBefore
            : '',
          rowDropTarget?.id === row.id && rowDropTarget.position === 'after'
            ? styles.frameItemDropAfter
            : '',
        ]
          .filter(Boolean)
          .join(' ')
        return (
          <div
            key={row.id}
            className={className}
            data-testid={`view-frame-item-${rank}`}
            data-row-id={row.id}
            onDragOver={(event) => {
              event.preventDefault()
              event.dataTransfer.dropEffect = 'move'
              const rect = event.currentTarget.getBoundingClientRect()
              const position =
                event.clientX < rect.left + rect.width / 2 ? 'before' : 'after'
              setRowDropTarget({ id: row.id, position })
            }}
            onDragLeave={() => setRowDropTarget(null)}
            onDrop={(event) => {
              event.preventDefault()
              const fromRowId =
                event.dataTransfer.getData('text/plain') || draggedRowIdRef.current
              const position = rowDropTarget?.position ?? 'before'
              draggedRowIdRef.current = null
              setRowDropTarget(null)
              if (fromRowId) onReorderRow(fileId, fromRowId, row.id, position)
            }}
          >
            <span
              className={styles.frameDragHandle}
              role="button"
              tabIndex={rows.length > 1 ? 0 : -1}
              draggable={rows.length > 1}
              aria-label={`Drag frame ${row.videoId} ${row.frameIdx}`}
              title="Drag to reorder"
              onDragStart={(event) => {
                draggedRowIdRef.current = row.id
                event.dataTransfer.effectAllowed = 'move'
                event.dataTransfer.setData('text/plain', row.id)
              }}
              onDragEnd={() => {
                draggedRowIdRef.current = null
                setRowDropTarget(null)
              }}
            >
              ||
            </span>
            <ViewFileFrameCard row={row} rank={rank} onOpenFrame={onOpenFrame} />
          </div>
        )
      })}
    </div>
  )
})

const ViewFileFrameCard = memo(function ViewFileFrameCard({
  row,
  rank,
  onOpenFrame,
}: {
  row: ViewFileCsvRow
  rank: number
  onOpenFrame: (
    row: ViewFileCsvRow,
    rank: number,
    resolution: ViewFileFrameResolution | null,
  ) => void
}) {
  const cacheKey = `${row.videoId}:${row.frameIdx}`
  const cached = resolutionCache.get(cacheKey)
  const [resolution, setResolution] = useState<ViewFileFrameResolution | null>(cached ?? null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const existing = resolutionCache.get(cacheKey)
    if (existing) {
      setResolution(existing)
      setError(null)
      return undefined
    }

    const controller = new AbortController()
    setResolution(null)
    setError(null)
    getViewFileFrameResolution(row.videoId, row.frameIdx, controller.signal)
      .then((nextResolution) => {
        resolutionCache.set(cacheKey, nextResolution)
        setResolution(nextResolution)
      })
      .catch((nextError: unknown) => {
        if (controller.signal.aborted) return
        setError(nextError instanceof Error ? nextError.message : YOUTUBE_NOT_AVAILABLE)
      })

    return () => controller.abort()
  }, [cacheKey, row.frameIdx, row.videoId])

  const state = cardStateFromResolution(resolution, error)
  const isLoading = state.kind === 'loading'
  const className = [
    styles.frameCard,
    state.kind === 'local' ? styles.frameCardLocal : '',
    state.kind === 'youtube' ? styles.frameCardYoutube : '',
    state.kind === 'unavailable' || state.kind === 'error' ? styles.frameCardUnavailable : '',
  ]
    .filter(Boolean)
    .join(' ')

  return (
    <button
      type="button"
      className={className}
      aria-label={`Open ${row.videoId} frame ${row.frameIdx}`}
      disabled={isLoading}
      onClick={() => onOpenFrame(row, rank, resolution)}
    >
      <span className={styles.rankBadge}>#{rank}</span>
      <span className={styles.frameMedia}>
        {state.kind === 'local' ? (
          <img
            src={state.thumbnailUrl}
            alt={`${row.videoId} ${row.frameIdx}`}
            loading="lazy"
            decoding="async"
          />
        ) : state.kind === 'loading' ? (
          <span className={styles.frameSkeleton} aria-hidden="true" />
        ) : (
          <span className={styles.placeholderText}>{state.label}</span>
        )}
      </span>
      <span className={styles.frameFooter}>
        <span title={row.videoId}>{row.videoId}</span>
        <span>{row.frameIdx}</span>
      </span>
    </button>
  )
})

type CardState =
  | { kind: 'loading' }
  | { kind: 'local'; thumbnailUrl: string }
  | { kind: 'youtube'; label: string }
  | { kind: 'unavailable'; label: string }
  | { kind: 'error'; label: string }

function cardStateFromResolution(
  resolution: ViewFileFrameResolution | null,
  error: string | null,
): CardState {
  if (error) return { kind: 'error', label: YOUTUBE_NOT_AVAILABLE }
  if (!resolution) return { kind: 'loading' }
  if (resolution.source === 'local' && resolution.thumbnail_url) {
    return { kind: 'local', thumbnailUrl: resolution.thumbnail_url }
  }
  if (resolution.source === 'youtube') return { kind: 'youtube', label: 'YouTube' }
  return {
    kind: 'unavailable',
    label: resolution.error_message || YOUTUBE_NOT_AVAILABLE,
  }
}

function createPreview(
  row: ViewFileCsvRow,
  rank: number,
  resolution: ViewFileFrameResolution | null,
): ViewFilePreview {
  const timestamp = resolution?.timestamp_seconds ?? 0
  const hit: SearchHit = {
    keyframe_id: resolution?.keyframe_id ?? `view-file:${row.videoId}:${row.frameIdx}:${rank}`,
    video_id: row.videoId,
    score: 0,
    frame_idx: row.frameIdx,
    timestamp,
    thumbnail_path: '',
    thumbnail_url: resolution?.thumbnail_url ?? '',
    image_url: resolution?.image_url ?? '',
  }

  if (resolution?.source === 'local' && resolution.image_url) {
    return { hit, media: { kind: 'image' } }
  }
  if (resolution?.source === 'youtube' && resolution.youtube_url) {
    if (resolution.timestamp_seconds === null || !resolution.fps || resolution.fps <= 0) {
      return { hit, media: { kind: 'unavailable', message: TIMESTAMP_UNAVAILABLE } }
    }
    return {
      hit,
      media: {
        kind: 'youtube',
        youtubeUrl: resolution.youtube_url,
        timestampSeconds: resolution.timestamp_seconds,
        fps: resolution.fps,
      },
    }
  }
  return {
    hit,
    media: {
      kind: 'unavailable',
      message: resolution?.error_message || YOUTUBE_NOT_AVAILABLE,
    },
  }
}

function groupRowsByAnswer(
  rows: RankedViewFileRow[],
): { answer: string; rows: RankedViewFileRow[] }[] {
  const groups = new Map<string, RankedViewFileRow[]>()
  for (const row of rows) {
    const answer = row.row.answer ?? ''
    const group = groups.get(answer)
    if (group) {
      group.push(row)
    } else {
      groups.set(answer, [row])
    }
  }
  return Array.from(groups, ([answer, groupedRows]) => ({ answer, rows: groupedRows }))
}

function formatTime(value: number): string {
  return new Intl.DateTimeFormat(undefined, {
    hour: '2-digit',
    minute: '2-digit',
  }).format(value)
}

function orderRowsByIds(rows: ViewFileCsvRow[], order: string[]): ViewFileCsvRow[] {
  if (order.length === 0) return rows
  const rowById = new Map(rows.map((row) => [row.id, row]))
  const ordered = order.flatMap((rowId) => {
    const row = rowById.get(rowId)
    return row ? [row] : []
  })
  if (ordered.length === rows.length) return ordered
  const orderedIds = new Set(ordered.map((row) => row.id))
  return [...ordered, ...rows.filter((row) => !orderedIds.has(row.id))]
}

function currentCsvContentForFile(file: UploadedViewFile): string {
  if (!file.parsed) return file.rawText
  const rows = orderRowsByIds(file.parsed.rows, file.rowOrder)
  return rows
    .map((row) => {
      const cells = [row.videoId, String(Math.trunc(row.frameIdx))]
      if (file.parsed?.kind === 'vqa') cells.push(quoteCsvCell(row.answer ?? ''))
      return cells.join(',')
    })
    .join('\n')
}

function quoteCsvCell(value: string): string {
  if (!/[",\r\n]/.test(value)) return value
  return `"${value.replace(/"/g, '""')}"`
}

function compareUploadedViewFiles(a: UploadedViewFile, b: UploadedViewFile): number {
  const questionA = extractQuestionNumber(a.name)
  const questionB = extractQuestionNumber(b.name)
  if (questionA !== null && questionB !== null && questionA !== questionB) {
    return questionA - questionB
  }
  if (questionA !== null && questionB === null) return -1
  if (questionA === null && questionB !== null) return 1
  return a.name.localeCompare(b.name, undefined, { numeric: true, sensitivity: 'base' })
}

function extractQuestionNumber(name: string): number | null {
  const primary = name.match(/query-p\d+-(\d+)/i)
  if (primary) return Number(primary[1])
  const query = name.match(/query-(\d+)/i)
  if (query) return Number(query[1])
  const numbers = Array.from(name.matchAll(/\d+/g))
  if (numbers.length === 0) return null
  return Number(numbers[numbers.length - 1][0])
}

function parseRangeBoundary(value: string): number | null {
  const trimmed = value.trim()
  if (trimmed.length === 0) return null
  const parsed = Number(trimmed)
  return Number.isFinite(parsed) ? parsed : null
}

function isRangeInvalid(file: UploadedViewFile): boolean {
  const start = parseRangeBoundary(file.rangeStart)
  const end = parseRangeBoundary(file.rangeEnd)
  return start !== null && end !== null && start > end
}

function formatRangeSummary(file: UploadedViewFile): string | null {
  if (file.rangeStart.trim().length === 0 && file.rangeEnd.trim().length === 0) return null
  const start = file.rangeStart.trim() || '--'
  const end = file.rangeEnd.trim() || '--'
  return `Range ${start} -> ${end}`
}
