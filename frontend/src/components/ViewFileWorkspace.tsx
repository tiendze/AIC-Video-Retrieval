import {
  memo,
  type ChangeEvent,
  type CSSProperties,
  type DragEvent,
  type FormEvent,
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
const RANGE_COLORS = ['#22c55e', '#f59e0b', '#a78bfa', '#f472b6', '#2dd4bf', '#fb7185']

interface ViewFileRange {
  id: string
  startFrame: string
  endFrame: string
  description: string
}

interface UploadedViewFile {
  id: string
  name: string
  uploadedAt: number
  rawText: string
  rangeVideoId: string
  ranges: ViewFileRange[]
  sortedRangeIds: string[]
  videoFilter: string
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
  const nextRangeIdRef = useRef(1)
  const nextManualRowIdRef = useRef(1)
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
          rangeVideoId: '',
          ranges: [],
          sortedRangeIds: [],
          videoFilter: '',
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

  function addFileRange(fileId: string): void {
    setFiles((prev) =>
      prev.map((file) =>
        file.id === fileId
          ? {
              ...file,
              ranges: [
                ...file.ranges,
                {
                  id: `range-${Date.now()}-${nextRangeIdRef.current++}`,
                  startFrame: '',
                  endFrame: '',
                  description: '',
                },
              ],
            }
          : file,
      ),
    )
  }

  function updateFileRange(
    fileId: string,
    rangeId: string,
    field: 'startFrame' | 'endFrame' | 'description',
    value: string,
  ): void {
    setFiles((prev) =>
      prev.map((file) =>
        file.id === fileId
          ? {
              ...file,
              ranges: file.ranges.map((range) =>
                range.id === rangeId ? { ...range, [field]: value } : range,
              ),
            }
          : file,
      ),
    )
  }

  function updateRangeVideoId(fileId: string, value: string): void {
    setFiles((prev) =>
      prev.map((file) =>
        file.id === fileId
          ? {
              ...file,
              rangeVideoId: value,
              rowOrder: file.parsed
                ? orderRowsForSortedRanges(file, file.sortedRangeIds, value)
                : file.rowOrder,
            }
          : file,
      ),
    )
  }

  function removeFileRange(fileId: string, rangeId: string): void {
    setFiles((prev) =>
      prev.map((file) =>
        file.id === fileId
          ? {
              ...file,
              ranges: file.ranges.filter((range) => range.id !== rangeId),
              sortedRangeIds: file.sortedRangeIds.filter(
                (sortedRangeId) => sortedRangeId !== rangeId,
              ),
            }
          : file,
      ),
    )
  }

  function reorderFileRange(
    fileId: string,
    fromRangeId: string,
    targetRangeId: string,
    position: 'before' | 'after',
  ): void {
    if (fromRangeId === targetRangeId) return
    setFiles((prev) =>
      prev.map((file) => {
        if (file.id !== fileId) return file
        const fromIndex = file.ranges.findIndex((range) => range.id === fromRangeId)
        if (fromIndex < 0) return file
        const moving = file.ranges[fromIndex]
        const withoutMoving = file.ranges.filter((range) => range.id !== fromRangeId)
        const targetIndex = withoutMoving.findIndex((range) => range.id === targetRangeId)
        if (targetIndex < 0) return file
        const insertionIndex = position === 'after' ? targetIndex + 1 : targetIndex
        const ranges = [...withoutMoving]
        ranges.splice(insertionIndex, 0, moving)
        return { ...file, ranges }
      }),
    )
  }

  function updateVideoFilter(fileId: string, value: string): void {
    setFiles((prev) =>
      prev.map((file) => (file.id === fileId ? { ...file, videoFilter: value } : file)),
    )
  }

  function addFileRow(
    fileId: string,
    videoId: string,
    frameIdx: number,
    answer: string | null,
  ): void {
    setFiles((prev) =>
      prev.map((file) => {
        if (file.id !== fileId || !file.parsed) return file
        const row: ViewFileCsvRow = {
          id: `${file.name}:manual:${nextManualRowIdRef.current++}:${videoId}:${frameIdx}`,
          rowNumber:
            file.parsed.rows.reduce((maxRow, current) => Math.max(maxRow, current.rowNumber), 0) +
            1,
          videoId,
          frameIdx,
          answer: file.parsed.kind === 'vqa' ? answer ?? '' : null,
        }
        return {
          ...file,
          parsed: {
            ...file.parsed,
            rows: [...file.parsed.rows, row],
          },
          rowOrder: [...file.rowOrder, row.id],
        }
      }),
    )
  }

  function removeFileRow(fileId: string, rowId: string): void {
    setFiles((prev) =>
      prev.map((file) => {
        if (file.id !== fileId || !file.parsed) return file
        return {
          ...file,
          parsed: {
            ...file.parsed,
            rows: file.parsed.rows.filter((row) => row.id !== rowId),
          },
          rowOrder: file.rowOrder.filter((currentRowId) => currentRowId !== rowId),
        }
      }),
    )
  }

  function sortFileRowsByRange(fileId: string, rangeId: string, sortPosition: number): void {
    setFiles((prev) =>
      prev.map((file) => {
        if (file.id !== fileId || !file.parsed) return file
        const selectedRange = file.ranges.find((range) => range.id === rangeId)
        if (!selectedRange || !isViewFileRangeSortable(selectedRange)) return file
        const liveRangeIds = new Set(file.ranges.map((range) => range.id))
        const existingSortedRangeIds = file.sortedRangeIds.filter(
          (sortedRangeId) => sortedRangeId !== rangeId && liveRangeIds.has(sortedRangeId),
        )
        const insertionIndex = Math.min(
          Math.max(Math.trunc(sortPosition) - 1, 0),
          existingSortedRangeIds.length,
        )
        const sortedRangeIds = [...existingSortedRangeIds]
        sortedRangeIds.splice(insertionIndex, 0, rangeId)
        return {
          ...file,
          sortedRangeIds,
          rowOrder: orderRowsForSortedRanges(file, sortedRangeIds),
        }
      }),
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
        range_start: parseRangeBoundary(file.ranges[0]?.startFrame ?? ''),
        range_end: parseRangeBoundary(file.ranges[0]?.endFrame ?? ''),
        range_video_id: file.rangeVideoId.trim(),
        range_valid: !isRangeInvalid(file),
        ranges: file.ranges.map((range, rangeIndex) => ({
          index: rangeIndex + 1,
          video_id: file.rangeVideoId.trim(),
          range_start: parseRangeBoundary(range.startFrame),
          range_end: parseRangeBoundary(range.endFrame),
          description: range.description.trim(),
          valid: !isViewFileRangeInvalid(range),
        })),
        sorted_ranges: file.sortedRangeIds,
        video_filter: file.videoFilter.trim(),
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
              onAddRange={addFileRange}
              onRangeChange={updateFileRange}
              onRangeVideoIdChange={updateRangeVideoId}
              onRemoveRange={removeFileRange}
              onReorderRange={reorderFileRange}
              onSortRange={sortFileRowsByRange}
              onVideoFilterChange={updateVideoFilter}
              onAddFrame={addFileRow}
              onRemoveFrame={removeFileRow}
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
  onAddRange,
  onRangeChange,
  onRangeVideoIdChange,
  onRemoveRange,
  onReorderRange,
  onSortRange,
  onVideoFilterChange,
  onAddFrame,
  onRemoveFrame,
  onReorderRow,
  onOpenFrame,
}: {
  file: UploadedViewFile
  onAddRange: (fileId: string) => void
  onRangeChange: (
    fileId: string,
    rangeId: string,
    field: 'startFrame' | 'endFrame' | 'description',
    value: string,
  ) => void
  onRangeVideoIdChange: (fileId: string, value: string) => void
  onRemoveRange: (fileId: string, rangeId: string) => void
  onReorderRange: (
    fileId: string,
    fromRangeId: string,
    targetRangeId: string,
    position: 'before' | 'after',
  ) => void
  onSortRange: (fileId: string, rangeId: string, sortPosition: number) => void
  onVideoFilterChange: (fileId: string, value: string) => void
  onAddFrame: (fileId: string, videoId: string, frameIdx: number, answer: string | null) => void
  onRemoveFrame: (fileId: string, rowId: string) => void
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
  const draggedRangeIdRef = useRef<string | null>(null)
  const [rangeDropTarget, setRangeDropTarget] = useState<{
    id: string
    position: 'before' | 'after'
  } | null>(null)
  const parsed = file.parsed
  const orderedRows = useMemo(
    () => (parsed ? orderRowsByIds(parsed.rows, file.rowOrder) : []),
    [file.rowOrder, parsed],
  )
  const rankedRows = useMemo(
    () => orderedRows.map((row, index) => ({ row, rank: index + 1 })),
    [orderedRows],
  )
  const [newVideoId, setNewVideoId] = useState('')
  const [newFrameIdx, setNewFrameIdx] = useState('')
  const [newAnswer, setNewAnswer] = useState('')
  const [addFrameError, setAddFrameError] = useState<string | null>(null)

  useEffect(() => {
    setNewVideoId('')
    setNewFrameIdx('')
    setNewAnswer('')
    setAddFrameError(null)
  }, [file.id])

  if (!parsed) return null
  const parsedKind = parsed.kind
  const rangeInvalid = isRangeInvalid(file)
  const normalizedVideoFilter = normalizeVideoFilter(file.videoFilter)
  const videoFilterMatchCount = normalizedVideoFilter
    ? rankedRows.filter(({ row }) => videoMatchesFilter(row.videoId, normalizedVideoFilter)).length
    : 0

  function handleAddFrame(event: FormEvent<HTMLFormElement>): void {
    event.preventDefault()
    const videoId = newVideoId.trim()
    const frameText = newFrameIdx.trim()
    const answer = parsedKind === 'vqa' ? newAnswer.trim() : null
    if (!videoId) {
      setAddFrameError('Video ID is required.')
      return
    }
    if (!/^\d+$/.test(frameText)) {
      setAddFrameError('Frame must be a non-negative integer.')
      return
    }
    if (parsedKind === 'vqa' && !answer) {
      setAddFrameError('Answer is required for VQA rows.')
      return
    }
    onAddFrame(file.id, videoId, Number(frameText), answer)
    setNewVideoId('')
    setNewFrameIdx('')
    setNewAnswer('')
    setAddFrameError(null)
  }

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
        aria-label="Frame ranges"
      >
        <div className={styles.rangeEditorHeader}>
          <span>Frame ranges</span>
          <button
            type="button"
            className={styles.secondaryButton}
            onClick={() => onAddRange(file.id)}
          >
            Add Range
          </button>
        </div>
        <label className={styles.rangeVideoLabel}>
          video_id
          <input
            type="text"
            aria-label="Range video ID"
            value={file.rangeVideoId}
            onChange={(event) => onRangeVideoIdChange(file.id, event.target.value)}
            placeholder="L21_V001"
          />
        </label>
        {file.ranges.map((range, rangeIndex) => {
          const color = RANGE_COLORS[rangeIndex % RANGE_COLORS.length]
          const currentSortPosition = file.sortedRangeIds.indexOf(range.id) + 1
          return (
            <div
              key={range.id}
              data-testid={`view-range-item-${rangeIndex + 1}`}
              data-range-id={range.id}
              className={`${styles.rangeRow} ${
                rangeDropTarget?.id === range.id && rangeDropTarget.position === 'before'
                  ? styles.rangeRowDropBefore
                  : ''
              } ${
                rangeDropTarget?.id === range.id && rangeDropTarget.position === 'after'
                  ? styles.rangeRowDropAfter
                  : ''
              }`}
              style={{ '--range-color': color } as CSSProperties}
              onDragOver={(event) => {
                event.preventDefault()
                event.dataTransfer.dropEffect = 'move'
                const rect = event.currentTarget.getBoundingClientRect()
                const position =
                  event.clientY < rect.top + rect.height / 2 ? 'before' : 'after'
                setRangeDropTarget({ id: range.id, position })
              }}
              onDragLeave={() => setRangeDropTarget(null)}
              onDrop={(event) => {
                event.preventDefault()
                const fromRangeId =
                  event.dataTransfer.getData('text/plain') || draggedRangeIdRef.current
                const position = rangeDropTarget?.position ?? 'before'
                draggedRangeIdRef.current = null
                setRangeDropTarget(null)
                if (fromRangeId) onReorderRange(file.id, fromRangeId, range.id, position)
              }}
            >
              <span
                className={styles.rangeDragHandle}
                role="button"
                tabIndex={file.ranges.length > 1 ? 0 : -1}
                draggable={file.ranges.length > 1}
                aria-label={`Drag range ${rangeIndex + 1}`}
                title="Drag to reorder"
                onDragStart={(event) => {
                  draggedRangeIdRef.current = range.id
                  event.dataTransfer.effectAllowed = 'move'
                  event.dataTransfer.setData('text/plain', range.id)
                }}
                onDragEnd={() => {
                  draggedRangeIdRef.current = null
                  setRangeDropTarget(null)
                }}
              >
                ||
              </span>
              <span className={styles.rangeIndex}>#{rangeIndex + 1}</span>
              <label>
                Start
                <input
                  type="number"
                  aria-label={`Range ${rangeIndex + 1} start frame`}
                  value={range.startFrame}
                  onChange={(event) =>
                    onRangeChange(file.id, range.id, 'startFrame', event.target.value)
                  }
                />
              </label>
              <label>
                End
                <input
                  type="number"
                  aria-label={`Range ${rangeIndex + 1} end frame`}
                  value={range.endFrame}
                  onChange={(event) =>
                    onRangeChange(file.id, range.id, 'endFrame', event.target.value)
                  }
                />
              </label>
              <label className={styles.rangeDescriptionLabel}>
                Description
                <input
                  type="text"
                  aria-label={`Range ${rangeIndex + 1} description`}
                  value={range.description}
                  onChange={(event) =>
                    onRangeChange(file.id, range.id, 'description', event.target.value)
                  }
                />
              </label>
              <label className={styles.rangeSortLabel}>
                Sort
                <select
                  aria-label={`Sort position for range ${rangeIndex + 1}`}
                  disabled={!isViewFileRangeSortable(range)}
                  value={currentSortPosition || ''}
                  onChange={(event) =>
                    onSortRange(file.id, range.id, Number(event.currentTarget.value))
                  }
                >
                  <option value="">Top</option>
                  {file.ranges.map((_, optionIndex) => (
                    <option key={optionIndex + 1} value={optionIndex + 1}>
                      Top {optionIndex + 1}
                    </option>
                  ))}
                </select>
              </label>
              <button
                type="button"
                className={styles.iconButton}
                aria-label={`Remove range ${rangeIndex + 1}`}
                onClick={() => onRemoveRange(file.id, range.id)}
              >
                x
              </button>
            </div>
          )
        })}
        {rangeInvalid && (
          <strong role="alert">Start must be less than or equal to end.</strong>
        )}
      </div>
      <div className={styles.frameTools}>
        <label className={styles.videoFilterLabel}>
          Video filter
          <input
            type="search"
            aria-label="Video filter"
            value={file.videoFilter}
            onChange={(event) => onVideoFilterChange(file.id, event.target.value)}
            placeholder="L21_V001"
          />
        </label>
        {normalizedVideoFilter && (
          <span className={styles.filterMatchCount}>{videoFilterMatchCount} matched</span>
        )}
        <form className={styles.addFrameForm} aria-label="Add frame" onSubmit={handleAddFrame}>
          <input
            type="text"
            aria-label="New frame video ID"
            value={newVideoId}
            onChange={(event) => setNewVideoId(event.target.value)}
            placeholder="video_id"
          />
          <input
            type="number"
            aria-label="New frame index"
            value={newFrameIdx}
            onChange={(event) => setNewFrameIdx(event.target.value)}
            placeholder="frame"
          />
          {parsed.kind === 'vqa' && (
            <input
              type="text"
              aria-label="New frame answer"
              value={newAnswer}
              onChange={(event) => setNewAnswer(event.target.value)}
              placeholder="answer"
            />
          )}
          <button type="submit" className={styles.secondaryButton}>
            Add Frame
          </button>
        </form>
        {addFrameError && (
          <strong className={styles.inlineError} role="alert">
            {addFrameError}
          </strong>
        )}
      </div>

      <FrameGrid
        fileId={file.id}
        rows={rankedRows}
        ranges={file.ranges}
        rangeVideoId={file.rangeVideoId}
        videoFilter={file.videoFilter}
        onRemoveFrame={onRemoveFrame}
        onOpenFrame={onOpenFrame}
        onReorderRow={onReorderRow}
      />
    </section>
  )
}

const FrameGrid = memo(function FrameGrid({
  fileId,
  rows,
  ranges,
  rangeVideoId,
  videoFilter,
  onRemoveFrame,
  onOpenFrame,
  onReorderRow,
}: {
  fileId: string
  rows: RankedViewFileRow[]
  ranges: ViewFileRange[]
  rangeVideoId: string
  videoFilter: string
  onRemoveFrame: (fileId: string, rowId: string) => void
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
  const normalizedVideoFilter = normalizeVideoFilter(videoFilter)

  return (
    <div className={styles.frameGrid}>
      {rows.length === 0 && <p className={styles.emptyGrid}>No frames in this file.</p>}
      {rows.map(({ row, rank }) => {
        const matchingRanges = getMatchingRanges(row, ranges, rangeVideoId)
        const videoFilterMatched = normalizedVideoFilter
          ? videoMatchesFilter(row.videoId, normalizedVideoFilter)
          : false
        const className = [
          styles.frameItem,
          normalizedVideoFilter && !videoFilterMatched ? styles.frameItemDimmed : '',
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
            <button
              type="button"
              className={styles.deleteFrameButton}
              aria-label={`Delete ${row.videoId} frame ${row.frameIdx}`}
              onClick={() => onRemoveFrame(fileId, row.id)}
            >
              x
            </button>
            <ViewFileFrameCard
              row={row}
              rank={rank}
              matchingRanges={matchingRanges}
              videoFilterMatched={videoFilterMatched}
              onOpenFrame={onOpenFrame}
            />
          </div>
        )
      })}
    </div>
  )
})

const ViewFileFrameCard = memo(function ViewFileFrameCard({
  row,
  rank,
  matchingRanges,
  videoFilterMatched,
  onOpenFrame,
}: {
  row: ViewFileCsvRow
  rank: number
  matchingRanges: { range: ViewFileRange; index: number }[]
  videoFilterMatched: boolean
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
  const firstRange = matchingRanges[0]
  const rangeColor = firstRange ? RANGE_COLORS[(firstRange.index - 1) % RANGE_COLORS.length] : null
  const className = [
    styles.frameCard,
    state.kind === 'local' ? styles.frameCardLocal : '',
    state.kind === 'youtube' ? styles.frameCardYoutube : '',
    state.kind === 'unavailable' || state.kind === 'error' ? styles.frameCardUnavailable : '',
    matchingRanges.length > 0 ? styles.frameCardInRange : '',
  ]
    .filter(Boolean)
    .join(' ')

  return (
    <button
      type="button"
      className={className}
      style={
        rangeColor
          ? ({ '--range-color': rangeColor } as CSSProperties)
          : undefined
      }
      aria-label={`Open ${row.videoId} frame ${row.frameIdx}`}
      disabled={isLoading}
      onClick={() => onOpenFrame(row, rank, resolution)}
    >
      <span className={styles.rankBadge}>#{rank}</span>
      {videoFilterMatched && (
        <span className={styles.videoMatchBadge} aria-label="Video filter match">
          ✓
        </span>
      )}
      {matchingRanges.length > 0 && (
        <span className={styles.rangeBadges}>
          {matchingRanges.map(({ range, index }) => {
            const color = RANGE_COLORS[(index - 1) % RANGE_COLORS.length]
            return (
              <span
                key={range.id}
                className={styles.rangeBadge}
                style={{ '--range-color': color } as CSSProperties}
                title={range.description || `Range ${index}`}
              >
                R{index}
              </span>
            )
          })}
        </span>
      )}
      {row.answer && (
        <span className={styles.answerBadge} title={row.answer}>
          {row.answer}
        </span>
      )}
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
      if (file.parsed?.kind === 'vqa') cells.push(quoteCsvCell(row.answer ?? '', true))
      return cells.join(',')
    })
    .join('\n')
}

function quoteCsvCell(value: string, always = false): string {
  if (!always && !/[",\r\n]/.test(value)) return value
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
  return file.ranges.some(isViewFileRangeInvalid)
}

function isViewFileRangeInvalid(range: ViewFileRange): boolean {
  const start = parseRangeBoundary(range.startFrame)
  const end = parseRangeBoundary(range.endFrame)
  return start !== null && end !== null && start > end
}

function isViewFileRangeSortable(range: ViewFileRange): boolean {
  const start = parseRangeBoundary(range.startFrame)
  const end = parseRangeBoundary(range.endFrame)
  return start !== null && end !== null && start <= end
}

function formatRangeSummary(file: UploadedViewFile): string | null {
  if (file.ranges.length === 0) return null
  const videoPrefix = file.rangeVideoId.trim() ? `${file.rangeVideoId.trim()} ` : ''
  if (file.ranges.length > 1) return `${videoPrefix}${file.ranges.length} ranges`
  const range = file.ranges[0]
  const start = range.startFrame.trim() || '--'
  const end = range.endFrame.trim() || '--'
  return `${videoPrefix}Range ${start} -> ${end}`
}

function getMatchingRanges(
  row: ViewFileCsvRow,
  ranges: ViewFileRange[],
  rangeVideoId: string,
): { range: ViewFileRange; index: number }[] {
  return ranges.flatMap((range, index) => {
    return rangeMatchesRow(range, row, rangeVideoId) ? [{ range, index: index + 1 }] : []
  })
}

function orderRowsForSortedRanges(
  file: UploadedViewFile,
  sortedRangeIds: string[],
  rangeVideoId = file.rangeVideoId,
): string[] {
  if (!file.parsed) return file.rowOrder
  const currentRows = orderRowsByIds(file.parsed.rows, file.rowOrder)
  const claimedRowIds = new Set<string>()
  const sortedRows: ViewFileCsvRow[] = []

  for (const sortedRangeId of sortedRangeIds) {
    const range = file.ranges.find((candidate) => candidate.id === sortedRangeId)
    if (!range || !isViewFileRangeSortable(range)) continue
    const rowsInRange = currentRows
      .filter((row) => !claimedRowIds.has(row.id) && rangeMatchesRow(range, row, rangeVideoId))
      .sort(compareRowsByFrame)
    for (const row of rowsInRange) {
      claimedRowIds.add(row.id)
      sortedRows.push(row)
    }
  }

  const remainingRows = currentRows.filter((row) => !claimedRowIds.has(row.id))
  return [...sortedRows, ...remainingRows].map((row) => row.id)
}

function rangeMatchesRow(
  range: ViewFileRange,
  row: ViewFileCsvRow,
  rangeVideoId: string,
): boolean {
  const normalizedRangeVideoId = normalizeVideoFilter(rangeVideoId)
  if (normalizedRangeVideoId && row.videoId.toLowerCase() !== normalizedRangeVideoId) {
    return false
  }
  return rangeContainsFrame(range, row.frameIdx)
}

function rangeContainsFrame(range: ViewFileRange, frameIdx: number): boolean {
  const start = parseRangeBoundary(range.startFrame)
  const end = parseRangeBoundary(range.endFrame)
  if (start === null || end === null || start > end) return false
  return frameIdx >= start && frameIdx <= end
}

function compareRowsByFrame(a: ViewFileCsvRow, b: ViewFileCsvRow): number {
  if (a.frameIdx !== b.frameIdx) return a.frameIdx - b.frameIdx
  const videoOrder = a.videoId.localeCompare(b.videoId, undefined, {
    numeric: true,
    sensitivity: 'base',
  })
  if (videoOrder !== 0) return videoOrder
  return a.rowNumber - b.rowNumber
}

function normalizeVideoFilter(value: string): string {
  return value.trim().toLowerCase()
}

function videoMatchesFilter(videoId: string, normalizedFilter: string): boolean {
  return videoId.toLowerCase().includes(normalizedFilter)
}
