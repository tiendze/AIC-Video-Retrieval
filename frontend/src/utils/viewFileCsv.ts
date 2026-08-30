export type ViewFileCsvKind = 'kis' | 'vqa'

export interface ViewFileCsvRow {
  id: string
  rowNumber: number
  videoId: string
  frameIdx: number
  answer: string | null
}

export interface ParsedViewFileCsv {
  kind: ViewFileCsvKind
  rows: ViewFileCsvRow[]
}

export type ViewFileCsvParseResult =
  | { ok: true; parsed: ParsedViewFileCsv }
  | { ok: false; error: string }

const VIDEO_COLUMNS = new Set(['video_id', 'videoid', 'video'])
const FRAME_COLUMNS = new Set(['frame_idx', 'frame_index', 'frameid', 'frame'])
const ANSWER_COLUMNS = new Set(['answer', 'answers', 'ans'])

export function parseViewFileCsv(fileName: string, text: string): ViewFileCsvParseResult {
  const csvRows = parseCsvRows(text).filter((row) => row.some((cell) => cell.trim().length > 0))
  if (csvRows[0]?.[0] === '__PARSE_ERROR_UNCLOSED_QUOTE__') {
    return { ok: false, error: 'CSV has an unclosed quoted value.' }
  }
  if (csvRows.length === 0) {
    return { ok: false, error: 'CSV is empty.' }
  }

  const firstRow = csvRows[0]
  const headerMap = detectHeader(firstRow)
  const hasHeader = headerMap.hasKnownColumn
  const kind: ViewFileCsvKind =
    (hasHeader && headerMap.answerIndex !== -1) || (!hasHeader && firstRow.length >= 3)
      ? 'vqa'
      : 'kis'
  const videoIndex = hasHeader ? headerMap.videoIndex : 0
  const frameIndex = hasHeader ? headerMap.frameIndex : 1
  const answerIndex = hasHeader ? headerMap.answerIndex : kind === 'vqa' ? 2 : -1

  if (videoIndex === -1 || frameIndex === -1) {
    return {
      ok: false,
      error: 'CSV must include video_id and frame_idx columns.',
    }
  }
  if (kind === 'vqa' && answerIndex === -1) {
    return { ok: false, error: 'VQA CSV must include an answer column.' }
  }

  const dataRows = hasHeader ? csvRows.slice(1) : csvRows
  if (dataRows.length === 0) {
    return { ok: false, error: 'CSV has no data rows.' }
  }

  const rows: ViewFileCsvRow[] = []
  for (let index = 0; index < dataRows.length; index += 1) {
    const row = dataRows[index]
    const rowNumber = index + (hasHeader ? 2 : 1)
    const videoId = (row[videoIndex] ?? '').trim()
    const frameText = (row[frameIndex] ?? '').trim()
    const answer = kind === 'vqa' ? (row[answerIndex] ?? '').trim() : null

    if (!videoId) {
      return { ok: false, error: `Row ${rowNumber} is missing video_id.` }
    }
    if (!/^\d+$/.test(frameText)) {
      return { ok: false, error: `Row ${rowNumber} has invalid frame_idx.` }
    }
    if (kind === 'vqa' && !answer) {
      return { ok: false, error: `Row ${rowNumber} is missing answer.` }
    }

    const frameIdx = Number(frameText)
    rows.push({
      id: `${fileName}:${rowNumber}:${videoId}:${frameIdx}`,
      rowNumber,
      videoId,
      frameIdx,
      answer,
    })
  }

  return { ok: true, parsed: { kind, rows } }
}

function parseCsvRows(text: string): string[][] {
  const rows: string[][] = []
  let row: string[] = []
  let cell = ''
  let inQuotes = false

  for (let index = 0; index < text.length; index += 1) {
    const char = text[index]
    if (inQuotes) {
      if (char === '"' && text[index + 1] === '"') {
        cell += '"'
        index += 1
      } else if (char === '"') {
        inQuotes = false
      } else {
        cell += char
      }
      continue
    }

    if (char === '"') {
      inQuotes = true
    } else if (char === ',') {
      row.push(cell)
      cell = ''
    } else if (char === '\n' || char === '\r') {
      row.push(cell)
      rows.push(row)
      row = []
      cell = ''
      if (char === '\r' && text[index + 1] === '\n') {
        index += 1
      }
    } else {
      cell += char
    }
  }

  if (inQuotes) {
    return [[`__PARSE_ERROR_UNCLOSED_QUOTE__`]]
  }
  if (cell.length > 0 || row.length > 0) {
    row.push(cell)
    rows.push(row)
  }
  return rows
}

function detectHeader(row: string[]): {
  hasKnownColumn: boolean
  videoIndex: number
  frameIndex: number
  answerIndex: number
} {
  const normalized = row.map(normalizeColumnName)
  const videoIndex = normalized.findIndex((name) => VIDEO_COLUMNS.has(name))
  const frameIndex = normalized.findIndex((name) => FRAME_COLUMNS.has(name))
  const answerIndex = normalized.findIndex((name) => ANSWER_COLUMNS.has(name))
  return {
    hasKnownColumn: videoIndex !== -1 || frameIndex !== -1 || answerIndex !== -1,
    videoIndex,
    frameIndex,
    answerIndex,
  }
}

function normalizeColumnName(value: string): string {
  return value
    .replace(/^\uFEFF/, '')
    .trim()
    .toLowerCase()
    .replace(/[\s-]+/g, '_')
    .replace(/[^a-z0-9_]/g, '')
}
