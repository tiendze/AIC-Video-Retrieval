import type { SearchHit, TrakeVideoResultItem } from '../api/types'

export type ExportTask = 'kis' | 'qa' | 'trake'

export interface SelectedSearchItem {
  key: string
  hit: SearchHit
  originalRank: number
}

export interface SelectedVqaItem extends SelectedSearchItem {
  answer: string
}

export interface SelectedTrakeSequence {
  key: string
  result: TrakeVideoResultItem
  originalRank: number
}

export interface FilenameValidation {
  filename: string
  isValid: boolean
  message: string | null
}

export interface RawFrameSweepOptions {
  step: number
  marginCount: number
  maxFrameIdx?: number
}

const TASK_SUFFIX: Record<ExportTask, string> = {
  kis: '-kis',
  qa: '-qa',
  trake: '-trake',
}

const MAX_SUBMISSION_ROWS = 100

export function searchHitKey(hit: SearchHit): string {
  return `${hit.video_id}:${hit.frame_idx}`
}

export function trakeSequenceKey(result: TrakeVideoResultItem): string {
  return `${result.video_id}:${result.keyframes.map((keyframe) => keyframe.frame_idx).join(':')}`
}

export function normalizeCsvFilename(input: string): string {
  const trimmed = input.trim()
  if (trimmed.toLowerCase().endsWith('.csv')) {
    return trimmed
  }
  return `${trimmed}.csv`
}

export function validateCsvFilename(input: string, task: ExportTask): FilenameValidation {
  const baseName = input.trim().replace(/\.csv$/i, '')
  const filename = normalizeCsvFilename(input)
  if (baseName.length === 0) {
    return { filename, isValid: false, message: 'Enter an export name.' }
  }
  const suffix = TASK_SUFFIX[task]
  if (!baseName.endsWith(suffix)) {
    return {
      filename,
      isValid: false,
      message: `Filename should end with ${suffix}.csv for this task.`,
    }
  }
  return { filename, isValid: true, message: null }
}

export function suggestedFilename(baseName: string, task: ExportTask): string {
  const stripped = baseName.trim().replace(/\.csv$/i, '').replace(/-(kis|qa|trake)$/i, '')
  return normalizeCsvFilename(`${stripped || 'query-1'}${TASK_SUFFIX[task]}`)
}

export function buildKisFinalRanking(
  results: SearchHit[],
  selectedItems: SelectedSearchItem[],
): SearchHit[] {
  const resultByKey = new Map(results.map((hit) => [searchHitKey(hit), hit]))
  const selectedKeys = new Set<string>()
  const selected = selectedItems
    .map((item) => resultByKey.get(item.key) ?? item.hit)
    .filter((hit) => {
      const key = searchHitKey(hit)
      if (selectedKeys.has(key)) return false
      selectedKeys.add(key)
      return true
    })
  const remaining = results.filter((hit) => !selectedKeys.has(searchHitKey(hit)))
  return [...selected, ...remaining].slice(0, MAX_SUBMISSION_ROWS)
}

export function buildVqaFinalRows(selectedItems: SelectedVqaItem[]): SelectedVqaItem[] {
  return selectedItems
    .filter((item) => item.answer.length > 0 && item.answer.length <= 100)
    .slice(0, MAX_SUBMISSION_ROWS)
}

export function buildTrakeFinalRanking(
  results: TrakeVideoResultItem[],
  selectedItems: SelectedTrakeSequence[],
): TrakeVideoResultItem[] {
  const resultByKey = new Map(results.map((result) => [trakeSequenceKey(result), result]))
  const selectedKeys = new Set<string>()
  const selected = selectedItems
    .map((item) => resultByKey.get(item.key) ?? item.result)
    .filter((result) => {
      const key = trakeSequenceKey(result)
      if (selectedKeys.has(key)) return false
      selectedKeys.add(key)
      return true
    })
  const remaining = results.filter((result) => !selectedKeys.has(trakeSequenceKey(result)))
  return [...selected, ...remaining].slice(0, MAX_SUBMISSION_ROWS)
}

export function buildRawFrameSweep(
  centerFrameIdx: number,
  options: RawFrameSweepOptions,
): number[] {
  const center = Math.trunc(centerFrameIdx)
  const step = Math.max(1, Math.trunc(options.step))
  const marginCount = Math.max(0, Math.trunc(options.marginCount))
  const frames: number[] = []
  for (let offset = -marginCount; offset <= marginCount; offset += 1) {
    const frame = center + offset * step
    if (frame < 0) continue
    if (options.maxFrameIdx !== undefined && frame > options.maxFrameIdx) continue
    frames.push(frame)
  }
  if (!frames.includes(center) && center >= 0) {
    frames.push(center)
  }
  return Array.from(new Set(frames)).sort((left, right) => left - right)
}

export function serializeKisCsv(
  results: SearchHit[],
  sweepOptions: RawFrameSweepOptions = { step: 1, marginCount: 0 },
): string {
  const seen = new Set<string>()
  const rows: string[] = []
  for (const hit of results) {
    for (const frame of buildRawFrameSweep(hit.frame_idx, sweepOptions)) {
      const key = `${hit.video_id}:${frame}`
      if (seen.has(key)) continue
      seen.add(key)
      rows.push(`${hit.video_id},${frame}`)
    }
  }
  return rows.join('\n')
}

export function serializeVqaCsv(rows: SelectedVqaItem[]): string {
  return rows
    .map((item) => `${item.hit.video_id},${Math.trunc(item.hit.frame_idx)},${quoteCsv(item.answer)}`)
    .join('\n')
}

export function serializeTrakeCsv(results: TrakeVideoResultItem[]): string {
  return results
    .map(
      (result) =>
        `${result.video_id},${result.keyframes
          .map((keyframe) => Math.trunc(keyframe.frame_idx))
          .join(',')}`,
    )
    .join('\n')
}

export function validateTrakeEventCount(
  results: TrakeVideoResultItem[],
  eventCount: number,
): string | null {
  const invalid = results.find((result) => result.keyframes.length !== eventCount)
  if (!invalid) return null
  return `Every TRAKE row must have ${eventCount} event frame${eventCount === 1 ? '' : 's'}.`
}

export function downloadCsv(filename: string, content: string): void {
  const blob = new Blob([content], { type: 'text/csv;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  link.click()
  URL.revokeObjectURL(url)
}

function quoteCsv(value: string): string {
  return `"${value.replace(/"/g, '""')}"`
}
