import type { SearchHit, TrakeVideoResultItem } from '../api/types'

export type ExportTask = 'kis' | 'qa' | 'trake'

export interface SelectedSearchItem {
  key: string
  hit: SearchHit
  originalRank: number
}

export interface SelectedKisItem extends SelectedSearchItem {
  source: 'manual' | 'visual'
  step: number
  margin: number
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
  margin?: number
  marginCount?: number
  maxFrameIdx?: number
  maxRows?: number
}

export interface KisSubmissionPreviewGroup {
  item: SelectedKisItem
  marginStart: number | null
  marginEnd: number | null
  emittedCount: number
}

export interface KisSubmissionPlan {
  rows: SearchHit[]
  groups: KisSubmissionPreviewGroup[]
  rankingFallbackCount: number
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

export function createManualKisSelection(
  videoId: string,
  frameIdx: number,
  step: number,
  margin: number,
): SelectedKisItem {
  const normalizedVideoId = videoId.trim()
  const normalizedFrameIdx = Math.trunc(frameIdx)
  const key = `manual:${normalizedVideoId}:${normalizedFrameIdx}`
  return {
    key,
    source: 'manual',
    step: normalizeSweepStep(step),
    margin: normalizeSweepMargin(margin),
    originalRank: -1,
    hit: {
      keyframe_id: key,
      video_id: normalizedVideoId,
      score: 0,
      frame_idx: normalizedFrameIdx,
      timestamp: 0,
      thumbnail_path: '',
      thumbnail_url: '',
      image_url: '',
    },
  }
}

export function createVisualKisSelection(
  hit: SearchHit,
  originalRank: number,
  step: number,
  margin: number,
): SelectedKisItem {
  return {
    key: searchHitKey(hit),
    source: 'visual',
    hit,
    originalRank,
    step: normalizeSweepStep(step),
    margin: normalizeSweepMargin(margin),
  }
}

export function buildKisSubmissionPlan(
  selectionGroups: SelectedKisItem[],
  ranking: SearchHit[],
  maxRows = MAX_SUBMISSION_ROWS,
): KisSubmissionPlan {
  const rowLimit = Math.max(1, Math.trunc(maxRows))
  const seen = new Set<string>()
  const rows: SearchHit[] = []
  const groups: KisSubmissionPreviewGroup[] = []

  for (const item of selectionGroups) {
    if (rows.length >= rowLimit) {
      groups.push({ item, marginStart: null, marginEnd: null, emittedCount: 0 })
      continue
    }
    const before = rows.length
    addSubmissionRow(rows, seen, item.hit)
    if (rows.length >= rowLimit) {
      groups.push({
        item,
        ...marginBoundsForSelection(item),
        emittedCount: rows.length - before,
      })
      continue
    }

    const marginFrames = buildRawFrameSweep(item.hit.frame_idx, {
      step: item.step,
      margin: item.margin,
      maxRows: rowLimit,
    }).filter((frame) => frame !== Math.trunc(item.hit.frame_idx))

    for (const frame of marginFrames) {
      addSubmissionRow(rows, seen, rawFrameForSelection(item, frame))
      if (rows.length >= rowLimit) break
    }
    groups.push({
      item,
      ...marginBoundsForSelection(item),
      emittedCount: rows.length - before,
    })
  }

  const beforeFallback = rows.length
  if (rows.length < rowLimit) {
    for (const candidate of ranking) {
      addSubmissionRow(rows, seen, candidate)
      if (rows.length >= rowLimit) break
    }
  }

  return {
    rows,
    groups,
    rankingFallbackCount: rows.length - beforeFallback,
  }
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

export interface TrakeSubmissionPlan {
  rows: TrakeVideoResultItem[]
  centerCount: number
  variantCount: number
  rankingFallbackCount: number
}

// Strict temporal ordering (τ < t, CLAUDE.md §3/§4): a sequence's frame_idx must
// stay strictly increasing. Returns null when shifting event `eventIndex` to
// `frameIdx` would break that -- the caller drops the variant rather than
// submitting an out-of-order sequence.
function shiftTrakeEventFrame(
  result: TrakeVideoResultItem,
  eventIndex: number,
  frameIdx: number,
): TrakeVideoResultItem | null {
  if (frameIdx < 0) return null
  const keyframes = result.keyframes.map((keyframe, index) =>
    index === eventIndex
      ? {
          ...keyframe,
          frame_idx: frameIdx,
          // Keep the React key / dedupe key distinct from the original frame.
          keyframe_id: `${keyframe.keyframe_id}:sweep:${frameIdx}`,
        }
      : keyframe,
  )
  for (let i = 1; i < keyframes.length; i += 1) {
    if (keyframes[i].frame_idx <= keyframes[i - 1].frame_idx) return null
  }
  return { ...result, keyframes }
}

// Per-event margin-sweep: keep the center sequence, then for EACH event emit
// variants that shift ONLY that event's frame_idx by ±{step…margin}, others
// fixed. Covers the common TRAKE miss "N−1 events right, one event a few frames
// off" without the combinatorial blow-up of sweeping every event at once. The
// graded window is <10 frames (CLAUDE.md §1.2), so a single keyframe frame_idx
// per event rarely lands in it -- this is the hedge.
export function buildTrakePerEventSweep(
  result: TrakeVideoResultItem,
  options: { step: number; margin: number },
): TrakeVideoResultItem[] {
  const step = Math.max(1, Math.trunc(options.step))
  const margin = Math.max(0, Math.trunc(options.margin))
  if (margin <= 0) return []
  const seen = new Set<string>([trakeSequenceKey(result)])
  const variants: TrakeVideoResultItem[] = []
  result.keyframes.forEach((keyframe, eventIndex) => {
    const center = Math.trunc(keyframe.frame_idx)
    for (const frame of buildRawFrameSweep(center, { step, margin })) {
      if (frame === center) continue
      const variant = shiftTrakeEventFrame(result, eventIndex, frame)
      if (!variant) continue
      const key = trakeSequenceKey(variant)
      if (seen.has(key)) continue
      seen.add(key)
      variants.push(variant)
    }
  })
  return variants
}

// Lay out the final TRAKE submission (≤100 rows): center sequences of every
// selected sequence first (distinct-video coverage → R@1/R@5), then per-event
// variants round-robin across those sequences (→ R@50/R@100), then the
// remaining ranked sequences as fallback. Everything deduped by
// `trakeSequenceKey`, capped at `maxRows`.
export function buildTrakeSubmissionPlan(
  selected: SelectedTrakeSequence[],
  ranking: TrakeVideoResultItem[],
  options: { step: number; margin: number; maxRows?: number },
): TrakeSubmissionPlan {
  const maxRows = Math.max(1, Math.trunc(options.maxRows ?? MAX_SUBMISSION_ROWS))
  const rows: TrakeVideoResultItem[] = []
  const seen = new Set<string>()
  const add = (item: TrakeVideoResultItem): boolean => {
    if (rows.length >= maxRows) return false
    const key = trakeSequenceKey(item)
    if (seen.has(key)) return false
    seen.add(key)
    rows.push(item)
    return true
  }

  const resultByKey = new Map(ranking.map((result) => [trakeSequenceKey(result), result]))
  const centers: TrakeVideoResultItem[] = []
  const centerSeen = new Set<string>()
  for (const item of selected) {
    const resolved = resultByKey.get(item.key) ?? item.result
    const key = trakeSequenceKey(resolved)
    if (centerSeen.has(key)) continue
    centerSeen.add(key)
    centers.push(resolved)
  }

  let centerCount = 0
  for (const center of centers) {
    if (add(center)) centerCount += 1
    if (rows.length >= maxRows) break
  }

  const variantLists = centers.map((center) =>
    buildTrakePerEventSweep(center, { step: options.step, margin: options.margin }),
  )
  let variantCount = 0
  const maxDepth = variantLists.reduce((max, list) => Math.max(max, list.length), 0)
  for (let depth = 0; depth < maxDepth && rows.length < maxRows; depth += 1) {
    for (const list of variantLists) {
      if (rows.length >= maxRows) break
      if (depth < list.length && add(list[depth])) variantCount += 1
    }
  }

  const beforeFallback = rows.length
  for (const candidate of ranking) {
    if (rows.length >= maxRows) break
    add(candidate)
  }

  return { rows, centerCount, variantCount, rankingFallbackCount: rows.length - beforeFallback }
}

export function buildRawFrameSweep(
  centerFrameIdx: number,
  options: RawFrameSweepOptions,
): number[] {
  const center = Math.trunc(centerFrameIdx)
  const step = Math.max(1, Math.trunc(options.step))
  const margin = Math.max(0, Math.trunc(options.margin ?? (options.marginCount ?? 0) * step))
  const maxRows = Math.max(1, Math.trunc(options.maxRows ?? Number.MAX_SAFE_INTEGER))
  const frames: number[] = []
  pushFrame(frames, center, options.maxFrameIdx)
  for (let distance = step; distance <= margin && frames.length < maxRows; distance += step) {
    pushFrame(frames, center - distance, options.maxFrameIdx)
    if (frames.length >= maxRows) break
    pushFrame(frames, center + distance, options.maxFrameIdx)
  }
  return Array.from(new Set(frames)).slice(0, maxRows)
}

export function buildKisSubmissionRows(
  selectionGroups: SelectedKisItem[],
  ranking: SearchHit[],
  maxRows = MAX_SUBMISSION_ROWS,
): SearchHit[] {
  return buildKisSubmissionPlan(selectionGroups, ranking, maxRows).rows
}

export function serializeKisCsv(results: SearchHit[]): string {
  return results
    .map((hit) => `${hit.video_id},${Math.trunc(hit.frame_idx)}`)
    .join('\n')
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

function pushFrame(frames: number[], frame: number, maxFrameIdx?: number): void {
  if (frame < 0) return
  if (maxFrameIdx !== undefined && frame > maxFrameIdx) return
  frames.push(frame)
}

function addSubmissionRow(rows: SearchHit[], seen: Set<string>, hit: SearchHit): void {
  const key = searchHitKey(hit)
  if (seen.has(key)) return
  seen.add(key)
  rows.push(hit)
}

function rawFrameForSelection(item: SelectedKisItem, frame: number): SearchHit {
  return {
    ...item.hit,
    frame_idx: frame,
    keyframe_id:
      frame === item.hit.frame_idx ? item.hit.keyframe_id : `${item.hit.keyframe_id}:raw:${frame}`,
  }
}

function marginBoundsForSelection(
  item: SelectedKisItem,
): Pick<KisSubmissionPreviewGroup, 'marginStart' | 'marginEnd'> {
  if (item.margin <= 0) return { marginStart: null, marginEnd: null }
  const frames = buildRawFrameSweep(item.hit.frame_idx, {
    step: item.step,
    margin: item.margin,
  })
  const marginFrames = frames.filter((frame) => frame !== Math.trunc(item.hit.frame_idx))
  if (marginFrames.length === 0) return { marginStart: null, marginEnd: null }
  return {
    marginStart: Math.min(...marginFrames),
    marginEnd: Math.max(...marginFrames),
  }
}

function normalizeSweepStep(step: number): number {
  return Math.max(1, Math.trunc(step))
}

function normalizeSweepMargin(margin: number): number {
  return Math.max(0, Math.trunc(margin))
}
