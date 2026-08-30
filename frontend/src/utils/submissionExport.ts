import type { SearchHit, TrakeVideoResultItem } from '../api/types'

export type ExportTask = 'kis' | 'qa' | 'trake'

export interface SelectedSearchItem {
  key: string
  hit: SearchHit
  originalRank: number
}

export type KisSweepDirection = 'none' | 'left' | 'right'

export interface SelectedKisItem extends SelectedSearchItem {
  source: 'manual' | 'visual'
  sweepDirection: KisSweepDirection
  step: number
  targetFrameIdx: number | null
}

export interface SelectedVqaItem extends SelectedSearchItem {
  source: 'manual' | 'visual'
  sweepDirection: KisSweepDirection
  step: number
  targetFrameIdx: number | null
  answer: string
}

export interface SelectedTrakeSequence {
  key: string
  source: 'manual' | 'visual'
  result: TrakeVideoResultItem
  originalRank: number
  stepTargetFrames: Array<number | null>
  stepCount: number
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
  sweepStart: number | null
  sweepEnd: number | null
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
  return `${result.video_id}::${result.keyframes.map((keyframe) => Math.trunc(keyframe.frame_idx)).join('|')}`
}

export function trakeVisualSelectionKey(
  result: TrakeVideoResultItem,
  originalRank: number,
  instanceId = 'rank',
): string {
  return `visual-trake:${originalRank}:${instanceId}:${trakeSequenceKey(result)}`
}

function defaultTrakeStepTargetFrames(result: TrakeVideoResultItem): number[] {
  return result.keyframes.map((keyframe) => Math.trunc(keyframe.frame_idx))
}

export function createVisualTrakeSelection(
  result: TrakeVideoResultItem,
  originalRank: number,
  selectionKey = trakeVisualSelectionKey(result, originalRank),
): SelectedTrakeSequence {
  return {
    key: selectionKey,
    source: 'visual',
    result,
    originalRank,
    stepTargetFrames: defaultTrakeStepTargetFrames(result),
    stepCount: 5,
  }
}

export function createManualTrakeSelection(
  videoId: string,
  frameIdxs: number[],
): SelectedTrakeSequence {
  const normalizedVideoId = videoId.trim()
  const normalizedFrames = frameIdxs.map((frameIdx) => Math.trunc(frameIdx))
  const result: TrakeVideoResultItem = {
    video_id: normalizedVideoId,
    score: 0,
    keyframes: normalizedFrames.map((frameIdx, index) => ({
      keyframe_id: `manual-trake:${normalizedVideoId}:${normalizedFrames.join(':')}:event:${index}`,
      frame_idx: frameIdx,
      timestamp: 0,
      thumbnail_url: '',
      image_url: '',
    })),
  }
  return {
    key: trakeSequenceKey(result),
    source: 'manual',
    originalRank: -1,
    result,
    stepTargetFrames: defaultTrakeStepTargetFrames(result),
    stepCount: 5,
  }
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
  sweepDirection: KisSweepDirection = 'none',
  step = 1,
  targetFrameIdx: number | null = null,
): SelectedKisItem {
  const normalizedVideoId = videoId.trim()
  const normalizedFrameIdx = Math.trunc(frameIdx)
  const normalizedTarget = targetFrameIdx === null ? null : Math.trunc(targetFrameIdx)
  const key = `manual:${normalizedVideoId}:${normalizedFrameIdx}`
  return {
    key,
    source: 'manual',
    sweepDirection,
    step: normalizeSweepStep(step),
    targetFrameIdx: normalizedTarget,
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
  sweepDirection: KisSweepDirection = 'none',
  step = 1,
  targetFrameIdx: number | null = null,
): SelectedKisItem {
  return {
    key: searchHitKey(hit),
    source: 'visual',
    hit,
    originalRank,
    sweepDirection,
    step: normalizeSweepStep(step),
    targetFrameIdx: targetFrameIdx === null ? null : Math.trunc(targetFrameIdx),
  }
}

export function createManualVqaSelection(
  videoId: string,
  frameIdx: number,
  answer: string,
  sweepDirection: KisSweepDirection = 'none',
  step = 1,
  targetFrameIdx: number | null = null,
): SelectedVqaItem {
  const normalizedVideoId = videoId.trim()
  const normalizedFrameIdx = Math.trunc(frameIdx)
  const key = `manual-vqa:${normalizedVideoId}:${normalizedFrameIdx}`
  return {
    key,
    source: 'manual',
    sweepDirection,
    step: normalizeSweepStep(step),
    targetFrameIdx: targetFrameIdx === null ? null : Math.trunc(targetFrameIdx),
    answer,
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

export function createVisualVqaSelection(hit: SearchHit, originalRank: number): SelectedVqaItem {
  return {
    key: searchHitKey(hit),
    source: 'visual',
    hit,
    originalRank,
    sweepDirection: 'none',
    step: 1,
    targetFrameIdx: null,
    answer: '',
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
      groups.push({ item, sweepStart: null, sweepEnd: null, emittedCount: 0 })
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
      direction: item.sweepDirection,
      targetFrameIdx: item.targetFrameIdx,
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
  const rows: SelectedVqaItem[] = []
  const seen = new Set<string>()
  for (const item of selectedItems) {
    if (item.answer.length === 0 || item.answer.length > 100) continue
    const frames = buildDirectionalFrameSweep(item.hit.frame_idx, {
      direction: item.sweepDirection,
      step: item.step,
      targetFrameIdx: item.targetFrameIdx,
      maxRows: MAX_SUBMISSION_ROWS,
    })
    for (const frame of frames) {
      const next = frame === item.hit.frame_idx ? item : vqaFrameForSelection(item, frame)
      const key = searchHitKey(next.hit)
      if (seen.has(key)) continue
      seen.add(key)
      rows.push(next)
      if (rows.length >= MAX_SUBMISSION_ROWS) return rows
    }
  }
  return rows
}

export function buildTrakeFinalRanking(
  results: TrakeVideoResultItem[],
  selectedItems: SelectedTrakeSequence[],
): TrakeVideoResultItem[] {
  const resultByKey = new Map(results.map((result) => [trakeSequenceKey(result), result]))
  const selectedKeys = new Set<string>()
  const selected = selectedItems
    .map((item) => resultByKey.get(trakeSequenceKey(item.result)) ?? item.result)
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
  entries: TrakeSubmissionPlanEntry[]
  centerCount: number
  variantCount: number
  rankingFallbackCount: number
}

export interface TrakeSubmissionPlanEntry {
  row: TrakeVideoResultItem
  label: string
  selectedKey: string | null
  stepIndex: number | null
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

export function buildTrakeSynchronizedSteps(
  result: TrakeVideoResultItem,
  options: {
    targetFrames: Array<number | null>
    stepCount: number
  },
): TrakeVideoResultItem[] {
  const stepCount = Math.max(1, Math.trunc(options.stepCount || 1))
  const starts = result.keyframes.map((keyframe) => Math.trunc(keyframe.frame_idx))
  const targets = result.keyframes.map((_, index) =>
    Math.max(0, Math.trunc(options.targetFrames[index] ?? starts[index])),
  )
  if (targets.every((target, index) => target === starts[index])) return []

  const variants: TrakeVideoResultItem[] = []
  const seen = new Set<string>([trakeSequenceKey(result)])
  for (let iteration = 1; iteration <= stepCount; iteration += 1) {
    const keyframes = result.keyframes.map((keyframe, index) => {
      const frameIdx = interpolateClampedFrame(starts[index], targets[index], iteration, stepCount)
      return {
        ...keyframe,
        frame_idx: frameIdx,
        keyframe_id: `${keyframe.keyframe_id}:sync:${iteration}:${frameIdx}`,
      }
    })
    const variant = { ...result, keyframes }
    const key = trakeSequenceKey(variant)
    if (seen.has(key)) continue
    seen.add(key)
    variants.push(variant)
  }
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
  options: { maxRows?: number } = {},
): TrakeSubmissionPlan {
  const maxRows = Math.max(1, Math.trunc(options.maxRows ?? MAX_SUBMISSION_ROWS))
  const rows: TrakeVideoResultItem[] = []
  const entries: TrakeSubmissionPlanEntry[] = []
  const seen = new Set<string>()
  const add = (
    item: TrakeVideoResultItem,
    label: string,
    selectedKey: string | null,
    stepIndex: number | null,
  ): boolean => {
    if (rows.length >= maxRows) return false
    const key = trakeSequenceKey(item)
    if (seen.has(key)) return false
    seen.add(key)
    rows.push(item)
    entries.push({ row: item, label, selectedKey, stepIndex })
    return true
  }

  const resultByKey = new Map(ranking.map((result) => [trakeSequenceKey(result), result]))
  const centerSeen = new Set<string>()
  let centerCount = 0
  let variantCount = 0
  for (const item of selected) {
    const resolved = resultByKey.get(trakeSequenceKey(item.result)) ?? item.result
    const key = trakeSequenceKey(resolved)
    if (centerSeen.has(key)) continue
    centerSeen.add(key)
    if (add(resolved, 'Original', item.key, null)) centerCount += 1
    if (rows.length >= maxRows) break
    const synchronizedSteps = buildTrakeSynchronizedSteps(resolved, {
      targetFrames: item.stepTargetFrames,
      stepCount: item.stepCount,
    })
    for (const [variantIndex, variant] of synchronizedSteps.entries()) {
      if (rows.length >= maxRows) break
      const stepIndex = variantIndex + 1
      if (add(variant, `Step #${stepIndex}`, item.key, stepIndex)) variantCount += 1
    }
  }

  const beforeFallback = rows.length
  for (const candidate of ranking) {
    if (rows.length >= maxRows) break
    add(candidate, 'Fallback', null, null)
  }

  return {
    rows,
    entries,
    centerCount,
    variantCount,
    rankingFallbackCount: rows.length - beforeFallback,
  }
}

export function buildRawFrameSweep(
  centerFrameIdx: number,
  options: RawFrameSweepOptions & {
    direction?: KisSweepDirection
    targetFrameIdx?: number | null
  },
): number[] {
  const center = Math.trunc(centerFrameIdx)
  const step = Math.max(1, Math.trunc(options.step))
  const direction = options.direction ?? 'none'
  if (direction !== 'none') {
    return buildDirectionalFrameSweep(center, {
      direction,
      step,
      targetFrameIdx: options.targetFrameIdx ?? null,
      maxRows: options.maxRows,
    })
  }
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

export function buildDirectionalFrameSweep(
  centerFrameIdx: number,
  options: {
    direction: KisSweepDirection
    step: number
    targetFrameIdx: number | null
    maxRows?: number
  },
): number[] {
  const center = Math.trunc(centerFrameIdx)
  const step = Math.max(1, Math.trunc(options.step))
  const maxRows = Math.max(1, Math.trunc(options.maxRows ?? Number.MAX_SAFE_INTEGER))
  const frames: number[] = []
  pushFrame(frames, center)
  if (options.direction === 'none' || options.targetFrameIdx === null) {
    return frames.slice(0, maxRows)
  }

  const target = Math.trunc(options.targetFrameIdx)
  if (options.direction === 'left' && target >= center) return frames.slice(0, maxRows)
  if (options.direction === 'right' && target <= center) return frames.slice(0, maxRows)
  const shouldAddOvershoot = Math.abs(center - target) % step !== 0
  if (options.direction === 'left') {
    let frame = center - step
    for (; frame >= target && frames.length < maxRows; frame -= step) {
      pushFrame(frames, frame)
    }
    if (shouldAddOvershoot && frame < target && frames.length < maxRows) {
      pushFrame(frames, frame)
    }
  } else {
    let frame = center + step
    for (; frame <= target && frames.length < maxRows; frame += step) {
      pushFrame(frames, frame)
    }
    if (shouldAddOvershoot && frame > target && frames.length < maxRows) {
      pushFrame(frames, frame)
    }
  }
  return Array.from(new Set(frames)).slice(0, maxRows)
}

export function kisSweepReachWarning(
  centerFrameIdx: number,
  direction: KisSweepDirection,
  step: number,
  targetFrameIdx: number | null,
): string | null {
  if (direction === 'none' || targetFrameIdx === null) return null
  const center = Math.trunc(centerFrameIdx)
  const target = Math.trunc(targetFrameIdx)
  const normalizedStep = Math.max(1, Math.trunc(step))
  if (direction === 'left' && target >= center) {
    return 'Target frame must be smaller than the selected frame when sweeping left.'
  }
  if (direction === 'right' && target <= center) {
    return 'Target frame must be larger than the selected frame when sweeping right.'
  }
  const frameCount = directionalSweepFrameCount(center, target, normalizedStep)
  if (frameCount > MAX_SUBMISSION_ROWS) {
    return `Sweep creates ${frameCount} frames; CSV export will keep the first ${MAX_SUBMISSION_ROWS}.`
  }
  return null
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

function directionalSweepFrameCount(center: number, target: number, step: number): number {
  const distance = Math.abs(center - target)
  return 1 + Math.ceil(distance / Math.max(1, step))
}

function interpolateClampedFrame(
  startFrameIdx: number,
  targetFrameIdx: number,
  stepIndex: number,
  stepCount: number,
): number {
  const start = Math.trunc(startFrameIdx)
  const target = Math.trunc(targetFrameIdx)
  const raw = Math.round(start + ((target - start) * stepIndex) / Math.max(1, stepCount))
  if (target >= start) return Math.min(target, Math.max(start, raw))
  return Math.max(target, Math.min(start, raw))
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
): Pick<KisSubmissionPreviewGroup, 'sweepStart' | 'sweepEnd'> {
  const frames = buildRawFrameSweep(item.hit.frame_idx, {
    step: item.step,
    direction: item.sweepDirection,
    targetFrameIdx: item.targetFrameIdx,
  })
  const marginFrames = frames.filter((frame) => frame !== Math.trunc(item.hit.frame_idx))
  if (marginFrames.length === 0) return { sweepStart: null, sweepEnd: null }
  return {
    sweepStart: Math.min(...marginFrames),
    sweepEnd: Math.max(...marginFrames),
  }
}

function vqaFrameForSelection(item: SelectedVqaItem, frame: number): SelectedVqaItem {
  return {
    ...item,
    key: `${item.key}:raw:${frame}`,
    hit: {
      ...item.hit,
      frame_idx: frame,
      keyframe_id:
        frame === item.hit.frame_idx ? item.hit.keyframe_id : `${item.hit.keyframe_id}:raw:${frame}`,
    },
  }
}

function normalizeSweepStep(step: number): number {
  return Math.max(1, Math.trunc(step))
}
