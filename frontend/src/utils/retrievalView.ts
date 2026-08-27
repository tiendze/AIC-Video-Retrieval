import { DEFAULT_RRF_K } from '../api/constants'
import type { SearchHit } from '../api/types'
import { searchHitKey } from './submissionExport'

export type RetrievalViewMode = 'ranked' | 'grouped'
export type ImageRankSortMode = 'local' | 'global'
export type VideoGroupSortMode = 'local' | 'global' | 'rrf'

export interface RankedHit {
  hit: SearchHit
  rank: number
  localRank: number | null
  globalRank: number | null
}

export interface VideoGroup {
  videoId: string
  bestRank: number
  bestLocalRank: number | null
  bestLocalScore: number | null
  bestGlobalRank: number | null
  bestImageRank: number
  bestImageScore: number | null
  matchCount: number
  rrfScore: number
  ocrDetected: boolean
  titleOcrDetected: boolean
  priorityTier: number
  finalVideoRank: number
  frames: RankedHit[]
}

export function parseVideoIdFilter(input: string): string[] {
  return Array.from(
    new Set(
      input
        .split(/[\s,;]+/)
        .map((value) => value.trim())
        .filter(Boolean),
    ),
  )
}

export function filterHitsByVideoIds(hits: SearchHit[], input: string): SearchHit[] {
  const ids = parseVideoIdFilter(input)
  if (ids.length === 0) return hits
  const allowed = new Set(ids)
  return hits.filter((hit) => allowed.has(hit.video_id))
}

export function rankedHits(hits: SearchHit[], rankSourceHits: SearchHit[] = hits): RankedHit[] {
  const localRanks = localRankByHitKey(rankSourceHits)
  const globalRanks = globalRankByHitKey(rankSourceHits)
  return hits.map((hit, index) => {
    const globalRank = globalRanks.get(searchHitKey(hit)) ?? globalRankForHit(hit, index)
    return {
      hit,
      rank: globalRank ?? index + 1,
      localRank: localRanks.get(searchHitKey(hit)) ?? null,
      globalRank,
    }
  })
}

export function rankByHitKey(hits: SearchHit[]): Map<string, number> {
  return new Map(rankedHits(hits).map((item) => [searchHitKey(item.hit), item.rank]))
}

export function hasLocalRanking(hits: SearchHit[]): boolean {
  return hits.some((hit) => hasLocalScore(hit))
}

export function localRankByHitKey(hits: SearchHit[]): Map<string, number> {
  const ranked = hits
    .map((hit, index) => ({ hit, index, globalRank: globalRankForHit(hit, index) }))
    .filter(({ hit }) => hasLocalScore(hit))
    .sort(compareByLocalRank)
  return new Map(ranked.map(({ hit }, index) => [searchHitKey(hit), index + 1]))
}

export function globalRankByHitKey(hits: SearchHit[]): Map<string, number> {
  return new Map(
    hits.flatMap((hit, index) => {
      const globalRank = globalRankForHit(hit, index)
      return globalRank === null ? [] : [[searchHitKey(hit), globalRank]]
    }),
  )
}

export function sortHitsByImageRankMode(
  hits: SearchHit[],
  sortMode: ImageRankSortMode,
): SearchHit[] {
  const indexed = hits.map((hit, index) => ({ hit, index, globalRank: globalRankForHit(hit, index) }))
  if (sortMode === 'local' && indexed.some(({ hit }) => hasLocalScore(hit))) {
    return indexed.sort(compareByLocalRank).map(({ hit }) => hit)
  }
  return indexed.sort(compareByGlobalRank).map(({ hit }) => hit)
}

export function buildVideoGroups(
  hits: SearchHit[],
  sortMode: VideoGroupSortMode,
  rrfK = DEFAULT_RRF_K,
  rankSourceHits: SearchHit[] = hits,
): VideoGroup[] {
  const groupsByVideo = new Map<string, RankedHit[]>()
  const frameSortMode: ImageRankSortMode = sortMode === 'local' ? 'local' : 'global'
  for (const item of rankedHits(hits, rankSourceHits)) {
    const frames = groupsByVideo.get(item.hit.video_id) ?? []
    frames.push(item)
    groupsByVideo.set(item.hit.video_id, frames)
  }
  const ocrActive = rankSourceHits.some((hit) => hit.stage1?.ocr !== undefined)
  const titleOcrActive = rankSourceHits.some((hit) => hit.stage1?.title !== undefined)

  const groups = Array.from(groupsByVideo.entries()).map(([videoId, frames]) => {
    const sortedFrames = [...frames].sort((a, b) =>
      frameSortMode === 'local'
        ? compareRankedHitsByLocal(a, b)
        : compareNullableRanks(a.globalRank, b.globalRank),
    )
    const bestFrame = sortedFrames[0]
    const ocrDetected = frames.some(({ hit }) => hitHasOcrEvidence(hit))
    const titleOcrDetected = frames.some(({ hit }) => hitHasTitleOcrEvidence(hit))
    const localFrames = frames.filter(({ hit }) => hasLocalScore(hit))
    const bestLocalFrame = [...localFrames].sort(compareRankedHitsByLocal)[0]
    const bestGlobalFrame = [...frames].sort((a, b) =>
      compareNullableRanks(a.globalRank, b.globalRank),
    )[0]
    const bestLocalRank = bestLocalFrame?.localRank ?? null
    const bestGlobalRank = bestGlobalFrame?.globalRank ?? null
    const bestImageRank =
      sortMode === 'local' && bestLocalRank !== null
        ? bestLocalRank
        : (bestGlobalRank ?? Number.MAX_SAFE_INTEGER)
    return {
      videoId,
      bestRank: bestImageRank,
      bestLocalRank,
      bestLocalScore: bestLocalFrame?.hit.local_score ?? null,
      bestGlobalRank,
      bestImageRank,
      bestImageScore: bestFrame?.hit.fused_image_score ?? bestFrame?.hit.score ?? null,
      matchCount: sortedFrames.length,
      rrfScore: videoRrfScore(sortedFrames, rrfK),
      ocrDetected,
      titleOcrDetected,
      priorityTier: videoPriorityTier({
        ocrActive,
        titleOcrActive,
        ocrDetected,
        titleOcrDetected,
      }),
      finalVideoRank: 0,
      frames: sortedFrames,
    }
  })

  const sortedGroups = groups.sort((left, right) => {
    const priorityDelta = left.priorityTier - right.priorityTier
    if (priorityDelta !== 0) return priorityDelta
    if (sortMode === 'rrf') {
      const rrfDelta = right.rrfScore - left.rrfScore
      if (rrfDelta !== 0) return rrfDelta
    }
    if (sortMode === 'local') {
      const localDelta = compareNullableRanks(left.bestLocalRank, right.bestLocalRank)
      if (localDelta !== 0) return localDelta
      const scoreDelta = (right.bestLocalScore ?? -Infinity) - (left.bestLocalScore ?? -Infinity)
      if (scoreDelta !== 0) return scoreDelta
    }
    const globalDelta = compareNullableRanks(left.bestGlobalRank, right.bestGlobalRank)
    if (globalDelta !== 0) return globalDelta
    return left.videoId.localeCompare(right.videoId)
  })
  return sortedGroups.map((group, index) => ({ ...group, finalVideoRank: index + 1 }))
}

function globalRankForHit(hit: SearchHit, index: number): number | null {
  if (typeof hit.fused_image_rank === 'number') return hit.fused_image_rank
  if (typeof hit.stage1?.fused_image_rank === 'number') return hit.stage1.fused_image_rank
  return hit.local_retrieval_used ? null : index + 1
}

function hasLocalScore(hit: SearchHit): boolean {
  return typeof hit.local_score === 'number' && Number.isFinite(hit.local_score)
}

function compareByLocalRank(
  left: { hit: SearchHit; index: number; globalRank: number | null },
  right: { hit: SearchHit; index: number; globalRank: number | null },
): number {
  const leftHasLocal = hasLocalScore(left.hit)
  const rightHasLocal = hasLocalScore(right.hit)
  if (leftHasLocal !== rightHasLocal) return leftHasLocal ? -1 : 1
  const localDelta = (right.hit.local_score ?? -Infinity) - (left.hit.local_score ?? -Infinity)
  if (localDelta !== 0) return localDelta
  const globalDelta = compareNullableRanks(left.globalRank, right.globalRank)
  if (globalDelta !== 0) return globalDelta
  return left.index - right.index
}

function compareByGlobalRank(
  left: { hit: SearchHit; index: number; globalRank: number | null },
  right: { hit: SearchHit; index: number; globalRank: number | null },
): number {
  const globalDelta = compareNullableRanks(left.globalRank, right.globalRank)
  if (globalDelta !== 0) return globalDelta
  return left.index - right.index
}

function compareRankedHitsByLocal(left: RankedHit, right: RankedHit): number {
  const localDelta = compareNullableRanks(left.localRank, right.localRank)
  if (localDelta !== 0) return localDelta
  return compareNullableRanks(left.globalRank, right.globalRank)
}

function compareNullableRanks(left: number | null, right: number | null): number {
  if (left !== null && right !== null) return left - right
  if (left !== null) return -1
  if (right !== null) return 1
  return 0
}

function videoRrfScore(frames: RankedHit[], rrfK: number): number {
  const bestRankByQuery = new Map<number, number>()
  for (const { hit } of frames) {
    for (const query of hit.stage1?.text_queries ?? []) {
      if (query.original_rank === null) continue
      const current = bestRankByQuery.get(query.query_index)
      if (current === undefined || query.original_rank < current) {
        bestRankByQuery.set(query.query_index, query.original_rank)
      }
    }
  }
  let score = 0
  for (const rank of bestRankByQuery.values()) {
    score += 1 / (rrfK + rank)
  }
  return score
}

function hitHasOcrEvidence(hit: SearchHit): boolean {
  if (hit.stage1?.source_modalities?.includes('ocr')) return true
  return (hit.stage1?.ocr?.query_score ?? 0) > 0
}

function hitHasTitleOcrEvidence(hit: SearchHit): boolean {
  if (hit.stage1?.source_modalities?.includes('title')) return true
  return (hit.stage1?.title?.title_score ?? 0) > 0
}

function videoPriorityTier({
  ocrActive,
  titleOcrActive,
  ocrDetected,
  titleOcrDetected,
}: {
  ocrActive: boolean
  titleOcrActive: boolean
  ocrDetected: boolean
  titleOcrDetected: boolean
}): number {
  if (ocrActive && titleOcrActive) {
    if (ocrDetected && titleOcrDetected) return 0
    if (ocrDetected || titleOcrDetected) return 1
    return 2
  }
  if (ocrActive) return ocrDetected ? 0 : 1
  if (titleOcrActive) return titleOcrDetected ? 0 : 1
  return 0
}
