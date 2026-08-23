import { DEFAULT_RRF_K } from '../api/constants'
import type { SearchHit } from '../api/types'
import { searchHitKey } from './submissionExport'

export type RetrievalViewMode = 'ranked' | 'grouped'
export type VideoGroupSortMode = 'best' | 'rrf'

export interface RankedHit {
  hit: SearchHit
  rank: number
}

export interface VideoGroup {
  videoId: string
  bestRank: number
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

export function rankedHits(hits: SearchHit[]): RankedHit[] {
  return hits.map((hit, index) => ({
    hit,
    rank: hit.fused_image_rank ?? index + 1,
  }))
}

export function rankByHitKey(hits: SearchHit[]): Map<string, number> {
  return new Map(rankedHits(hits).map((item) => [searchHitKey(item.hit), item.rank]))
}

export function buildVideoGroups(
  hits: SearchHit[],
  sortMode: VideoGroupSortMode,
  rrfK = DEFAULT_RRF_K,
): VideoGroup[] {
  const groupsByVideo = new Map<string, RankedHit[]>()
  for (const item of rankedHits(hits)) {
    const frames = groupsByVideo.get(item.hit.video_id) ?? []
    frames.push(item)
    groupsByVideo.set(item.hit.video_id, frames)
  }
  const ocrActive = hits.some((hit) => hit.stage1?.ocr !== undefined)
  const titleOcrActive = hits.some((hit) => hit.stage1?.title !== undefined)

  const groups = Array.from(groupsByVideo.entries()).map(([videoId, frames]) => {
    const sortedFrames = [...frames].sort((a, b) => a.rank - b.rank)
    const bestFrame = sortedFrames[0]
    const ocrDetected = frames.some(({ hit }) => hitHasOcrEvidence(hit))
    const titleOcrDetected = frames.some(({ hit }) => hitHasTitleOcrEvidence(hit))
    const bestImageRank = bestFrame?.rank ?? Number.MAX_SAFE_INTEGER
    return {
      videoId,
      bestRank: bestImageRank,
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
    return left.bestImageRank - right.bestImageRank || left.videoId.localeCompare(right.videoId)
  })
  return sortedGroups.map((group, index) => ({ ...group, finalVideoRank: index + 1 }))
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
