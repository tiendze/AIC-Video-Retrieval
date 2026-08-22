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
  matchCount: number
  rrfScore: number
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

  const groups = Array.from(groupsByVideo.entries()).map(([videoId, frames]) => {
    const sortedFrames = [...frames].sort((a, b) => a.rank - b.rank)
    return {
      videoId,
      bestRank: sortedFrames[0]?.rank ?? Number.MAX_SAFE_INTEGER,
      matchCount: sortedFrames.length,
      rrfScore: videoRrfScore(sortedFrames, rrfK),
      frames: sortedFrames,
    }
  })

  return groups.sort((left, right) => {
    if (sortMode === 'rrf') {
      const rrfDelta = right.rrfScore - left.rrfScore
      if (rrfDelta !== 0) return rrfDelta
    }
    return left.bestRank - right.bestRank || left.videoId.localeCompare(right.videoId)
  })
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
