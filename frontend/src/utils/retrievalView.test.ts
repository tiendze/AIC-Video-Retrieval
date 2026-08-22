import { describe, expect, it } from 'vitest'
import type { SearchHit } from '../api/types'
import {
  buildVideoGroups,
  filterHitsByVideoIds,
  parseVideoIdFilter,
} from './retrievalView'

function hit(videoId: string, frameIdx: number, rank: number): SearchHit {
  return {
    keyframe_id: `${videoId}_${frameIdx}`,
    video_id: videoId,
    score: 1 / rank,
    fused_image_rank: rank,
    frame_idx: frameIdx,
    timestamp: frameIdx / 25,
    thumbnail_path: '',
    thumbnail_url: `/api/keyframes/${videoId}_${frameIdx}/thumbnail`,
    image_url: `/api/keyframes/${videoId}_${frameIdx}/image`,
  }
}

describe('retrievalView utilities', () => {
  it('filters one or many video ids without changing the source hits', () => {
    const hits = [hit('L22_V002', 1, 1), hit('L25_V076', 2, 2), hit('L25_V087', 3, 3)]

    expect(parseVideoIdFilter('L22_V002\nL25_V076, L25_V076')).toEqual([
      'L22_V002',
      'L25_V076',
    ])
    expect(filterHitsByVideoIds(hits, 'L25_V076 L25_V087').map((item) => item.video_id)).toEqual([
      'L25_V076',
      'L25_V087',
    ])
    expect(hits).toHaveLength(3)
  })

  it('builds video groups by best global image rank', () => {
    const groups = buildVideoGroups(
      [hit('A', 10, 1), hit('B', 20, 2), hit('A', 30, 8), hit('A', 40, 31)],
      'best',
      60,
    )

    expect(groups[0]).toMatchObject({ videoId: 'A', bestRank: 1, matchCount: 3 })
    expect(groups[1]).toMatchObject({ videoId: 'B', bestRank: 2, matchCount: 1 })
  })

  it('sorts video groups by cross-query RRF from original per-query ranks', () => {
    const hits: SearchHit[] = [
      withOriginalRanks(hit('A', 10, 2), [
        [0, 14],
        [1, 6],
      ]),
      withOriginalRanks(hit('B', 20, 1), [
        [0, 9],
        [1, 16],
      ]),
      withOriginalRanks(hit('C', 30, 3), [
        [0, 40],
        [1, 1],
      ]),
    ]

    const groups = buildVideoGroups(hits, 'rrf', 60)

    expect(groups.map((group) => group.videoId)).toEqual(['A', 'B', 'C'])
  })
})

function withOriginalRanks(hitItem: SearchHit, ranks: Array<[number, number]>): SearchHit {
  return {
    ...hitItem,
    stage1: {
      source_modalities: ['text'],
      text_queries: ranks.map(([queryIndex, originalRank]) => ({
        query_index: queryIndex,
        query: `Q${queryIndex + 1}`,
        raw_cosine: 0.5,
        original_rank: originalRank,
        rank: originalRank,
      })),
    },
  }
}
