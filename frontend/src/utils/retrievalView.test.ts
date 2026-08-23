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

  it('applies OCR and Title OCR priority before best image rank', () => {
    const groups = buildVideoGroups(
      [
        withEvidence(hit('A', 10, 50), { ocr: true, title: true }),
        withEvidence(hit('B', 20, 1), {}),
        withEvidence(hit('C', 30, 4), { title: true }),
        withEvidence(hit('D', 40, 10), { ocr: true }),
      ],
      'best',
      60,
    )

    expect(groups.map((group) => group.videoId)).toEqual(['A', 'C', 'D', 'B'])
    expect(groups.map((group) => group.finalVideoRank)).toEqual([1, 2, 3, 4])
    expect(groups[0]).toMatchObject({
      priorityTier: 0,
      ocrDetected: true,
      titleOcrDetected: true,
    })
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

  it('applies OCR priority before RRF and then RRF within the same priority tier', () => {
    const hits: SearchHit[] = [
      withEvidence(withOriginalRanks(hit('A', 10, 50), [[0, 40]]), {
        ocr: true,
        title: true,
      }),
      withEvidence(withOriginalRanks(hit('B', 20, 10), [[0, 30]]), { ocr: true }),
      withEvidence(withOriginalRanks(hit('C', 30, 4), [[0, 5]]), { title: true }),
      withEvidence(withOriginalRanks(hit('D', 40, 1), [[0, 1]]), {}),
    ]

    const groups = buildVideoGroups(hits, 'rrf', 60)

    expect(groups.map((group) => group.videoId)).toEqual(['A', 'C', 'B', 'D'])
  })

  it('assigns final video ranks after the caller hides a video', () => {
    const visibleHits = [hit('B', 20, 2), hit('C', 30, 3)]
    const groups = buildVideoGroups(visibleHits, 'best', 60)

    expect(groups.map((group) => [group.videoId, group.finalVideoRank])).toEqual([
      ['B', 1],
      ['C', 2],
    ])
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

function withEvidence(
  hitItem: SearchHit,
  evidence: { ocr?: boolean; title?: boolean },
): SearchHit {
  const sourceModalities = ['text']
  if (evidence.ocr) sourceModalities.push('ocr')
  if (evidence.title) sourceModalities.push('title')
  return {
    ...hitItem,
    stage1: {
      ...hitItem.stage1,
      source_modalities: sourceModalities as Array<'text' | 'ocr' | 'title'>,
      ocr: {
        query: 'London Zoo',
        raw_bm25: evidence.ocr ? 5 : 0,
        bm25_reference: 5,
        bm25_normalized: evidence.ocr ? 1 : 0,
        coverage: evidence.ocr ? 1 : 0,
        query_score: evidence.ocr ? 1 : 0,
        rank: evidence.ocr ? 1 : 4,
      },
      title: {
        query: 'chicken',
        video_title: evidence.title ? 'chicken soup' : '',
        title_score: evidence.title ? 1 : 0,
        rank: evidence.title ? 1 : 4,
      },
    },
  }
}
