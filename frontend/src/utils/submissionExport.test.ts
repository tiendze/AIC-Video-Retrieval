import { describe, expect, it } from 'vitest'
import type { SearchHit, TrakeVideoResultItem } from '../api/types'
import {
  buildKisFinalRanking,
  buildRawFrameSweep,
  buildTrakeFinalRanking,
  buildVqaFinalRows,
  normalizeCsvFilename,
  searchHitKey,
  serializeKisCsv,
  serializeTrakeCsv,
  serializeVqaCsv,
  trakeSequenceKey,
  validateCsvFilename,
  validateTrakeEventCount,
  type SelectedSearchItem,
  type SelectedTrakeSequence,
  type SelectedVqaItem,
} from './submissionExport'

function hit(frame: number, videoId = 'L30_V006'): SearchHit {
  return {
    keyframe_id: `${videoId}_f${frame}`,
    video_id: videoId,
    score: 1,
    frame_idx: frame,
    timestamp: frame / 25,
    thumbnail_path: '',
    thumbnail_url: `/api/keyframes/${videoId}_f${frame}/thumbnail`,
    image_url: `/api/keyframes/${videoId}_f${frame}/image`,
  }
}

function sequence(videoId: string, frames: number[]): TrakeVideoResultItem {
  return {
    video_id: videoId,
    score: 0.5,
    keyframes: frames.map((frame) => ({
      keyframe_id: `${videoId}_f${frame}`,
      frame_idx: frame,
      timestamp: frame / 25,
      thumbnail_url: `/api/keyframes/${videoId}_f${frame}/thumbnail`,
      image_url: `/api/keyframes/${videoId}_f${frame}/image`,
    })),
  }
}

describe('submissionExport', () => {
  it('builds KIS final ranking as manual order followed by remaining retrieval order', () => {
    const results = [1, 2, 3, 4, 5].map((frame) => hit(frame))
    const selected: SelectedSearchItem[] = [results[1], results[3]].map((item, index) => ({
      key: searchHitKey(item),
      hit: item,
      originalRank: index,
    }))

    const finalRanking = buildKisFinalRanking(results, selected)

    expect(finalRanking.map((item) => item.frame_idx)).toEqual([2, 4, 1, 3, 5])
    expect(new Set(finalRanking.map(searchHitKey)).size).toBe(5)
  })

  it('exports KIS rows as video and integer frame without spaces or header', () => {
    expect(serializeKisCsv([hit(894), hit(1220, 'L30_V003')])).toBe(
      'L30_V006,894\nL30_V003,1220',
    )
  })

  it('builds raw-frame sweep around a center frame and keeps the center', () => {
    expect(buildRawFrameSweep(1000, { step: 10, marginCount: 3 })).toEqual([
      970,
      980,
      990,
      1000,
      1010,
      1020,
      1030,
    ])
    expect(buildRawFrameSweep(15, { step: 10, marginCount: 3 })).toEqual([5, 15, 25, 35, 45])
  })

  it('expands KIS CSV rows with raw-frame sweep settings', () => {
    expect(serializeKisCsv([hit(1000)], { step: 10, marginCount: 1 })).toBe(
      'L30_V006,990\nL30_V006,1000\nL30_V006,1010',
    )
  })

  it('exports only selected and answered VQA rows', () => {
    const selected: SelectedVqaItem[] = [
      { key: searchHitKey(hit(2)), hit: hit(2), originalRank: 1, answer: '5' },
      { key: searchHitKey(hit(5)), hit: hit(5), originalRank: 4, answer: 'Năm người' },
      { key: searchHitKey(hit(3)), hit: hit(3), originalRank: 2, answer: '' },
    ]

    const rows = buildVqaFinalRows(selected)

    expect(rows.map((row) => row.hit.frame_idx)).toEqual([2, 5])
  })

  it('keeps VQA answers independent and quotes comma answers', () => {
    const rows: SelectedVqaItem[] = [
      {
        key: searchHitKey(hit(530, 'L30_V001')),
        hit: hit(530, 'L30_V001'),
        originalRank: 0,
        answer: 'Có 3 người, bao gồm nam và nữ',
      },
      {
        key: searchHitKey(hit(894)),
        hit: hit(894),
        originalRank: 1,
        answer: 'Anh ấy nói "Xin chào"',
      },
    ]

    expect(serializeVqaCsv(rows)).toBe(
      'L30_V001,530,"Có 3 người, bao gồm nam và nữ"\n' +
        'L30_V006,894,"Anh ấy nói ""Xin chào"""',
    )
  })

  it('builds TRAKE final ranking as selected sequence order plus remaining order', () => {
    const results = [
      sequence('A', [1, 2]),
      sequence('B', [3, 4]),
      sequence('C', [5, 6]),
      sequence('D', [7, 8]),
    ]
    const selected: SelectedTrakeSequence[] = [results[0], results[2]].map((result, index) => ({
      key: trakeSequenceKey(result),
      result,
      originalRank: index,
    }))

    const finalRanking = buildTrakeFinalRanking(results, selected)

    expect(finalRanking.map((result) => result.video_id)).toEqual(['A', 'C', 'B', 'D'])
  })

  it('exports TRAKE rows with one frame id per event', () => {
    expect(serializeTrakeCsv([sequence('L30_V006', [894, 969])])).toBe('L30_V006,894,969')
    expect(validateTrakeEventCount([sequence('L30_V006', [894, 969])], 2)).toBeNull()
    expect(validateTrakeEventCount([sequence('L30_V006', [894])], 2)).toMatch(/2 event frames/)
  })

  it('normalizes filenames without appending csv twice and validates task suffixes', () => {
    expect(normalizeCsvFilename('query-1-kis')).toBe('query-1-kis.csv')
    expect(normalizeCsvFilename('query-1-kis.csv')).toBe('query-1-kis.csv')
    expect(validateCsvFilename('query-1-kis', 'kis')).toMatchObject({ isValid: true })
    expect(validateCsvFilename('query-1-kis', 'qa')).toMatchObject({ isValid: false })
  })
})
