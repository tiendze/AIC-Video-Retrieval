import { describe, expect, it } from 'vitest'
import type { SearchHit, TrakeVideoResultItem } from '../api/types'
import {
  buildKisFinalRanking,
  buildKisSubmissionRows,
  buildRawFrameSweep,
  buildTrakeFinalRanking,
  buildVqaFinalRows,
  createManualKisSelection,
  createVisualKisSelection,
  normalizeCsvFilename,
  searchHitKey,
  serializeKisCsv,
  serializeTrakeCsv,
  serializeVqaCsv,
  trakeSequenceKey,
  validateCsvFilename,
  validateTrakeEventCount,
  type SelectedKisItem,
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
    const selected: SelectedKisItem[] = [results[1], results[3]].map((item, index) =>
      createVisualKisSelection(item, index, 10, 0),
    )

    const finalRanking = buildKisFinalRanking(results, selected)

    expect(finalRanking.map((item) => item.frame_idx)).toEqual([2, 4, 1, 3, 5])
    expect(new Set(finalRanking.map(searchHitKey)).size).toBe(5)
  })

  it('exports KIS rows as video and integer frame without spaces or header', () => {
    expect(serializeKisCsv([hit(894), hit(1220, 'L30_V003')])).toBe(
      'L30_V006,894\nL30_V003,1220',
    )
  })

  it('builds raw-frame sweep in center-first alternating order', () => {
    expect(buildRawFrameSweep(90, { step: 10, margin: 40 })).toEqual([
      90,
      80,
      100,
      70,
      110,
      60,
      120,
      50,
      130,
    ])
    expect(buildRawFrameSweep(90, { step: 10, margin: 40, maxRows: 5 })).toEqual([
      90,
      80,
      100,
      70,
      110,
    ])
    expect(buildRawFrameSweep(15, { step: 10, margin: 30 })).toEqual([15, 5, 25, 35, 45])
  })

  it('builds manual-only KIS submission as primary first, then ranking fallback', () => {
    const rows = buildKisSubmissionRows(
      [createManualKisSelection('L30_V046', 3255, 10, 0)],
      [hit(10), hit(20), hit(30)],
      4,
    )

    expect(rows.map((row) => `${row.video_id}:${row.frame_idx}`)).toEqual([
      'L30_V046:3255',
      'L30_V006:10',
      'L30_V006:20',
      'L30_V006:30',
    ])
  })

  it('exports each selected primary before that selection margin and ranking fallback', () => {
    const rows = buildKisSubmissionRows(
      [createManualKisSelection('L30_V006', 90, 10, 20)],
      [hit(100), hit(200, 'L30_V007')],
      6,
    )

    expect(rows.map((row) => `${row.video_id}:${row.frame_idx}`)).toEqual([
      'L30_V006:90',
      'L30_V006:80',
      'L30_V006:100',
      'L30_V006:70',
      'L30_V006:110',
      'L30_V007:200',
    ])
  })

  it('preserves multiple manual and visual selection groups before fallback', () => {
    const visual = hit(30025, 'L25_V041')
    const rows = buildKisSubmissionRows(
      [
        createManualKisSelection('L30_V046', 3255, 10, 20),
        createVisualKisSelection(visual, 0, 10, 0),
        createManualKisSelection('L28_V003', 9210, 10, 20),
      ],
      [visual, hit(777, 'L31_V001')],
      8,
    )

    expect(rows.map((row) => `${row.video_id}:${row.frame_idx}`)).toEqual([
      'L30_V046:3255',
      'L30_V046:3245',
      'L30_V046:3265',
      'L30_V046:3235',
      'L30_V046:3275',
      'L25_V041:30025',
      'L28_V003:9210',
      'L28_V003:9200',
    ])
  })

  it('deduplicates selected and margin frames before image-ranking fallback', () => {
    const rows = buildKisSubmissionRows(
      [createManualKisSelection('L30_V006', 90, 10, 10)],
      [hit(90), hit(80), hit(100), hit(200, 'L30_V007')],
      4,
    )

    expect(serializeKisCsv(rows)).toBe(
      'L30_V006,90\n' + 'L30_V006,80\n' + 'L30_V006,100\n' + 'L30_V007,200',
    )
  })

  it('keeps a later primary group visible while deduplicating an overlapping margin row', () => {
    const rows = buildKisSubmissionRows(
      [
        createManualKisSelection('L30_V006', 90, 10, 10),
        createManualKisSelection('L30_V006', 80, 10, 0),
      ],
      [hit(200, 'L30_V007')],
      4,
    )

    expect(rows.map((row) => `${row.video_id}:${row.frame_idx}`)).toEqual([
      'L30_V006:90',
      'L30_V006:80',
      'L30_V006:100',
      'L30_V007:200',
    ])
  })

  it('fills directly from ranking when no manual input was added', () => {
    const rows = buildKisSubmissionRows([], [hit(1), hit(2), hit(3)], 2)

    expect(rows.map((row) => row.frame_idx)).toEqual([1, 2])
  })

  it('honors the submission limit with earlier selection groups taking priority', () => {
    const rows = buildKisSubmissionRows(
      [
        createManualKisSelection('L30_V006', 90, 10, 20),
        createManualKisSelection('L30_V007', 200, 10, 20),
      ],
      [hit(300)],
      3,
    )

    expect(rows.map((row) => `${row.video_id}:${row.frame_idx}`)).toEqual([
      'L30_V006:90',
      'L30_V006:80',
      'L30_V006:100',
    ])
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
