import { describe, expect, it } from 'vitest'
import type { SearchHit, TrakeVideoResultItem } from '../api/types'
import {
  buildKisFinalRanking,
  buildKisSubmissionRows,
  buildDirectionalFrameSweep,
  buildRawFrameSweep,
  buildTrakeFinalRanking,
  buildTrakeSubmissionPlan,
  buildTrakeSynchronizedSteps,
  buildVqaFinalRows,
  createManualKisSelection,
  createManualTrakeSelection,
  createManualVqaSelection,
  createVisualKisSelection,
  createVisualTrakeSelection,
  kisSweepReachWarning,
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

function vqa(frame: number, answer: string, videoId = 'L30_V006'): SelectedVqaItem {
  return createManualVqaSelection(videoId, frame, answer)
}

describe('submissionExport', () => {
  it('builds KIS final ranking as manual order followed by remaining retrieval order', () => {
    const results = [1, 2, 3, 4, 5].map((frame) => hit(frame))
    const selected: SelectedKisItem[] = [results[1], results[3]].map((item, index) =>
      createVisualKisSelection(item, index),
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

  it('allows directional sweeps to overshoot the target by one step frame', () => {
    expect(
      buildDirectionalFrameSweep(90, {
        direction: 'left',
        step: 10,
        targetFrameIdx: 75,
      }),
    ).toEqual([90, 80, 70])
    expect(
      buildDirectionalFrameSweep(90, {
        direction: 'right',
        step: 10,
        targetFrameIdx: 115,
      }),
    ).toEqual([90, 100, 110, 120])
  })

  it('does not overshoot when a directional sweep lands exactly on the target', () => {
    expect(
      buildDirectionalFrameSweep(90, {
        direction: 'left',
        step: 10,
        targetFrameIdx: 70,
      }),
    ).toEqual([90, 80, 70])
    expect(kisSweepReachWarning(90, 'right', 10, 120)).toBeNull()
    expect(kisSweepReachWarning(90, 'right', 10, 115)).toBeNull()
  })

  it('warns only when a directional sweep exceeds the 100-row CSV budget', () => {
    expect(kisSweepReachWarning(1000, 'left', 10, 20)).toBeNull()
    expect(kisSweepReachWarning(1000, 'left', 10, 0)).toContain('first 100')
  })

  it('builds manual-only KIS submission as primary first, then ranking fallback', () => {
    const rows = buildKisSubmissionRows(
      [createManualKisSelection('L30_V046', 3255)],
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

  it('exports each selected primary before that selection sweep and ranking fallback', () => {
    const rows = buildKisSubmissionRows(
      [createManualKisSelection('L30_V006', 90, 'right', 10, 120)],
      [hit(100), hit(200, 'L30_V007')],
      6,
    )

    expect(rows.map((row) => `${row.video_id}:${row.frame_idx}`)).toEqual([
      'L30_V006:90',
      'L30_V006:100',
      'L30_V006:110',
      'L30_V006:120',
      'L30_V007:200',
    ])
  })

  it('preserves multiple manual and visual selection groups before fallback', () => {
    const visual = hit(30025, 'L25_V041')
    const rows = buildKisSubmissionRows(
      [
        createManualKisSelection('L30_V046', 3255, 'left', 10, 3235),
        createVisualKisSelection(visual, 0),
        createManualKisSelection('L28_V003', 9210, 'left', 10, 9190),
      ],
      [visual, hit(777, 'L31_V001')],
      8,
    )

    expect(rows.map((row) => `${row.video_id}:${row.frame_idx}`)).toEqual([
      'L30_V046:3255',
      'L30_V046:3245',
      'L30_V046:3235',
      'L25_V041:30025',
      'L28_V003:9210',
      'L28_V003:9200',
      'L28_V003:9190',
      'L31_V001:777',
    ])
  })

  it('deduplicates selected and margin frames before image-ranking fallback', () => {
    const rows = buildKisSubmissionRows(
      [createManualKisSelection('L30_V006', 90, 'left', 10, 80)],
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
        createManualKisSelection('L30_V006', 90, 'left', 10, 80),
        createManualKisSelection('L30_V006', 80),
      ],
      [hit(200, 'L30_V007')],
      4,
    )

    expect(rows.map((row) => `${row.video_id}:${row.frame_idx}`)).toEqual([
      'L30_V006:90',
      'L30_V006:80',
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
        createManualKisSelection('L30_V006', 90, 'right', 10, 120),
        createManualKisSelection('L30_V007', 200, 'right', 10, 230),
      ],
      [hit(300)],
      3,
    )

    expect(rows.map((row) => `${row.video_id}:${row.frame_idx}`)).toEqual([
      'L30_V006:90',
      'L30_V006:100',
      'L30_V006:110',
    ])
  })

  it('exports only selected and answered VQA rows', () => {
    const selected: SelectedVqaItem[] = [
      vqa(2, '5'),
      vqa(5, 'Năm người'),
      vqa(3, ''),
    ]

    const rows = buildVqaFinalRows(selected)

    expect(rows.map((row) => row.hit.frame_idx)).toEqual([2, 5])
  })

  it('keeps VQA answers independent and quotes comma answers', () => {
    const rows: SelectedVqaItem[] = [
      vqa(530, 'Có 3 người, bao gồm nam và nữ', 'L30_V001'),
      vqa(894, 'Anh ấy nói "Xin chào"'),
    ]

    expect(serializeVqaCsv(rows)).toBe(
      'L30_V001,530,"Có 3 người, bao gồm nam và nữ"\n' +
        'L30_V006,894,"Anh ấy nói ""Xin chào"""',
    )
  })

  it('expands VQA sweep rows with the same answer before CSV export', () => {
    const rows = buildVqaFinalRows([
      createManualVqaSelection('L30_V001', 90, 'hai người', 'right', 10, 115),
    ])

    expect(serializeVqaCsv(rows)).toBe(
      'L30_V001,90,"hai người"\n' +
        'L30_V001,100,"hai người"\n' +
        'L30_V001,110,"hai người"\n' +
        'L30_V001,120,"hai người"',
    )
  })

  it('builds TRAKE final ranking as selected sequence order plus remaining order', () => {
    const results = [
      sequence('A', [1, 2]),
      sequence('B', [3, 4]),
      sequence('C', [5, 6]),
      sequence('D', [7, 8]),
    ]
    const selected: SelectedTrakeSequence[] = [results[0], results[2]].map((result, index) =>
      createVisualTrakeSelection(result, index),
    )

    const finalRanking = buildTrakeFinalRanking(results, selected)

    expect(finalRanking.map((result) => result.video_id)).toEqual(['A', 'C', 'B', 'D'])
  })

  it('exports TRAKE rows with one frame id per event', () => {
    expect(serializeTrakeCsv([sequence('L30_V006', [894, 969])])).toBe('L30_V006,894,969')
    expect(validateTrakeEventCount([sequence('L30_V006', [894, 969])], 2)).toBeNull()
    expect(validateTrakeEventCount([sequence('L30_V006', [894])], 2)).toMatch(/2 event frames/)
  })

  it('keys TRAKE sequences by video id and event frames', () => {
    expect(trakeSequenceKey(sequence('A', [100, 200, 300]))).toBe('A::100|200|300')
    expect(trakeSequenceKey(sequence('B', [100, 200, 300]))).toBe('B::100|200|300')
  })

  it('builds synchronized TRAKE target-step variants without Cartesian combinations', () => {
    const variants = buildTrakeSynchronizedSteps(sequence('A', [100, 200, 300]), {
      targetFrames: [150, 280, 260],
      stepCount: 5,
    })

    expect(variants.map((variant) => variant.keyframes.map((keyframe) => keyframe.frame_idx))).toEqual([
      [110, 216, 292],
      [120, 232, 284],
      [130, 248, 276],
      [140, 264, 268],
      [150, 280, 260],
    ])
  })

  it('does not create synchronized TRAKE variants when every target equals its start', () => {
    expect(
      buildTrakeSynchronizedSteps(sequence('A', [100, 200]), {
        targetFrames: [100, 200],
        stepCount: 5,
      }),
    ).toEqual([])
  })

  it('drops duplicate synchronized TRAKE sequences created by rounding', () => {
    const variants = buildTrakeSynchronizedSteps(sequence('A', [100, 200]), {
      targetFrames: [101, 201],
      stepCount: 5,
    })

    expect(variants.map((variant) => variant.keyframes.map((keyframe) => keyframe.frame_idx))).toEqual([
      [101, 201],
    ])
  })

  it('builds TRAKE submission: original and synchronized steps stay grouped', () => {
    const results = [sequence('A', [100, 200]), sequence('B', [10, 20])]
    const selected: SelectedTrakeSequence[] = results.map((result, index) => ({
      ...createVisualTrakeSelection(result, index),
      stepTargetFrames:
        result.video_id === 'A'
          ? [105, 205]
          : [15, 25],
      stepCount: 1,
    }))
    const plan = buildTrakeSubmissionPlan(selected, results)

    expect(plan.rows.slice(0, 4).map((row) => trakeSequenceKey(row))).toEqual([
      trakeSequenceKey(results[0]),
      trakeSequenceKey(sequence('A', [105, 205])),
      trakeSequenceKey(results[1]),
      trakeSequenceKey(sequence('B', [15, 25])),
    ])
    expect(plan.centerCount).toBe(2)
    expect(plan.variantCount).toBe(2)
    const keys = plan.rows.map((row) => trakeSequenceKey(row))
    expect(new Set(keys).size).toBe(keys.length)
    // Every emitted row is still a full N-frame sequence.
    expect(plan.rows.every((row) => row.keyframes.length === 2)).toBe(true)
  })

  it('TRAKE submission plan respects maxRows', () => {
    const results = [sequence('A', [100, 200])]
    const selected: SelectedTrakeSequence[] = [
      {
        ...createVisualTrakeSelection(results[0], 0),
        stepTargetFrames: [120, 220],
        stepCount: 20,
      },
    ]
    const plan = buildTrakeSubmissionPlan(selected, results, { maxRows: 5 })
    expect(plan.rows.length).toBe(5)
  })

  it('creates manual TRAKE rows and exports selected rows before ranking fallback', () => {
    const manual = createManualTrakeSelection('L30_V046', [894, 969])
    const ranking = [sequence('L30_V001', [10, 20])]

    const plan = buildTrakeSubmissionPlan([manual], ranking)

    expect(manual.source).toBe('manual')
    expect(serializeTrakeCsv(plan.rows)).toBe('L30_V046,894,969\nL30_V001,10,20')
  })

  it('normalizes filenames without appending csv twice and validates task suffixes', () => {
    expect(normalizeCsvFilename('query-1-kis')).toBe('query-1-kis.csv')
    expect(normalizeCsvFilename('query-1-kis.csv')).toBe('query-1-kis.csv')
    expect(validateCsvFilename('query-1-kis', 'kis')).toMatchObject({ isValid: true })
    expect(validateCsvFilename('query-1-kis', 'qa')).toMatchObject({ isValid: false })
  })
})
