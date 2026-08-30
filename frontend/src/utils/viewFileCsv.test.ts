import { describe, expect, it } from 'vitest'
import { parseViewFileCsv } from './viewFileCsv'

describe('parseViewFileCsv', () => {
  it('parses current headerless KIS exports', () => {
    const result = parseViewFileCsv('query-1-kis.csv', 'L21_V001,10\nL21_V002,20\n')

    expect(result.ok).toBe(true)
    if (!result.ok) return
    expect(result.parsed.kind).toBe('kis')
    expect(result.parsed.rows).toMatchObject([
      { videoId: 'L21_V001', frameIdx: 10, answer: null },
      { videoId: 'L21_V002', frameIdx: 20, answer: null },
    ])
  })

  it('parses VQA exports with flexible header names', () => {
    const result = parseViewFileCsv(
      'qa.csv',
      'video_id,Frame Index,Answer\nL21_V001,10,"green, yellow bus"\n',
    )

    expect(result.ok).toBe(true)
    if (!result.ok) return
    expect(result.parsed.kind).toBe('vqa')
    expect(result.parsed.rows[0]).toMatchObject({
      videoId: 'L21_V001',
      frameIdx: 10,
      answer: 'green, yellow bus',
    })
  })

  it('returns a readable error for malformed rows', () => {
    const result = parseViewFileCsv('bad.csv', 'video_id,frame_idx\nL21_V001,abc\n')

    expect(result).toEqual({ ok: false, error: 'Row 2 has invalid frame_idx.' })
  })
})
