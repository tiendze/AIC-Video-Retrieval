import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { ResultCard } from './ResultCard'
import type { SearchHit } from '../api/types'

// RED reproducer for PRD §F9 / plan §P1.6 acceptance criterion:
// "video_id + timestamp hiển thị NGAY TRÊN mỗi thumbnail, không cần hover."

const HIT: SearchHit = {
  keyframe_id: 'L21_V001_f10',
  video_id: 'L21_V001',
  score: 0.9,
  frame_idx: 10,
  timestamp: 42.5,
  thumbnail_path: 't.jpg',
  thumbnail_url: '/api/keyframes/L21_V001_f10/thumbnail',
  image_url: '/api/keyframes/L21_V001_f10/image',
}

const ASR_VIDEO = {
  video_id: 'L21_V001',
  video_rank: 1,
  best_bm25_score: 8.73,
  segments: [
    {
      segment_id: 'L21_V001_seg10',
      text: 'công trình thủy lợi',
      start_time: 151.2,
      end_time: 158.4,
      raw_bm25: 8.73,
    },
  ],
}

describe('ResultCard', () => {
  it('shows video_id and timestamp as always-visible text, not only on hover', () => {
    render(<ResultCard hit={HIT} onSelect={vi.fn()} onFindSimilar={vi.fn()} />)

    // getByText fails on content hidden via CSS (e.g. opacity:0 shown only
    // on :hover) unless it is actually visible in the accessibility tree --
    // this is the behavioral check, not a class-name check.
    expect(screen.getByText('L21_V001')).toBeVisible()
    expect(screen.getByText('42.5s')).toBeVisible()
  })

  it('shows the ASR video rank without rendering transcript text in the card', () => {
    render(
      <ResultCard
        hit={{ ...HIT, asr_evidence: ASR_VIDEO }}
        onSelect={vi.fn()}
        onFindSimilar={vi.fn()}
      />,
    )

    expect(screen.getByText('ASR #1')).toBeInTheDocument()
    expect(screen.queryByText('công trình thủy lợi')).not.toBeInTheDocument()
  })

  it('calls onSelect with the hit when clicked', async () => {
    const onSelect = vi.fn()
    render(<ResultCard hit={HIT} onSelect={onSelect} onFindSimilar={vi.fn()} />)

    screen.getByRole('button', { name: /L21_V001/i }).click()

    expect(onSelect).toHaveBeenCalledWith(HIT)
  })

  it('toggles manual selection without triggering the card selection action', async () => {
    const user = userEvent.setup()
    const onSelect = vi.fn()
    const onToggleManualSelection = vi.fn()
    render(
      <ResultCard
        hit={HIT}
        onSelect={onSelect}
        onFindSimilar={vi.fn()}
        onToggleManualSelection={onToggleManualSelection}
        manualSelectionLabel="Select KIS"
      />,
    )

    await user.click(screen.getByText('Select KIS'))

    expect(onToggleManualSelection).toHaveBeenCalledWith(HIT)
    expect(onSelect).not.toHaveBeenCalled()
  })
})
