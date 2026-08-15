import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import type { TrakeVideoResultItem } from '../api/types'
import { TrakeResultList } from './TrakeResultList'

// RED reproducer for DANTE/TRAKE's results UI (plan §P2.2, PRD §F6/US-4 +
// F9(C): "Trả về top-k video, mỗi video kèm đúng N keyframe, thứ tự thời
// gian không đảo"). Each result is one video's *ordered* keyframe
// sequence, not a flat grid of individual hits -- this component must
// never re-sort `keyframes`, and must show frame_idx (the AIC submission
// unit, references.md §1.2), not just a thumbnail.
const RESULTS: TrakeVideoResultItem[] = [
  {
    video_id: 'L21_V001',
    score: 0.8123,
    keyframes: [
      {
        keyframe_id: 'L21_V001_f10',
        frame_idx: 10,
        timestamp: 1.0,
        thumbnail_url: '/api/keyframes/L21_V001_f10/thumbnail',
        image_url: '/api/keyframes/L21_V001_f10/image',
      },
      {
        keyframe_id: 'L21_V001_f50',
        frame_idx: 50,
        timestamp: 5.0,
        thumbnail_url: '/api/keyframes/L21_V001_f50/thumbnail',
        image_url: '/api/keyframes/L21_V001_f50/image',
      },
    ],
  },
]

describe('TrakeResultList', () => {
  it('shows an empty-state message when there are no results', () => {
    render(<TrakeResultList results={[]} />)

    expect(screen.getByText(/no results yet/i)).toBeInTheDocument()
  })

  it('renders the video_id for each result', () => {
    render(<TrakeResultList results={RESULTS} />)

    expect(screen.getByText('L21_V001')).toBeInTheDocument()
  })

  it('renders every keyframe thumbnail in the exact order given', () => {
    render(<TrakeResultList results={RESULTS} />)

    const images = screen.getAllByRole('img')
    expect(images.map((img) => img.getAttribute('alt'))).toEqual([
      'frame 10',
      'frame 50',
    ])
  })

  it('shows the frame_idx submission string in video_id, frame order', () => {
    render(<TrakeResultList results={RESULTS} />)

    expect(screen.getByText('L21_V001, 10, 50')).toBeInTheDocument()
  })
})
