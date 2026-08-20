import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import type { TrakeVideoResultItem } from '../api/types'
import { formatTrakeTimestamp, TrakeResultList } from './TrakeResultList'

// RED reproducer for DANTE/TRAKE's results UI (plan §P2.2, PRD §F6/US-4 +
// F9(C): "Trả về top-k video, mỗi video kèm đúng N keyframe, thứ tự thời
// gian không đảo"). Each result is one ordered keyframe sequence, not a
// flat grid of individual hits -- this component must never re-sort
// `keyframes`, and must show frame_idx (the AIC submission unit,
// references.md §1.2), not just a thumbnail.
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

const GROUPED_RESULTS: TrakeVideoResultItem[] = [
  RESULTS[0] as TrakeVideoResultItem,
  {
    video_id: 'L21_V001',
    score: 0.71234,
    keyframes: [
      {
        keyframe_id: 'L21_V001_f70',
        frame_idx: 70,
        timestamp: 7.0,
        thumbnail_url: '/api/keyframes/L21_V001_f70/thumbnail',
        image_url: '/api/keyframes/L21_V001_f70/image',
      },
      {
        keyframe_id: 'L21_V001_f90',
        frame_idx: 90,
        timestamp: 9.0,
        thumbnail_url: '/api/keyframes/L21_V001_f90/thumbnail',
        image_url: '/api/keyframes/L21_V001_f90/image',
      },
    ],
  },
  {
    video_id: 'L21_V002',
    score: 0.61234,
    keyframes: [
      {
        keyframe_id: 'L21_V002_f5',
        frame_idx: 5,
        timestamp: 0.5,
        thumbnail_url: '/api/keyframes/L21_V002_f5/thumbnail',
        image_url: '/api/keyframes/L21_V002_f5/image',
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
      'event 1 at 00:01.0',
      'event 2 at 00:05.0',
    ])
  })

  it('shows timestamps as the primary per-event temporal label', () => {
    render(<TrakeResultList results={RESULTS} />)

    expect(screen.getByText('Event 1 - 00:01.0')).toBeInTheDocument()
    expect(screen.getByText('Event 2 - 00:05.0')).toBeInTheDocument()
    expect(screen.getByText('frame 10')).toBeInTheDocument()
  })

  it('formats hour-long timestamps without losing precision', () => {
    expect(formatTrakeTimestamp(3672.4)).toBe('01:01:12.4')
  })

  it('shows the frame_idx submission string in video_id, frame order', () => {
    render(<TrakeResultList results={RESULTS} />)

    expect(screen.getByText('L21_V001, 10, 50')).toBeInTheDocument()
  })

  it('opens the clicked keyframe detail from a TRAKE sequence', async () => {
    const user = userEvent.setup()
    const onOpenDetail = vi.fn()
    render(<TrakeResultList results={RESULTS} onOpenDetail={onOpenDetail} />)

    await user.click(screen.getByRole('button', { name: /open event 2 frame 50/i }))

    expect(onOpenDetail).toHaveBeenCalledWith(RESULTS[0], 1)
  })

  it('selects a whole TRAKE sequence without mixing individual event frames', async () => {
    const user = userEvent.setup()
    const onToggleSequence = vi.fn()
    render(<TrakeResultList results={RESULTS} onToggleSequence={onToggleSequence} />)

    await user.click(screen.getByRole('button', { name: 'Select' }))

    expect(onToggleSequence).toHaveBeenCalledWith(RESULTS[0], 0)
  })

  it('groups duplicate video ids and separates their sequences', () => {
    render(<TrakeResultList results={GROUPED_RESULTS} />)

    expect(screen.getAllByText('L21_V001')).toHaveLength(1)
    expect(screen.getByText('2 sequences')).toBeInTheDocument()
    expect(screen.getAllByText(/Sequence /)).toHaveLength(3)
    expect(screen.getByText('Score 0.8123')).toBeInTheDocument()
    expect(screen.getByText('Score 0.7123')).toBeInTheDocument()
    expect(screen.getByRole('separator')).toBeInTheDocument()
    expect(screen.getByText('L21_V001, 70, 90')).toBeInTheDocument()
  })
})
