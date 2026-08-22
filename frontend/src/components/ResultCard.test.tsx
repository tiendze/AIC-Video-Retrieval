import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { ResultCard } from './ResultCard'
import type { SearchHit } from '../api/types'

// Result labels prioritize the submission identity: video_id + raw frame_idx.

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

describe('ResultCard', () => {
  it('shows video_id and frame as always-visible text, not only on hover', () => {
    render(<ResultCard hit={HIT} onSelect={vi.fn()} onFindSimilar={vi.fn()} />)

    // getByText fails on content hidden via CSS (e.g. opacity:0 shown only
    // on :hover) unless it is actually visible in the accessibility tree --
    // this is the behavioral check, not a class-name check.
    expect(screen.getByText('L21_V001')).toBeVisible()
    expect(screen.getByText('frame 10')).toBeVisible()
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
