import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { KeyframeDetail } from './KeyframeDetail'
import type { SearchHit } from '../api/types'

// RED reproducer: user found that clicking "Find Similar Images" inside the
// detail panel appeared to do nothing. It DID trigger a real search --
// but the panel is a full-screen overlay (KeyframeDetail.module.css
// .overlay: position: fixed; inset: 0), so the updated grid behind it was
// invisible. The fix: this button must also close the panel so the
// caller's already-updated grid becomes visible.

vi.mock('../api/client', () => ({
  getVideo: vi.fn().mockResolvedValue({
    video_id: 'L21_V001',
    watch_url: 'https://youtube.com/watch?v=abc',
    title: 'Ban tin',
    author: 'VTV',
    publish_date: '01/01/2026',
    duration: 600,
    fps: 25,
    num_keyframes: 1,
    s_v: 0,
    e_v: 0,
  }),
  getVideoKeyframes: vi.fn().mockResolvedValue({ keyframes: [] }),
}))

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

describe('KeyframeDetail', () => {
  it('closes itself when "Find Similar Images" is clicked, so the caller-owned grid becomes visible', async () => {
    const user = userEvent.setup()
    const onClose = vi.fn()
    const onFindSimilar = vi.fn()
    render(<KeyframeDetail hit={HIT} onClose={onClose} onFindSimilar={onFindSimilar} />)

    await user.click(screen.getByRole('button', { name: /find similar images/i }))

    expect(onFindSimilar).toHaveBeenCalledWith(HIT)
    expect(onClose).toHaveBeenCalledTimes(1)
  })

  it('still closes via the explicit close button', async () => {
    const user = userEvent.setup()
    const onClose = vi.fn()
    render(<KeyframeDetail hit={HIT} onClose={onClose} onFindSimilar={vi.fn()} />)

    await user.click(screen.getByRole('button', { name: /close/i }))

    expect(onClose).toHaveBeenCalledTimes(1)
  })
})
