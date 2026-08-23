import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { getVideoKeyframes } from '../api/client'
import {
  isInsideEvidence,
  KeyframeDetail,
  nearestKeyframe,
  type DetailFrame,
} from './KeyframeDetail'
import type { SearchHit, VideoKeyframe } from '../api/types'

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

function frame(keyframeId: string, timestamp: number): VideoKeyframe {
  const frameIdx = Number(keyframeId.replace('f', ''))
  return {
    keyframe_id: keyframeId,
    frame_idx: frameIdx,
    timestamp,
    global_index: frameIdx,
    thumbnail_url: `/api/keyframes/${keyframeId}/thumbnail`,
    image_url: `/api/keyframes/${keyframeId}/image`,
  }
}

describe('KeyframeDetail', () => {
  beforeEach(() => {
    vi.mocked(getVideoKeyframes).mockResolvedValue({ keyframes: [] })
  })

  it('closes itself when "Find Similar Images" is clicked, so the caller-owned grid becomes visible', async () => {
    const user = userEvent.setup()
    const onClose = vi.fn()
    const onFindSimilar = vi.fn()
    render(<KeyframeDetail hit={HIT} onClose={onClose} onFindSimilar={onFindSimilar} />)

    await user.click(screen.getByRole('button', { name: /find similar images/i }))

    expect(onFindSimilar).toHaveBeenCalledWith(HIT.image_url)
    expect(onClose).toHaveBeenCalledTimes(1)
  })

  it('still closes via the explicit close button', async () => {
    const user = userEvent.setup()
    const onClose = vi.fn()
    render(<KeyframeDetail hit={HIT} onClose={onClose} onFindSimilar={vi.fn()} />)

    await user.click(screen.getByRole('button', { name: /close/i }))

    expect(onClose).toHaveBeenCalledTimes(1)
  })

  it('keeps the displayed frame centered in the bottom strip and navigates prev/next', async () => {
    const user = userEvent.setup()
    vi.mocked(getVideoKeyframes).mockResolvedValueOnce({
      keyframes: [
        {
          keyframe_id: 'L21_V001_f5',
          frame_idx: 5,
          timestamp: 20,
          global_index: 0,
          thumbnail_url: '/api/keyframes/L21_V001_f5/thumbnail',
          image_url: '/api/keyframes/L21_V001_f5/image',
        },
        {
          keyframe_id: HIT.keyframe_id,
          frame_idx: HIT.frame_idx,
          timestamp: HIT.timestamp,
          global_index: 1,
          thumbnail_url: HIT.thumbnail_url,
          image_url: HIT.image_url,
        },
        {
          keyframe_id: 'L21_V001_f20',
          frame_idx: 20,
          timestamp: 50,
          global_index: 2,
          thumbnail_url: '/api/keyframes/L21_V001_f20/thumbnail',
          image_url: '/api/keyframes/L21_V001_f20/image',
        },
      ],
    })
    render(<KeyframeDetail hit={HIT} onClose={vi.fn()} onFindSimilar={vi.fn()} />)

    await waitFor(() =>
      expect(screen.getByRole('button', { name: /show frame 10/i })).toHaveAttribute(
        'aria-current',
        'true',
      ),
    )

    await user.click(screen.getByRole('button', { name: /next keyframe/i }))

    expect(screen.getByAltText('L21_V001 @ 50s')).toHaveAttribute(
      'src',
      '/api/keyframes/L21_V001_f20/image',
    )
    expect(screen.getByRole('button', { name: /show frame 20/i })).toHaveAttribute(
      'aria-current',
      'true',
    )

    await user.click(screen.getByRole('button', { name: /previous keyframe/i }))

    expect(screen.getByAltText('L21_V001 @ 42.5s')).toHaveAttribute('src', HIT.image_url)
  })

  it('finds similar images from the frame currently shown in the detail viewer', async () => {
    const user = userEvent.setup()
    const onFindSimilar = vi.fn()
    vi.mocked(getVideoKeyframes).mockResolvedValueOnce({
      keyframes: [
        {
          keyframe_id: HIT.keyframe_id,
          frame_idx: HIT.frame_idx,
          timestamp: HIT.timestamp,
          global_index: 1,
          thumbnail_url: HIT.thumbnail_url,
          image_url: HIT.image_url,
        },
        {
          keyframe_id: 'L21_V001_f20',
          frame_idx: 20,
          timestamp: 50,
          global_index: 2,
          thumbnail_url: '/api/keyframes/L21_V001_f20/thumbnail',
          image_url: '/api/keyframes/L21_V001_f20/image',
        },
      ],
    })
    render(<KeyframeDetail hit={HIT} onClose={vi.fn()} onFindSimilar={onFindSimilar} />)

    await waitFor(() => expect(screen.getByRole('button', { name: /next keyframe/i })).toBeEnabled())
    await user.click(screen.getByRole('button', { name: /next keyframe/i }))
    await user.click(screen.getByRole('button', { name: /find similar images/i }))

    expect(onFindSimilar).toHaveBeenCalledWith('/api/keyframes/L21_V001_f20/image')
  })

  it('highlights TRAKE event frames while keeping the clicked frame active', () => {
    const eventFrames: DetailFrame[] = [
      {
        keyframe_id: HIT.keyframe_id,
        video_id: HIT.video_id,
        frame_idx: HIT.frame_idx,
        timestamp: HIT.timestamp,
        thumbnail_url: HIT.thumbnail_url,
        image_url: HIT.image_url,
      },
      {
        keyframe_id: 'L21_V001_f20',
        video_id: HIT.video_id,
        frame_idx: 20,
        timestamp: 50,
        thumbnail_url: '/api/keyframes/L21_V001_f20/thumbnail',
        image_url: '/api/keyframes/L21_V001_f20/image',
      },
    ]

    render(
      <KeyframeDetail
        hit={HIT}
        highlightedFrames={eventFrames}
        onClose={vi.fn()}
        onFindSimilar={vi.fn()}
      />,
    )

    expect(screen.getByRole('button', { name: /show frame 10/i })).toHaveAttribute(
      'data-highlighted',
      'true',
    )
    expect(screen.getByRole('button', { name: /show frame 10/i })).toHaveAttribute(
      'aria-current',
      'true',
    )
    expect(screen.getByRole('button', { name: /show frame 20/i })).toHaveAttribute(
      'data-highlighted',
      'true',
    )
  })

  it('selects the earlier timestamp when keyframes tie around the evidence midpoint', () => {
    const segment = { ...ASR_VIDEO.segments[0], start_time: 10, end_time: 14 }

    expect(nearestKeyframe(segment, [frame('f13', 13), frame('f11', 11)])).toEqual(
      frame('f11', 11),
    )
  })

  it('treats both evidence interval boundaries as highlighted', () => {
    const segment = ASR_VIDEO.segments[0]

    expect(isInsideEvidence({ timestamp: 151.2 }, segment)).toBe(true)
    expect(isInsideEvidence({ timestamp: 158.4 }, segment)).toBe(true)
    expect(isInsideEvidence({ timestamp: 151.19 }, segment)).toBe(false)
    expect(isInsideEvidence({ timestamp: 158.41 }, segment)).toBe(false)
  })

  it('selects the nearest real keyframe to the evidence midpoint and highlights the interval', async () => {
    vi.mocked(getVideoKeyframes).mockResolvedValue({
      keyframes: [frame('f1', 150), frame('f2', 154), frame('f3', 160)],
    })
    render(
      <KeyframeDetail
        hit={{ ...HIT, asr_evidence: ASR_VIDEO }}
        onClose={vi.fn()}
        onFindSimilar={vi.fn()}
      />,
    )

    await userEvent.click(screen.getByRole('button', { name: /151\.2s–158\.4s/ }))

    expect(screen.getByRole('button', { name: 'Show frame 2' })).toHaveAttribute(
      'aria-current',
      'true',
    )
    expect(screen.getByRole('button', { name: 'Show frame 2' })).toHaveAttribute(
      'data-highlighted',
      'true',
    )
    expect(screen.getByText('151.2s–158.4s')).toBeInTheDocument()
    expect(screen.getByText('công trình thủy lợi')).toBeInTheDocument()
  })

  it('keeps the current real keyframe when the video has no loaded keyframes', async () => {
    vi.mocked(getVideoKeyframes).mockResolvedValue({ keyframes: [] })
    render(
      <KeyframeDetail
        hit={{ ...HIT, asr_evidence: ASR_VIDEO }}
        onClose={vi.fn()}
        onFindSimilar={vi.fn()}
      />,
    )

    await userEvent.click(screen.getByRole('button', { name: /151\.2s–158\.4s/ }))

    expect(screen.getByText(`frame ${HIT.frame_idx}`)).toBeInTheDocument()
  })
})
