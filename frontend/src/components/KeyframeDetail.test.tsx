import { act, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { getVideo, getVideoKeyframes } from '../api/client'
import {
  frameIndexFromTimestamp,
  isInsideEvidence,
  KeyframeDetail,
  nearestKeyframe,
  type DetailFrame,
  youtubeVideoIdFromUrl,
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
  detected: true,
  asr_matches: [
    {
      query: 'công trình thủy lợi',
      detected: true,
      matches: [
        {
          segment_id: 'L21_V001_seg10',
          text: 'công trình thủy lợi',
          start_time: 151.2,
          end_time: 158.4,
        },
      ],
    },
  ],
  segments: [
    {
      segment_id: 'L21_V001_seg10',
      text: 'công trình thủy lợi',
      start_time: 151.2,
      end_time: 158.4,
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

interface FakeYouTubePlayer {
  getCurrentTime: ReturnType<typeof vi.fn>
  seekTo: ReturnType<typeof vi.fn>
  destroy: ReturnType<typeof vi.fn>
}

interface FakeYouTubeOptions {
  events?: {
    onReady?: (event: { target: FakeYouTubePlayer }) => void
    onStateChange?: (event: { target: FakeYouTubePlayer; data?: number }) => void
  }
}

let fakeYouTubeTime = 0
let lastYouTubePlayer: FakeYouTubePlayer | null = null
let lastYouTubeOptions: FakeYouTubeOptions | null = null

function installYouTubeMock(): void {
  fakeYouTubeTime = 0
  lastYouTubePlayer = null
  lastYouTubeOptions = null
  const Player = vi.fn(function PlayerMock(
    _element: HTMLElement | string,
    options: FakeYouTubeOptions,
  ) {
    lastYouTubeOptions = options
    const player: FakeYouTubePlayer = {
      getCurrentTime: vi.fn(() => fakeYouTubeTime),
      seekTo: vi.fn((seconds: number) => {
        fakeYouTubeTime = seconds
      }),
      destroy: vi.fn(),
    }
    lastYouTubePlayer = player
    setTimeout(() => options.events?.onReady?.({ target: player }), 0)
    return player
  })
  Object.defineProperty(window, 'YT', {
    configurable: true,
    value: {
      Player,
      PlayerState: {
        PAUSED: 2,
      },
    },
  })
}

describe('KeyframeDetail', () => {
  beforeEach(() => {
    vi.mocked(getVideo).mockImplementation((videoId: string) =>
      Promise.resolve({
        video_id: videoId,
        watch_url: `https://youtube.com/watch?v=${videoId}`,
        title: `Title ${videoId}`,
        author: 'VTV',
        publish_date: '01/01/2026',
        duration: 600,
        fps: 25,
        num_keyframes: 1,
        s_v: 0,
        e_v: 0,
      }),
    )
    vi.mocked(getVideoKeyframes).mockResolvedValue({ keyframes: [] })
    delete window.YT
    delete window.onYouTubeIframeAPIReady
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

  it('uses the currently displayed frame video for the external YouTube link', async () => {
    const user = userEvent.setup()
    const otherFrame: DetailFrame = {
      keyframe_id: 'L21_V029_f11591',
      video_id: 'L21_V029',
      frame_idx: 11591,
      timestamp: 386.36,
      thumbnail_url: '/api/keyframes/L21_V029_f11591/thumbnail',
      image_url: '/api/keyframes/L21_V029_f11591/image',
    }

    render(
      <KeyframeDetail
        hit={HIT}
        highlightedFrames={[otherFrame]}
        onClose={vi.fn()}
        onFindSimilar={vi.fn()}
      />,
    )

    await waitFor(() =>
      expect(screen.getByRole('link', { name: /open youtube/i })).toHaveAttribute(
        'href',
        'https://youtube.com/watch?v=L21_V001&t=42s',
      ),
    )

    await user.click(screen.getByRole('button', { name: /show frame 11591/i }))

    await waitFor(() =>
      expect(screen.getByRole('link', { name: /open youtube/i })).toHaveAttribute(
        'href',
        'https://youtube.com/watch?v=L21_V029&t=386s',
      ),
    )
  })

  it('selects the earlier decimal timestamp when distances tie around the midpoint', () => {
    const segment = ASR_VIDEO.segments[0]

    expect(nearestKeyframe(segment, [frame('f1', 158.4), frame('f2', 151.2)])).toEqual(
      frame('f2', 151.2),
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

  it('extracts YouTube video ids from common watch urls', () => {
    expect(youtubeVideoIdFromUrl('https://youtube.com/watch?v=abc123')).toBe('abc123')
    expect(youtubeVideoIdFromUrl('https://youtu.be/abc123?t=5')).toBe('abc123')
    expect(youtubeVideoIdFromUrl('https://www.youtube.com/embed/abc123')).toBe('abc123')
    expect(youtubeVideoIdFromUrl('https://example.test/watch?v=abc123')).toBeNull()
  })

  it('calculates zero-based frame indices from timestamp and fps', () => {
    expect(frameIndexFromTimestamp(130.215, 25)).toBe(3255)
    expect(frameIndexFromTimestamp(0, 25)).toBe(0)
    expect(frameIndexFromTimestamp(10, 0)).toBeNull()
    expect(frameIndexFromTimestamp(-1, 25)).toBeNull()
  })

  it('updates the displayed timestamp when the embedded YouTube player pauses', async () => {
    const user = userEvent.setup()
    installYouTubeMock()
    render(<KeyframeDetail hit={HIT} onClose={vi.fn()} onFindSimilar={vi.fn()} />)

    await waitFor(() => expect(screen.getByRole('button', { name: /play video/i })).toBeEnabled())
    await user.click(screen.getByRole('button', { name: /play video/i }))
    await waitFor(() =>
      expect(screen.getByRole('button', { name: /get frame index/i })).toBeEnabled(),
    )

    fakeYouTubeTime = 1203.483
    act(() => {
      lastYouTubeOptions?.events?.onStateChange?.({
        target: lastYouTubePlayer as FakeYouTubePlayer,
        data: 2,
      })
    })

    expect(screen.getByText('Current: 1203.483 s')).toBeInTheDocument()
  })

  it('calculates and displays the current frame index from YouTube timestamp and video FPS', async () => {
    const user = userEvent.setup()
    installYouTubeMock()
    render(<KeyframeDetail hit={HIT} onClose={vi.fn()} onFindSimilar={vi.fn()} />)

    await waitFor(() => expect(screen.getByRole('button', { name: /play video/i })).toBeEnabled())
    await user.click(screen.getByRole('button', { name: /play video/i }))
    await waitFor(() =>
      expect(screen.getByRole('button', { name: /get frame index/i })).toBeEnabled(),
    )

    fakeYouTubeTime = 130.215
    await user.click(screen.getByRole('button', { name: /get frame index/i }))

    expect(screen.getByRole('dialog', { name: /video player/i })).toBeInTheDocument()
    expect(screen.getAllByText('Current: 130.215 s')).toHaveLength(2)
    expect(screen.getByText('FPS: 25')).toBeInTheDocument()
    expect(screen.getByText('Frame Index: 3255')).toBeInTheDocument()
  })

  it('shows a useful error when FPS metadata is missing', async () => {
    const user = userEvent.setup()
    installYouTubeMock()
    vi.mocked(getVideo).mockResolvedValueOnce({
      video_id: 'L21_V001',
      watch_url: 'https://youtube.com/watch?v=L21_V001',
      title: 'Title L21_V001',
      author: 'VTV',
      publish_date: '01/01/2026',
      duration: 600,
      fps: 0,
      num_keyframes: 1,
      s_v: 0,
      e_v: 0,
    })
    render(<KeyframeDetail hit={HIT} onClose={vi.fn()} onFindSimilar={vi.fn()} />)

    await waitFor(() => expect(screen.getByRole('button', { name: /play video/i })).toBeEnabled())
    await user.click(screen.getByRole('button', { name: /play video/i }))
    await waitFor(() =>
      expect(screen.getByRole('button', { name: /get frame index/i })).toBeEnabled(),
    )

    await user.click(screen.getByRole('button', { name: /get frame index/i }))

    expect(await screen.findByText('FPS metadata not found for L21_V001')).toBeInTheDocument()
  })
})
