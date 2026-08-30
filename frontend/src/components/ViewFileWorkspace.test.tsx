import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { getViewFileFrameResolution } from '../api/client'
import type { ViewFileFrameResolution } from '../api/types'
import { ViewFileWorkspace } from './ViewFileWorkspace'

vi.mock('../api/client', () => ({
  getViewFileFrameResolution: vi.fn(),
  getVideo: vi.fn().mockResolvedValue({
    video_id: 'L21_V001',
    watch_url: 'https://youtube.com/watch?v=abc123',
    title: 'View file video',
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

interface FakeYouTubePlayer {
  getCurrentTime: ReturnType<typeof vi.fn>
  seekTo: ReturnType<typeof vi.fn>
  cueVideoById: ReturnType<typeof vi.fn>
  loadVideoById: ReturnType<typeof vi.fn>
  mute: ReturnType<typeof vi.fn>
  unMute: ReturnType<typeof vi.fn>
  pauseVideo: ReturnType<typeof vi.fn>
  destroy: ReturnType<typeof vi.fn>
}

interface FakeYouTubeOptions {
  events?: {
    onReady?: (event: { target: FakeYouTubePlayer }) => void
  }
}

let lastYouTubePlayer: FakeYouTubePlayer | null = null

function installYouTubeMock(): void {
  lastYouTubePlayer = null
  const Player = vi.fn(function PlayerMock(
    _element: HTMLElement | string,
    options: FakeYouTubeOptions,
  ) {
    const player: FakeYouTubePlayer = {
      getCurrentTime: vi.fn(() => 746.5),
      seekTo: vi.fn(),
      cueVideoById: vi.fn(),
      loadVideoById: vi.fn(),
      mute: vi.fn(),
      unMute: vi.fn(),
      pauseVideo: vi.fn(),
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
        PLAYING: 1,
        PAUSED: 2,
      },
    },
  })
}

function localFrame(videoId: string, frameIdx: number): ViewFileFrameResolution {
  return {
    video_id: videoId,
    frame_idx: frameIdx,
    timestamp_seconds: frameIdx / 25,
    source: 'local',
    keyframe_id: `${videoId}_f${frameIdx}`,
    thumbnail_url: `/api/keyframes/${videoId}_f${frameIdx}/thumbnail`,
    image_url: `/api/keyframes/${videoId}_f${frameIdx}/image`,
    youtube_url: null,
    fps: 25,
    error_message: null,
  }
}

function mockRect(element: Element): void {
  vi.spyOn(element, 'getBoundingClientRect').mockReturnValue({
    x: 0,
    y: 0,
    top: 0,
    left: 0,
    right: 200,
    bottom: 100,
    width: 200,
    height: 100,
    toJSON: () => ({}),
  })
}

describe('ViewFileWorkspace', () => {
  beforeEach(() => {
    vi.mocked(getViewFileFrameResolution).mockReset()
    vi.mocked(getViewFileFrameResolution).mockImplementation((videoId: string, frameIdx: number) =>
      Promise.resolve(localFrame(videoId, frameIdx)),
    )
    delete window.YT
    delete window.onYouTubeIframeAPIReady
  })

  it('renders one local frame card per KIS CSV row and opens the shared image viewer', async () => {
    const user = userEvent.setup()
    render(<ViewFileWorkspace />)

    await user.upload(
      screen.getByLabelText('CSV file upload'),
      new File(['L21_VLOCAL,10\n'], 'query-1-kis.csv', { type: 'text/csv' }),
    )

    await waitFor(() =>
      expect(getViewFileFrameResolution).toHaveBeenCalledWith(
        'L21_VLOCAL',
        10,
        expect.any(AbortSignal),
      ),
    )
    expect(screen.getByAltText('L21_VLOCAL 10')).toHaveAttribute(
      'src',
      '/api/keyframes/L21_VLOCAL_f10/thumbnail',
    )
    expect(screen.getByText('#1')).toBeInTheDocument()
    expect(screen.getByText('L21_VLOCAL')).toBeInTheDocument()
    expect(screen.getByText('10')).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: /Open L21_VLOCAL frame 10/i }))

    expect(screen.getByRole('dialog', { name: /keyframe detail/i })).toBeInTheDocument()
    expect(screen.getByAltText('L21_VLOCAL @ 0.4s')).toHaveAttribute(
      'src',
      '/api/keyframes/L21_VLOCAL_f10/image',
    )
  })

  it('uses a lightweight YouTube placeholder and creates the player only after click', async () => {
    const user = userEvent.setup()
    vi.mocked(getViewFileFrameResolution).mockResolvedValue({
      video_id: 'L21_VYT',
      frame_idx: 22395,
      timestamp_seconds: 746.5,
      source: 'youtube',
      keyframe_id: null,
      thumbnail_url: null,
      image_url: null,
      youtube_url: 'https://youtube.com/watch?v=abc123',
      fps: 30,
      error_message: null,
    })
    const { container } = render(<ViewFileWorkspace />)

    await user.upload(
      screen.getByLabelText('CSV file upload'),
      new File(['L21_VYT,22395\n'], 'query-1-kis.csv', { type: 'text/csv' }),
    )

    await screen.findByText('YouTube')
    expect(container.querySelector('iframe')).toBeNull()

    installYouTubeMock()
    await user.click(screen.getByRole('button', { name: /Open L21_VYT frame 22395/i }))

    expect(screen.getByRole('dialog', { name: /keyframe detail/i })).toBeInTheDocument()
    expect(screen.getAllByText(/Time: 746\.5s/).length).toBeGreaterThan(0)
    await waitFor(() =>
      expect(lastYouTubePlayer?.loadVideoById).toHaveBeenCalledWith({
        videoId: 'abc123',
        startSeconds: 746.5,
      }),
    )
    await waitFor(() => expect(lastYouTubePlayer?.pauseVideo).toHaveBeenCalled())
    expect(lastYouTubePlayer?.seekTo).toHaveBeenCalledWith(746.5, true)
    expect(lastYouTubePlayer?.unMute).toHaveBeenCalled()
  })

  it('renders missing YouTube as a neutral unavailable card and opens a graceful viewer state', async () => {
    const user = userEvent.setup()
    vi.mocked(getViewFileFrameResolution).mockResolvedValue({
      video_id: 'L21_VMISS',
      frame_idx: 10,
      timestamp_seconds: null,
      source: 'unavailable',
      keyframe_id: null,
      thumbnail_url: null,
      image_url: null,
      youtube_url: null,
      fps: null,
      error_message: 'Youtube Not Available',
    })
    render(<ViewFileWorkspace />)

    await user.upload(
      screen.getByLabelText('CSV file upload'),
      new File(['L21_VMISS,10\n'], 'query-1-kis.csv', { type: 'text/csv' }),
    )

    await screen.findByText('Youtube Not Available')
    await user.click(screen.getByRole('button', { name: /Open L21_VMISS frame 10/i }))

    expect(screen.getByRole('dialog', { name: /keyframe detail/i })).toBeInTheDocument()
    expect(screen.getAllByText('Youtube Not Available').length).toBeGreaterThan(0)
  })

  it('does not truncate a 100-row KIS CSV', async () => {
    const user = userEvent.setup()
    const rows = Array.from({ length: 100 }, (_, index) => `L21_V100,${index + 1}`).join('\n')
    render(<ViewFileWorkspace />)

    await user.upload(
      screen.getByLabelText('CSV file upload'),
      new File([`${rows}\n`], 'query-100-kis.csv', { type: 'text/csv' }),
    )

    await waitFor(() =>
      expect(
        screen.getAllByRole('button', { name: /^Open L21_V100 frame /i }),
      ).toHaveLength(100),
    )
    expect(screen.getByText('#100')).toBeInTheDocument()
    expect(screen.getByText('KIS · 100 rows')).toBeInTheDocument()
  })

  it('keeps VQA answer groups while using the same one-card frame grid', async () => {
    const user = userEvent.setup()
    render(<ViewFileWorkspace />)

    await user.upload(
      screen.getByLabelText('CSV file upload'),
      new File(
        ['video_id,frame_idx,answer\nL21_VQA,10,12 kg\nL21_VQA,20,15 kg\n'],
        'query-qa.csv',
        { type: 'text/csv' },
      ),
    )

    expect(await screen.findByText('12 kg')).toBeInTheDocument()
    expect(screen.getByText('15 kg')).toBeInTheDocument()
    await waitFor(() =>
      expect(
        screen.getAllByRole('button', { name: /^Open L21_VQA frame /i }),
      ).toHaveLength(2),
    )
  })

  it('sorts uploaded files by natural question number until the list is manually reordered', async () => {
    const user = userEvent.setup()
    render(<ViewFileWorkspace />)

    await user.upload(screen.getByLabelText('CSV file upload'), [
      new File(['L21_VSORT,10\n'], 'query-p2-10-kis.csv', { type: 'text/csv' }),
      new File(['L21_VSORT,2\n'], 'query-p2-2-kis.csv', { type: 'text/csv' }),
      new File(['L21_VSORT,1\n'], 'query-p2-1-kis.csv', { type: 'text/csv' }),
      new File(['L21_VSORT,9\n'], 'query-p2-9-kis.csv', { type: 'text/csv' }),
    ])

    await waitFor(() =>
      expect(screen.getAllByTestId('view-file-name').map((node) => node.textContent)).toEqual([
        'query-p2-1-kis.csv',
        'query-p2-2-kis.csv',
        'query-p2-9-kis.csv',
        'query-p2-10-kis.csv',
      ]),
    )
  })

  it('keeps frame range notes independent for each loaded CSV', async () => {
    const user = userEvent.setup()
    render(<ViewFileWorkspace />)

    await user.upload(screen.getByLabelText('CSV file upload'), [
      new File(['L21_VRANGE,10\n'], 'query-p2-1-kis.csv', { type: 'text/csv' }),
      new File(['L21_VRANGE,20\n'], 'query-p2-2-kis.csv', { type: 'text/csv' }),
    ])

    await screen.findByRole('heading', { name: 'query-p2-1-kis.csv' })
    await user.type(screen.getByLabelText('Frame range start'), '100')
    await user.type(screen.getByLabelText('Frame range end'), '150')

    await user.click(screen.getByRole('button', { name: 'Select query-p2-2-kis.csv' }))
    expect((screen.getByLabelText('Frame range start') as HTMLInputElement).value).toBe('')
    await user.type(screen.getByLabelText('Frame range start'), '220')
    await user.type(screen.getByLabelText('Frame range end'), '200')
    expect(screen.getByRole('alert')).toHaveTextContent(/less than or equal/i)

    await user.click(screen.getByRole('button', { name: 'Select query-p2-1-kis.csv' }))
    expect((screen.getByLabelText('Frame range start') as HTMLInputElement).value).toBe('100')
    expect((screen.getByLabelText('Frame range end') as HTMLInputElement).value).toBe('150')
  })

  it('supports manual file reorder by drag handle while keeping the active CSV and range note', async () => {
    const user = userEvent.setup()
    render(<ViewFileWorkspace />)

    await user.upload(screen.getByLabelText('CSV file upload'), [
      new File(['L21_VDRAG,1\n'], 'query-p2-1-kis.csv', { type: 'text/csv' }),
      new File(['L21_VDRAG,2\n'], 'query-p2-2-kis.csv', { type: 'text/csv' }),
      new File(['L21_VDRAG,4\n'], 'query-p2-4-kis.csv', { type: 'text/csv' }),
    ])

    await user.click(screen.getByRole('button', { name: 'Select query-p2-4-kis.csv' }))
    await user.type(screen.getByLabelText('Frame range start'), '400')

    const source = screen.getByTestId('view-file-item-query-p2-4-kis.csv')
    const target = screen.getByTestId('view-file-item-query-p2-2-kis.csv')
    const sourceId = source.getAttribute('data-file-id')
    if (!sourceId) throw new Error('Expected source file id')
    mockRect(target)
    fireEvent.drop(target, {
      clientY: 1,
      dataTransfer: {
        getData: vi.fn(() => sourceId),
      },
    })

    await waitFor(() =>
      expect(screen.getAllByTestId('view-file-name').map((node) => node.textContent)).toEqual([
        'query-p2-1-kis.csv',
        'query-p2-4-kis.csv',
        'query-p2-2-kis.csv',
      ]),
    )
    expect(screen.getByRole('heading', { name: 'query-p2-4-kis.csv' })).toBeInTheDocument()
    expect((screen.getByLabelText('Frame range start') as HTMLInputElement).value).toBe('400')
  })

  it('reorders frames inside the active CSV by dragging a frame handle', async () => {
    const user = userEvent.setup()
    render(<ViewFileWorkspace />)

    await user.upload(
      screen.getByLabelText('CSV file upload'),
      new File(['L21_VORDER,10\nL21_VORDER,20\nL21_VORDER,30\n'], 'query-p2-1-kis.csv', {
        type: 'text/csv',
      }),
    )

    await waitFor(() =>
      expect(screen.getAllByRole('button', { name: /^Open L21_VORDER frame /i })).toHaveLength(3),
    )
    const source = screen.getByTestId('view-frame-item-3')
    const target = screen.getByTestId('view-frame-item-1')
    const sourceId = source.getAttribute('data-row-id')
    if (!sourceId) throw new Error('Expected source row id')
    mockRect(target)
    fireEvent.drop(target, {
      clientX: 1,
      dataTransfer: {
        getData: vi.fn(() => sourceId),
      },
    })

    await waitFor(() =>
      expect(
        screen
          .getAllByRole('button', { name: /^Open L21_VORDER frame /i })
          .map((button) => button.getAttribute('aria-label')),
      ).toEqual([
        'Open L21_VORDER frame 30',
        'Open L21_VORDER frame 10',
        'Open L21_VORDER frame 20',
      ]),
    )
  })

  it('exports a ZIP with current CSV contents and an ordered View File manifest', async () => {
    const user = userEvent.setup()
    const createObjectURL = vi.fn((blob: Blob) => {
      void blob
      return 'blob:view-file-export'
    })
    const revokeObjectURL = vi.fn()
    Object.defineProperty(URL, 'createObjectURL', {
      configurable: true,
      value: createObjectURL,
    })
    Object.defineProperty(URL, 'revokeObjectURL', {
      configurable: true,
      value: revokeObjectURL,
    })
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {})
    render(<ViewFileWorkspace />)

    await user.upload(screen.getByLabelText('CSV file upload'), [
      new File(['L21_VZIP,650\n'], 'query-p2-4-kis.csv', { type: 'text/csv' }),
      new File(['L21_VZIP,700\n'], 'query-p2-4-kis.csv', { type: 'text/csv' }),
    ])
    await screen.findByRole('heading', { name: 'query-p2-4-kis.csv' })
    await user.type(screen.getByLabelText('Frame range start'), '650')
    await user.type(screen.getByLabelText('Frame range end'), '900')

    await user.click(screen.getByRole('button', { name: 'Export ZIP' }))

    expect(createObjectURL).toHaveBeenCalledTimes(1)
    const blob = createObjectURL.mock.calls[0]?.[0]
    expect(blob).toBeInstanceOf(Blob)
    if (!blob) throw new Error('Expected ZIP blob')
    expect(blob.type).toBe('application/zip')
    const zipText = await blob.text()
    expect(zipText).toContain('submissions/query-p2-4-kis.csv')
    expect(zipText).toContain('submissions/query-p2-4-kis-2.csv')
    expect(zipText).toContain('submissions/view_file_manifest.json')
    expect(zipText).toContain('"range_start": 650')
    expect(zipText).toContain('"range_end": 900')
    expect(zipText).toContain('L21_VZIP,650')
    expect(zipText).toContain('L21_VZIP,700')
  })

  it('exports reordered frame order in the ZIP CSV content', async () => {
    const user = userEvent.setup()
    const createObjectURL = vi.fn((blob: Blob) => {
      void blob
      return 'blob:view-file-export'
    })
    Object.defineProperty(URL, 'createObjectURL', {
      configurable: true,
      value: createObjectURL,
    })
    Object.defineProperty(URL, 'revokeObjectURL', {
      configurable: true,
      value: vi.fn(),
    })
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {})
    render(<ViewFileWorkspace />)

    await user.upload(
      screen.getByLabelText('CSV file upload'),
      new File(['L21_VZIP,650\nL21_VZIP,700\n'], 'query-p2-1-kis.csv', {
        type: 'text/csv',
      }),
    )

    await waitFor(() =>
      expect(screen.getAllByRole('button', { name: /^Open L21_VZIP frame /i })).toHaveLength(2),
    )
    const source = screen.getByTestId('view-frame-item-2')
    const target = screen.getByTestId('view-frame-item-1')
    const sourceId = source.getAttribute('data-row-id')
    if (!sourceId) throw new Error('Expected source row id')
    mockRect(target)
    fireEvent.drop(target, {
      clientX: 1,
      dataTransfer: {
        getData: vi.fn(() => sourceId),
      },
    })

    await user.click(screen.getByRole('button', { name: 'Export ZIP' }))

    const blob = createObjectURL.mock.calls[0]?.[0]
    expect(blob).toBeInstanceOf(Blob)
    if (!blob) throw new Error('Expected ZIP blob')
    const zipText = await blob.text()
    expect(zipText).toContain('L21_VZIP,700\nL21_VZIP,650')
  })
})
