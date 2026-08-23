import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import {
  findImageCandidates,
  getHealth,
  getVideo,
  getVideoKeyframes,
  rerankKisCandidates,
  rewriteQuery,
  retrieveKisCandidates,
  trakeSearch,
} from './api/client'
import App from './App'

// RED reproducer for the plan §P1.6 invariant that survived from paper's
// functional description into PRD §F9's acceptance criteria: "Đổi mode
// không mất truy vấn đang gõ." No network calls are made -- api/client is
// mocked so this stays a pure UI-wiring test.
vi.mock('./api/client', () => ({
  search: vi.fn().mockResolvedValue({ results: [] }),
  searchByImage: vi.fn().mockResolvedValue({ results: [] }),
  getVideo: vi.fn(),
  getVideoKeyframes: vi.fn(),
  fetchImageBlob: vi.fn(),
  // Never resolves -- these tests aren't about health/degraded-mode
  // (plan §P1.8 has its own coverage in ModeSelector.test.tsx/
  // StatusBar.test.tsx), so useHealth() should just sit at its initial
  // "not checked yet" (null) state throughout.
  getHealth: vi.fn().mockReturnValue(new Promise(() => {})),
  retrieveKisCandidates: vi.fn().mockResolvedValue({
    candidate_ids: [],
    candidate_pool_size: 0,
    text_candidate_count: 0,
    ocr_candidate_count: 0,
    results: [],
  }),
  rerankKisCandidates: vi.fn().mockResolvedValue({
    candidate_pool_size: 0,
    final_top_k: 0,
    results: [],
  }),
  // QUEST (plan §P2.5) -- never resolves by default so tests unrelated to
  // QUEST don't need to care about it; the two QUEST-specific tests below
  // override these with a real resolved value.
  rewriteQuery: vi.fn().mockReturnValue(new Promise(() => {})),
  findImageCandidates: vi.fn().mockReturnValue(new Promise(() => {})),
  searchByCandidate: vi.fn().mockReturnValue(new Promise(() => {})),
  // TRAKE/DANTE (plan §P2.2) -- never resolves by default; the
  // TRAKE-specific test below overrides it with a real resolved value.
  trakeSearch: vi.fn().mockReturnValue(new Promise(() => {})),
  ApiError: class ApiError extends Error {},
}))

const STUBBED_PROPS = ['clientHeight', 'clientWidth', 'offsetHeight', 'offsetWidth'] as const

function stubViewportSizeGlobally(size: number): () => void {
  const originals = STUBBED_PROPS.map(
    (prop) => [prop, Object.getOwnPropertyDescriptor(HTMLElement.prototype, prop)] as const,
  )
  for (const prop of STUBBED_PROPS) {
    Object.defineProperty(HTMLElement.prototype, prop, { configurable: true, value: size })
  }
  return () => {
    for (const [prop, descriptor] of originals) {
      if (descriptor) Object.defineProperty(HTMLElement.prototype, prop, descriptor)
    }
  }
}

describe('App', () => {
  it('leaves ASR controls enabled without an unavailable message before health loads', () => {
    render(<App />)

    expect(screen.getByLabelText('stage 1 asr')).toBeEnabled()
    expect(screen.getByLabelText('asr candidate videos')).toBeEnabled()
    expect(screen.queryByText(/ASR transcript index|ASR is unavailable/)).not.toBeInTheDocument()
  })

  it('preserves the typed query when switching search mode', async () => {
    const user = userEvent.setup()
    render(<App />)

    const input = screen.getByLabelText('text q0')
    await user.type(input, 'người đàn ông áo đỏ')

    await user.click(screen.getByRole('radio', { name: /TRAKE/i }))
    await user.click(screen.getByRole('radio', { name: /KIS/i }))

    expect(screen.getByLabelText('text q0')).toHaveValue('người đàn ông áo đỏ')
  })

  it('renders the KIS mode selector and query editor on load', () => {
    render(<App />)

    expect(screen.getByLabelText('text q0')).toBeInTheDocument()
    expect(screen.getByRole('radio', { name: /KIS/i })).toBeInTheDocument()
    expect(screen.queryByRole('radio', { name: /OCR/i })).not.toBeInTheDocument()
  })

  // QUEST branch 1 (plan §P2.5, PRD §F5): clicking the AI enhancer must show
  // both the original and rewritten query, not search automatically.
  it('shows the query rewrite panel after clicking the AI enhancer button', async () => {
    const user = userEvent.setup()
    vi.mocked(rewriteQuery).mockResolvedValue({
      status: 'rewritten',
      original_query: 'Labubu',
      rewritten_query: 'một con búp bê tai thỏ màu hồng',
    })
    render(<App />)

    await user.type(screen.getByLabelText('text q0'), 'Labubu')
    await user.click(screen.getByRole('button', { name: /AI enhancer/i }))

    await waitFor(() =>
      expect(screen.getByText('một con búp bê tai thỏ màu hồng')).toBeInTheDocument(),
    )
    expect(screen.getByText('Labubu')).toBeInTheDocument()
  })

  // QUEST branch 2 (plan §P2.5, PRD §F5): clicking "Tìm ảnh mẫu trên web"
  // must show a picker to choose from -- never auto-pick a result.
  it('shows the image candidate picker after clicking the web image search button', async () => {
    const user = userEvent.setup()
    vi.mocked(findImageCandidates).mockResolvedValue({
      status: 'candidates',
      candidate_set_id: 'set-1',
      candidates: [{ candidate_id: 'c1', thumbnail_url: 'https://x/t1.jpg', title: 'Labubu doll' }],
      fallback_hits: [],
    })
    render(<App />)

    await user.type(screen.getByLabelText('text q0'), 'Labubu')
    await user.click(screen.getByRole('button', { name: /tìm ảnh mẫu/i }))

    await waitFor(() => expect(screen.getByRole('dialog')).toBeInTheDocument())
    expect(screen.getByAltText('Labubu doll')).toBeInTheDocument()
  })

  // Found via manual browser check (2026-08-15): clicking AI enhancer while
  // the request itself fails (not a backend fallback -- see
  // useQueryRewrite.test.tsx for that path) left no visible feedback at
  // all, violating .claude/docs/error-handling.md ("Lỗi hiện ra UI phải
  // đọc được"). error-handling.md's own wording is for backend-side
  // handling, but the same rule obviously governs the UI that surfaces it.
  it('shows a readable error message when the query rewrite request itself fails', async () => {
    const user = userEvent.setup()
    vi.mocked(rewriteQuery).mockRejectedValue(new Error('network down'))
    render(<App />)

    await user.type(screen.getByLabelText('text q0'), 'Labubu')
    await user.click(screen.getByRole('button', { name: /AI enhancer/i }))

    await waitFor(() => expect(screen.getByText('Query rewrite failed')).toBeInTheDocument())
  })

  it('shows a readable error message when the web image search request itself fails', async () => {
    const user = userEvent.setup()
    vi.mocked(findImageCandidates).mockRejectedValue(new Error('network down'))
    render(<App />)

    await user.type(screen.getByLabelText('text q0'), 'Labubu')
    await user.click(screen.getByRole('button', { name: /tìm ảnh mẫu/i }))

    await waitFor(() => expect(screen.getByText('Web image search failed')).toBeInTheDocument())
  })

  // TRAKE/DANTE (plan §P2.2, PRD §F6/US-4): selecting the TRAKE mode swaps
  // the single-query flow for the N-ordered-events flow -- these are
  // fundamentally different request shapes, not just a filter on the same
  // query box.
  it('shows the event-sequence input after selecting TRAKE mode', async () => {
    const user = userEvent.setup()
    render(<App />)

    await user.click(screen.getByRole('radio', { name: /TRAKE/i }))

    expect(screen.getByLabelText('event 1')).toBeInTheDocument()
    expect(screen.getByLabelText('TRAKE selection export')).toBeInTheDocument()
    expect(screen.queryByLabelText('text q0')).not.toBeInTheDocument()
  })

  it('adds a selected KIS result to the sidebar selection tab', async () => {
    const user = userEvent.setup()
    const restoreViewportSize = stubViewportSizeGlobally(600)
    vi.mocked(retrieveKisCandidates).mockResolvedValue({
      candidate_ids: ['L21_V001_f10'],
      candidate_pool_size: 1,
      text_candidate_count: 1,
      ocr_candidate_count: 0,
      results: [
        {
          keyframe_id: 'L21_V001_f10',
          video_id: 'L21_V001',
          score: 0.9,
          frame_idx: 10,
          timestamp: 1.0,
          thumbnail_path: 't.jpg',
          thumbnail_url: '/api/keyframes/L21_V001_f10/thumbnail',
          image_url: '/api/keyframes/L21_V001_f10/image',
        },
      ],
    })
    try {
      render(<App />)

      await user.type(screen.getByLabelText('text q0'), 'boat')
      await user.click(screen.getByRole('button', { name: 'Retrieve Candidates' }))
      await waitFor(() => expect(screen.getByText('Select KIS')).toBeInTheDocument())

      await user.click(screen.getByText('Select KIS'))
      await user.click(screen.getByRole('tab', { name: 'Selection' }))

      expect(screen.getByLabelText('KIS selection export')).toBeInTheDocument()
      expect(screen.getByText('Selected Results')).toBeInTheDocument()
      expect(screen.getByText('frame 10')).toBeInTheDocument()
      expect(screen.getByText('Will export: query-1-kis.csv')).toBeInTheDocument()
    } finally {
      restoreViewportSize()
    }
  })

  it('shows TRAKE results after searching in TRAKE mode', async () => {
    const user = userEvent.setup()
    vi.mocked(getVideo).mockResolvedValue({
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
    })
    vi.mocked(getVideoKeyframes).mockResolvedValue({ keyframes: [] })
    vi.mocked(trakeSearch).mockResolvedValue({
      results: [
        {
          video_id: 'L21_V001',
          score: 0.8,
          keyframes: [
            {
              keyframe_id: 'L21_V001_f10',
              frame_idx: 10,
              timestamp: 1.0,
              thumbnail_url: '/api/keyframes/L21_V001_f10/thumbnail',
              image_url: '/api/keyframes/L21_V001_f10/image',
            },
          ],
        },
      ],
    })
    render(<App />)
    await user.click(screen.getByRole('radio', { name: /TRAKE/i }))
    await user.type(screen.getByLabelText('event 1'), 'người đàn ông mở cửa')
    await user.type(screen.getByLabelText('event 2'), 'xe máy chạy qua')

    await user.click(screen.getByRole('button', { name: /Search TRAKE/i }))

    await waitFor(() => expect(screen.getByText('L21_V001')).toBeInTheDocument())
    expect(screen.getByText(/1 result · \d+ms/)).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: /open event 1 frame 10/i }))

    expect(screen.getByRole('dialog', { name: /keyframe detail/i })).toBeInTheDocument()
    expect(screen.getByAltText('L21_V001 @ 1s')).toHaveAttribute(
      'src',
      '/api/keyframes/L21_V001_f10/image',
    )
    expect(screen.getByRole('button', { name: /show frame 10/i })).toHaveAttribute(
      'data-highlighted',
      'true',
    )
  })

  it('shows the KIS score inspector for the selected result', async () => {
    const user = userEvent.setup()
    vi.mocked(retrieveKisCandidates).mockResolvedValue({
      candidate_ids: ['L21_V001_f10'],
      candidate_pool_size: 1,
      text_candidate_count: 1,
      ocr_candidate_count: 0,
      results: [
        {
          keyframe_id: 'L21_V001_f10',
          video_id: 'L21_V001',
          score: 0.9,
          frame_idx: 10,
          timestamp: 1.0,
          thumbnail_path: 't.jpg',
          thumbnail_url: '/api/keyframes/L21_V001_f10/thumbnail',
          image_url: '/api/keyframes/L21_V001_f10/image',
          stage1: {
            source_modalities: ['text'],
            text: { query: 'boat', raw_cosine: 0.9, rank: 1 },
          },
        },
      ],
    })
    vi.mocked(rerankKisCandidates).mockResolvedValue({
      candidate_pool_size: 1,
      final_top_k: 1,
      results: [
        {
          keyframe_id: 'L21_V001_f10',
          video_id: 'L21_V001',
          score: 1,
          final_score: 1,
          active_modalities: 1,
          frame_idx: 10,
          timestamp: 1.0,
          thumbnail_path: 't.jpg',
          thumbnail_url: '/api/keyframes/L21_V001_f10/thumbnail',
          image_url: '/api/keyframes/L21_V001_f10/image',
          modalities: {
            ocr: {
              geometric_mean: 1,
              score: 1,
              queries: [
                {
                  query_index: 0,
                  query: 'VINAMILK',
                  raw_bm25: 7.5,
                  bm25_reference: 7.5,
                  bm25_normalized: 1,
                  coverage: 1,
                  query_score: 1,
                  rank: 1,
                  rank_percentile: 1,
                },
              ],
            },
          },
        },
      ],
    })
    render(<App />)
    await user.type(screen.getByLabelText('text q0'), 'boat')
    await user.click(screen.getByRole('button', { name: 'Retrieve Candidates' }))
    await waitFor(() =>
      expect(screen.getByRole('button', { name: /\+ Add subquery/i })).toBeInTheDocument(),
    )
    await user.click(screen.getByRole('button', { name: /\+ Add subquery/i }))
    await user.selectOptions(screen.getByLabelText('stage 2 query type 1'), 'ocr')
    await user.type(screen.getByLabelText('stage 2 query 1'), 'VINAMILK')

    await user.click(screen.getByRole('button', { name: 'Rerank Candidates' }))

    await waitFor(() => expect(screen.getByLabelText('KIS score details')).toBeInTheDocument())
    expect(screen.getByText('Final Geometric Mean')).toBeInTheDocument()
    expect(screen.getByText('Candidate retrieval only')).toBeInTheDocument()
    expect(screen.getByText('BM25 raw')).toBeInTheDocument()
    expect(screen.getByText('OCR query score')).toBeInTheDocument()
    expect(screen.getByText('7.500')).toBeInTheDocument()
  })

  it('closes the detail viewer with Escape while keeping the KIS score rail mounted', async () => {
    const user = userEvent.setup()
    const restoreViewportSize = stubViewportSizeGlobally(600)
    vi.mocked(getVideo).mockResolvedValue({
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
    })
    vi.mocked(getVideoKeyframes).mockResolvedValue({ keyframes: [] })
    vi.mocked(retrieveKisCandidates).mockResolvedValue({
      candidate_ids: ['L21_V001_f10'],
      candidate_pool_size: 1,
      text_candidate_count: 1,
      ocr_candidate_count: 0,
      results: [
        {
          keyframe_id: 'L21_V001_f10',
          video_id: 'L21_V001',
          score: 0.9,
          frame_idx: 10,
          timestamp: 1.0,
          thumbnail_path: 't.jpg',
          thumbnail_url: '/api/keyframes/L21_V001_f10/thumbnail',
          image_url: '/api/keyframes/L21_V001_f10/image',
          stage1: {
            source_modalities: ['text'],
            text: { query: 'night scene', raw_cosine: 0.9, rank: 1 },
          },
        },
      ],
    })
    try {
      render(<App />)
      await user.type(screen.getByLabelText('text q0'), 'night scene')
      await user.click(screen.getByRole('button', { name: 'Retrieve Candidates' }))
      await waitFor(() =>
        expect(screen.getByRole('button', { name: /details/i })).toBeInTheDocument(),
      )

      await user.click(screen.getByRole('button', { name: /details/i }))
      expect(screen.getByRole('dialog', { name: /keyframe detail/i })).toBeInTheDocument()

      await user.keyboard('{Escape}')

      expect(screen.queryByRole('dialog', { name: /keyframe detail/i })).not.toBeInTheDocument()
      expect(screen.getByLabelText('KIS score details')).toBeInTheDocument()
    } finally {
      restoreViewportSize()
    }
  })

  it.each([
    {
      asr_status: 'unreachable' as const,
      elasticsearch_ok: false,
      message: 'Elasticsearch is unreachable; ASR is unavailable.',
    },
    {
      asr_status: 'unavailable' as const,
      elasticsearch_ok: true,
      message: 'The ASR transcript index is unavailable.',
    },
    {
      asr_status: 'empty' as const,
      elasticsearch_ok: true,
      message: 'The ASR transcript index is empty.',
    },
  ])('disables ASR controls for $asr_status health with the exact reason', async (state) => {
    vi.mocked(getHealth).mockResolvedValue({
      status: 'degraded',
      milvus_ok: true,
      elasticsearch_ok: state.elasticsearch_ok,
      mongodb_ok: true,
      asr_status: state.asr_status,
      asr_ready: false,
      asr_segment_count: state.asr_status === 'empty' ? 0 : null,
    })

    render(<App />)

    await waitFor(() => expect(screen.getByText(state.message)).toBeInTheDocument())
    expect(screen.getByLabelText('stage 1 asr')).toBeDisabled()
    expect(screen.getByLabelText('asr candidate videos')).toBeDisabled()
    if (state.elasticsearch_ok) {
      expect(screen.getByLabelText('stage 1 ocr')).toBeEnabled()
      expect(screen.getByLabelText('ocr candidates')).toBeEnabled()
    }
  })
})
