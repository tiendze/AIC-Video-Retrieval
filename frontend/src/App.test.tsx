import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { findImageCandidates, rewriteQuery, trakeSearch } from './api/client'
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

describe('App', () => {
  it('preserves the typed query when switching search mode', async () => {
    const user = userEvent.setup()
    render(<App />)

    const input = screen.getByLabelText('query')
    await user.type(input, 'người đàn ông áo đỏ')

    await user.click(screen.getByRole('radio', { name: /OCR/i }))

    expect(input).toHaveValue('người đàn ông áo đỏ')
  })

  it('renders the semantic/OCR mode selector and the search input on load', () => {
    render(<App />)

    expect(screen.getByLabelText('query')).toBeInTheDocument()
    expect(screen.getByRole('radio', { name: /Semantic/i })).toBeInTheDocument()
    expect(screen.getByRole('radio', { name: /OCR/i })).toBeInTheDocument()
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

    await user.type(screen.getByLabelText('query'), 'Labubu')
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

    await user.type(screen.getByLabelText('query'), 'Labubu')
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

    await user.type(screen.getByLabelText('query'), 'Labubu')
    await user.click(screen.getByRole('button', { name: /AI enhancer/i }))

    await waitFor(() => expect(screen.getByText('Query rewrite failed')).toBeInTheDocument())
  })

  it('shows a readable error message when the web image search request itself fails', async () => {
    const user = userEvent.setup()
    vi.mocked(findImageCandidates).mockRejectedValue(new Error('network down'))
    render(<App />)

    await user.type(screen.getByLabelText('query'), 'Labubu')
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
    expect(screen.queryByLabelText('query')).not.toBeInTheDocument()
  })

  it('shows TRAKE results after searching in TRAKE mode', async () => {
    const user = userEvent.setup()
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

    await user.click(screen.getByRole('button', { name: /tìm chuỗi/i }))

    await waitFor(() => expect(screen.getByText('L21_V001')).toBeInTheDocument())
  })
})
