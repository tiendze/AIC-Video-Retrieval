import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
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
})
