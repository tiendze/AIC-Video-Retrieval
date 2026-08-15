import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { QueryRewritePanel } from './QueryRewritePanel'

// RED reproducer for QUEST branch 1's UI (plan §P2.5, PRD §F5 acceptance
// criteria: "Luôn hiển thị truy vấn gốc VÀ truy vấn đã viết lại"). The
// operator picks which one to actually search with -- onSearch is called
// with plain text, never triggered automatically.
describe('QueryRewritePanel', () => {
  it('shows both queries and lets the operator search with the rewritten one', async () => {
    const user = userEvent.setup()
    const onSearch = vi.fn()
    render(
      <QueryRewritePanel
        result={{
          status: 'rewritten',
          original_query: 'Labubu',
          rewritten_query: 'một con búp bê tai thỏ màu hồng',
        }}
        onSearch={onSearch}
        onDismiss={vi.fn()}
      />,
    )

    expect(screen.getByText('Labubu')).toBeInTheDocument()
    expect(screen.getByText('một con búp bê tai thỏ màu hồng')).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: /dùng câu này/i }))

    expect(onSearch).toHaveBeenCalledWith('một con búp bê tai thỏ màu hồng')
  })

  it('lets the operator search with the original query instead', async () => {
    const user = userEvent.setup()
    const onSearch = vi.fn()
    render(
      <QueryRewritePanel
        result={{
          status: 'rewritten',
          original_query: 'Labubu',
          rewritten_query: 'một con búp bê tai thỏ màu hồng',
        }}
        onSearch={onSearch}
        onDismiss={vi.fn()}
      />,
    )

    await user.click(screen.getByRole('button', { name: /dùng câu gốc/i }))

    expect(onSearch).toHaveBeenCalledWith('Labubu')
  })

  it('shows a fallback message and no rewritten-query button when Gemini failed', () => {
    render(
      <QueryRewritePanel
        result={{ status: 'fallback', original_query: 'Labubu', rewritten_query: null }}
        onSearch={vi.fn()}
        onDismiss={vi.fn()}
      />,
    )

    expect(screen.getByText('Labubu')).toBeInTheDocument()
    expect(screen.getByText(/không thể viết lại/i)).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /dùng câu này/i })).not.toBeInTheDocument()
  })

  it('calls onDismiss when the close button is clicked', async () => {
    const user = userEvent.setup()
    const onDismiss = vi.fn()
    render(
      <QueryRewritePanel
        result={{ status: 'fallback', original_query: 'Labubu', rewritten_query: null }}
        onSearch={vi.fn()}
        onDismiss={onDismiss}
      />,
    )

    await user.click(screen.getByRole('button', { name: /close/i }))

    expect(onDismiss).toHaveBeenCalledTimes(1)
  })
})
