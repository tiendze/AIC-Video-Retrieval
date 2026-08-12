import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { useKeyboardShortcuts } from './useKeyboardShortcuts'

// RED reproducer for plan §P1.6's shortcut invariant: "Enter search · / focus
// ô nhập · ←/→ prev/next · 1/2/3 đổi mode · Esc đóng detail. Không kích hoạt
// khi con trỏ đang trong ô nhập (trừ Enter/Esc)."

function Harness(props: {
  onSubmit: () => void
  onFocusQuery: () => void
  onNext: () => void
  onPrev: () => void
  onSetMode: (index: number) => void
  onClose: () => void
}) {
  useKeyboardShortcuts(props)
  return <input aria-label="query" type="text" />
}

describe('useKeyboardShortcuts', () => {
  it('does not fire mode/nav shortcuts while typing in the query input', async () => {
    const user = userEvent.setup()
    const onSetMode = vi.fn()
    const onNext = vi.fn()
    render(
      <Harness
        onSubmit={vi.fn()}
        onFocusQuery={vi.fn()}
        onNext={onNext}
        onPrev={vi.fn()}
        onSetMode={onSetMode}
        onClose={vi.fn()}
      />,
    )

    const input = screen.getByLabelText('query')
    await user.click(input)
    await user.keyboard('1')
    await user.keyboard('{ArrowRight}')

    expect(onSetMode).not.toHaveBeenCalled()
    expect(onNext).not.toHaveBeenCalled()
  })

  it('still submits on Enter while typing in the query input', async () => {
    const user = userEvent.setup()
    const onSubmit = vi.fn()
    render(
      <Harness
        onSubmit={onSubmit}
        onFocusQuery={vi.fn()}
        onNext={vi.fn()}
        onPrev={vi.fn()}
        onSetMode={vi.fn()}
        onClose={vi.fn()}
      />,
    )

    await user.click(screen.getByLabelText('query'))
    await user.keyboard('{Enter}')

    expect(onSubmit).toHaveBeenCalledTimes(1)
  })

  it('still closes on Escape while typing in the query input', async () => {
    const user = userEvent.setup()
    const onClose = vi.fn()
    render(
      <Harness
        onSubmit={vi.fn()}
        onFocusQuery={vi.fn()}
        onNext={vi.fn()}
        onPrev={vi.fn()}
        onSetMode={vi.fn()}
        onClose={onClose}
      />,
    )

    await user.click(screen.getByLabelText('query'))
    await user.keyboard('{Escape}')

    expect(onClose).toHaveBeenCalledTimes(1)
  })

  it('fires mode shortcut when focus is outside the query input', async () => {
    const user = userEvent.setup()
    const onSetMode = vi.fn()
    render(
      <Harness
        onSubmit={vi.fn()}
        onFocusQuery={vi.fn()}
        onNext={vi.fn()}
        onPrev={vi.fn()}
        onSetMode={onSetMode}
        onClose={vi.fn()}
      />,
    )

    document.body.focus()
    await user.keyboard('2')

    expect(onSetMode).toHaveBeenCalledWith(1)
  })
})
