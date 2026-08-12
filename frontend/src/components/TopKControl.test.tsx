import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { TopKControl } from './TopKControl'

// RED reproducer: the user found only 20 results ever showed up because
// useSearch.ts hard-coded top_k=20 with no UI control -- this exposes
// top_k (bounded 1..100, matching backend max_top_k) as a real input.
//
// Commits on blur/Enter, not per keystroke: a `value={topK}` bound
// straight to the parent's committed number would fight the user mid-type
// (e.g. re-clamping to "1" the instant they type the leading digit of
// "100"). The component keeps its own draft text and only clamps+emits
// once typing is done.

describe('TopKControl', () => {
  it('shows the current top_k value', () => {
    render(<TopKControl topK={20} onChange={vi.fn()} />)

    expect(screen.getByLabelText(/top k/i)).toHaveValue(20)
  })

  it('commits the typed value on blur when within range', async () => {
    const user = userEvent.setup()
    const onChange = vi.fn()
    render(<TopKControl topK={20} onChange={onChange} />)

    const input = screen.getByLabelText(/top k/i)
    await user.clear(input)
    await user.type(input, '50')
    await user.tab()

    expect(onChange).toHaveBeenCalledWith(50)
  })

  it('clamps a value above the backend max_top_k (100) on blur', async () => {
    const user = userEvent.setup()
    const onChange = vi.fn()
    render(<TopKControl topK={20} onChange={onChange} />)

    const input = screen.getByLabelText(/top k/i)
    await user.clear(input)
    await user.type(input, '500')
    await user.tab()

    expect(onChange).toHaveBeenCalledWith(100)
  })

  it('clamps a value below 1 on blur', async () => {
    const user = userEvent.setup()
    const onChange = vi.fn()
    render(<TopKControl topK={20} onChange={onChange} />)

    const input = screen.getByLabelText(/top k/i)
    await user.clear(input)
    await user.type(input, '0')
    await user.tab()

    expect(onChange).toHaveBeenCalledWith(1)
  })

  it('commits on Enter without requiring blur', async () => {
    const user = userEvent.setup()
    const onChange = vi.fn()
    render(<TopKControl topK={20} onChange={onChange} />)

    const input = screen.getByLabelText(/top k/i)
    await user.clear(input)
    await user.type(input, '30{Enter}')

    expect(onChange).toHaveBeenCalledWith(30)
  })

  it('does not call onChange when blurring without changing the value', async () => {
    const user = userEvent.setup()
    const onChange = vi.fn()
    render(<TopKControl topK={20} onChange={onChange} />)

    await user.click(screen.getByLabelText(/top k/i))
    await user.tab()

    expect(onChange).not.toHaveBeenCalled()
  })
})
