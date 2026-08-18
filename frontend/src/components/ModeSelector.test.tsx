import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { ModeSelector } from './ModeSelector'

describe('ModeSelector', () => {
  it('renders KIS and TRAKE without a standalone OCR option', () => {
    render(<ModeSelector mode="kis" onChange={vi.fn()} />)

    expect(screen.getByRole('radio', { name: /KIS/i })).toBeInTheDocument()
    expect(screen.getByRole('radio', { name: /TRAKE/i })).toBeInTheDocument()
    expect(screen.queryByRole('radio', { name: /OCR/i })).not.toBeInTheDocument()
  })

  it('shows KIS as checked when mode is "kis"', () => {
    render(<ModeSelector mode="kis" onChange={vi.fn()} />)

    expect(screen.getByRole('radio', { name: /KIS/i })).toBeChecked()
  })

  it('renders TRAKE as a selectable, non-disabled option', () => {
    render(<ModeSelector mode="kis" onChange={vi.fn()} />)

    expect(screen.getByRole('radio', { name: /TRAKE/i })).not.toBeDisabled()
  })

  it('calls onChange with "dante" when the TRAKE option is clicked', async () => {
    const user = userEvent.setup()
    const onChange = vi.fn()
    render(<ModeSelector mode="kis" onChange={onChange} />)

    await user.click(screen.getByRole('radio', { name: /TRAKE/i }))

    expect(onChange).toHaveBeenCalledWith('dante')
  })

  it('shows TRAKE as checked when mode is "dante"', () => {
    render(<ModeSelector mode="dante" onChange={vi.fn()} />)

    expect(screen.getByRole('radio', { name: /TRAKE/i })).toBeChecked()
  })
})
