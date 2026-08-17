import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { ModeSelector } from './ModeSelector'

// RED reproducer for plan §P1.8 (PRD T5): OCR must be unselectable while
// Elasticsearch is known to be down, so a person doesn't pick a mode that
// will just error -- the degraded-mode banner (StatusBar) explains why.
describe('ModeSelector', () => {
  it('disables the OCR option when ocrDisabled is true', () => {
    render(<ModeSelector mode="semantic" onChange={vi.fn()} ocrDisabled />)

    expect(screen.getByRole('radio', { name: /OCR/i })).toBeDisabled()
  })

  it('leaves the OCR option enabled by default', () => {
    render(<ModeSelector mode="semantic" onChange={vi.fn()} />)

    expect(screen.getByRole('radio', { name: /OCR/i })).not.toBeDisabled()
  })

  it('does not call onChange when clicking a disabled OCR option', async () => {
    const user = userEvent.setup()
    const onChange = vi.fn()
    render(<ModeSelector mode="semantic" onChange={onChange} ocrDisabled />)

    await user.click(screen.getByRole('radio', { name: /OCR/i }))

    expect(onChange).not.toHaveBeenCalled()
  })

  it('still allows switching to semantic when OCR is disabled', async () => {
    const user = userEvent.setup()
    const onChange = vi.fn()
    render(<ModeSelector mode="ocr" onChange={onChange} ocrDisabled />)

    await user.click(screen.getByRole('radio', { name: /Semantic/i }))

    expect(onChange).toHaveBeenCalledWith('semantic')
  })

  // TRAKE/DANTE (plan §P2.2) went live 2026-08-15 -- no longer the
  // disabled P2 placeholder from plan §P1.5.
  it('renders TRAKE as a selectable, non-disabled option', () => {
    render(<ModeSelector mode="semantic" onChange={vi.fn()} />)

    expect(screen.getByRole('radio', { name: /TRAKE/i })).not.toBeDisabled()
  })

  it('calls onChange with "dante" when the TRAKE option is clicked', async () => {
    const user = userEvent.setup()
    const onChange = vi.fn()
    render(<ModeSelector mode="semantic" onChange={onChange} />)

    await user.click(screen.getByRole('radio', { name: /TRAKE/i }))

    expect(onChange).toHaveBeenCalledWith('dante')
  })

  it('shows TRAKE as checked when mode is "dante"', () => {
    render(<ModeSelector mode="dante" onChange={vi.fn()} />)

    expect(screen.getByRole('radio', { name: /TRAKE/i })).toBeChecked()
  })
})
