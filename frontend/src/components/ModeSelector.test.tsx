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
})
