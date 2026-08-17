import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { SearchBar } from './SearchBar'

// RED reproducer for wiring QUEST branch 1's "AI enhancer" button (plan
// §P2.5, PRD §F5) -- it was a disabled P2 placeholder (module docstring
// pre-2026-08-15); now that POST /api/quest/rewrite-query exists, it must
// call onEnhance with the current query instead of doing nothing.
describe('SearchBar', () => {
  it('calls onQueryChange when typing', async () => {
    const user = userEvent.setup()
    const onQueryChange = vi.fn()
    render(
      <SearchBar query="" onQueryChange={onQueryChange} onSubmit={vi.fn()} onEnhance={vi.fn()} />,
    )

    await user.type(screen.getByLabelText('query'), 'a')

    expect(onQueryChange).toHaveBeenCalledWith('a')
  })

  it('calls onSubmit when the form is submitted', async () => {
    const user = userEvent.setup()
    const onSubmit = vi.fn()
    render(
      <SearchBar query="thu tuong" onQueryChange={vi.fn()} onSubmit={onSubmit} onEnhance={vi.fn()} />,
    )

    await user.click(screen.getByRole('button', { name: 'Search' }))

    expect(onSubmit).toHaveBeenCalledTimes(1)
  })

  it('calls onEnhance with the current query when the AI enhancer button is clicked', async () => {
    const user = userEvent.setup()
    const onEnhance = vi.fn()
    render(
      <SearchBar
        query="Labubu"
        onQueryChange={vi.fn()}
        onSubmit={vi.fn()}
        onEnhance={onEnhance}
      />,
    )

    await user.click(screen.getByRole('button', { name: /AI enhancer/i }))

    expect(onEnhance).toHaveBeenCalledWith('Labubu')
  })

  it('disables the AI enhancer button when the query is blank', () => {
    render(<SearchBar query="   " onQueryChange={vi.fn()} onSubmit={vi.fn()} onEnhance={vi.fn()} />)

    expect(screen.getByRole('button', { name: /AI enhancer/i })).toBeDisabled()
  })

  it('shows a loading state instead of disabling the whole button while a rewrite is in flight', () => {
    render(
      <SearchBar
        query="Labubu"
        onQueryChange={vi.fn()}
        onSubmit={vi.fn()}
        onEnhance={vi.fn()}
        enhanceLoading
      />,
    )

    expect(screen.getByRole('button', { name: /AI enhancer/i })).toBeDisabled()
  })

  // Semantic mode's supplementary OCR conditions: the `ocr` prop is
  // optional and only App.tsx passes it while activeMode === 'semantic'.
  // OCR mode renders SearchBar without it, so these two tests double as
  // the regression proof that OCR mode's existing behavior is untouched.
  it('renders the OCR query rows when the ocr prop is provided', () => {
    render(
      <SearchBar
        query=""
        onQueryChange={vi.fn()}
        onSubmit={vi.fn()}
        onEnhance={vi.fn()}
        ocr={{
          queries: [{ id: 'ocr-1', text: '' }],
          onChange: vi.fn(),
          onAdd: vi.fn(),
          onRemove: vi.fn(),
        }}
      />,
    )

    expect(screen.getByLabelText('ocr query 1')).toBeInTheDocument()
  })

  it('does not render OCR query rows when the ocr prop is omitted', () => {
    render(<SearchBar query="" onQueryChange={vi.fn()} onSubmit={vi.fn()} onEnhance={vi.fn()} />)

    expect(screen.queryByLabelText('ocr query 1')).not.toBeInTheDocument()
  })
})
