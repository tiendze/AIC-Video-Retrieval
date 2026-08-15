import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { WebImageSearch } from './WebImageSearch'

// RED reproducer for QUEST branch 2's trigger (plan §P2.5, PRD §F5): kicks
// off POST /api/quest/image-candidates for the operator's current query.
describe('WebImageSearch', () => {
  it('calls onSearch with the current query when clicked', async () => {
    const user = userEvent.setup()
    const onSearch = vi.fn()
    render(<WebImageSearch query="Labubu" onSearch={onSearch} />)

    await user.click(screen.getByRole('button', { name: /tìm ảnh mẫu/i }))

    expect(onSearch).toHaveBeenCalledWith('Labubu')
  })

  it('is disabled when the query is blank', () => {
    render(<WebImageSearch query="   " onSearch={vi.fn()} />)

    expect(screen.getByRole('button', { name: /tìm ảnh mẫu/i })).toBeDisabled()
  })

  it('is disabled while a search is already in flight', () => {
    render(<WebImageSearch query="Labubu" onSearch={vi.fn()} disabled />)

    expect(screen.getByRole('button', { name: /tìm ảnh mẫu/i })).toBeDisabled()
  })
})
