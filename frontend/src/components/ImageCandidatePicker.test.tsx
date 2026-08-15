import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import type { ImageCandidate } from '../api/types'
import { ImageCandidatePicker } from './ImageCandidatePicker'

// RED reproducer for QUEST branch 2's picker (plan §P2.5, PRD §F5
// acceptance criteria: "Nhánh 2 cho phép người dùng CHỌN THỦ CÔNG ảnh
// mẫu, không tự động chọn ảnh đầu tiên"). Every candidate is rendered as
// its own explicit button -- there is no "pick first" affordance anywhere
// in this component.
const CANDIDATES: ImageCandidate[] = [
  { candidate_id: 'c1', thumbnail_url: 'https://x/t1.jpg', title: 'Labubu doll' },
  { candidate_id: 'c2', thumbnail_url: 'https://x/t2.jpg', title: 'Labubu keychain' },
]

describe('ImageCandidatePicker', () => {
  it('renders every candidate as a clickable image', () => {
    render(<ImageCandidatePicker candidates={CANDIDATES} onPick={vi.fn()} onDismiss={vi.fn()} />)

    expect(screen.getByAltText('Labubu doll')).toBeInTheDocument()
    expect(screen.getByAltText('Labubu keychain')).toBeInTheDocument()
  })

  it('calls onPick with the clicked candidate_id, not the first one by default', async () => {
    const user = userEvent.setup()
    const onPick = vi.fn()
    render(<ImageCandidatePicker candidates={CANDIDATES} onPick={onPick} onDismiss={vi.fn()} />)

    await user.click(screen.getByAltText('Labubu keychain'))

    expect(onPick).toHaveBeenCalledWith('c2')
    expect(onPick).toHaveBeenCalledTimes(1)
  })

  it('calls onDismiss when the close button is clicked', async () => {
    const user = userEvent.setup()
    const onDismiss = vi.fn()
    render(<ImageCandidatePicker candidates={CANDIDATES} onPick={vi.fn()} onDismiss={onDismiss} />)

    await user.click(screen.getByRole('button', { name: /close/i }))

    expect(onDismiss).toHaveBeenCalledTimes(1)
  })
})
