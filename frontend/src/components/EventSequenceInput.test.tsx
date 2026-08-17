import { render, screen } from '@testing-library/react'
import { useState } from 'react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { EventSequenceInput, type EventSequenceInputProps } from './EventSequenceInput'

// RED reproducer for DANTE/TRAKE's input UI (plan §P2.2, PRD §F6/US-4):
// "tôi nhập N mô tả sự kiện theo thứ tự". Each event is its own labeled
// row so the operator can see and edit the time order directly.
describe('EventSequenceInput', () => {
  function renderInput(overrides: Partial<EventSequenceInputProps> = {}) {
    return render(
      <EventSequenceInput
        events={['', '']}
        onEventChange={vi.fn()}
        onAddEvent={vi.fn()}
        onRemoveEvent={vi.fn()}
        threshold="0"
        onThresholdChange={vi.fn()}
        onSubmit={vi.fn()}
        {...overrides}
      />,
    )
  }

  it('renders one labeled textbox per event, in order', () => {
    renderInput({ events: ['người đàn ông mở cửa', 'xe máy chạy qua'] })

    expect(screen.getByLabelText('event 1')).toHaveValue('người đàn ông mở cửa')
    expect(screen.getByLabelText('event 2')).toHaveValue('xe máy chạy qua')
  })

  it('calls onEventChange with the row index when typing', async () => {
    const user = userEvent.setup()
    const onEventChange = vi.fn()
    renderInput({ onEventChange })

    await user.type(screen.getByLabelText('event 2'), 'a')

    expect(onEventChange).toHaveBeenCalledWith(1, 'a')
  })

  it('calls onAddEvent when the add-row button is clicked', async () => {
    const user = userEvent.setup()
    const onAddEvent = vi.fn()
    renderInput({ onAddEvent })

    await user.click(screen.getByRole('button', { name: /thêm sự kiện/i }))

    expect(onAddEvent).toHaveBeenCalledTimes(1)
  })

  it('calls onRemoveEvent with the correct index', async () => {
    const user = userEvent.setup()
    const onRemoveEvent = vi.fn()
    renderInput({ events: ['a', 'b', 'c'], onRemoveEvent })

    await user.click(screen.getAllByRole('button', { name: /xoá sự kiện/i })[1] as HTMLElement)

    expect(onRemoveEvent).toHaveBeenCalledWith(1)
  })

  it('hides the remove button when only one event row remains', () => {
    renderInput({ events: ['a'] })

    expect(screen.queryByRole('button', { name: /xoá sự kiện/i })).not.toBeInTheDocument()
  })

  it('renders and edits the score threshold input', async () => {
    const user = userEvent.setup()
    const onThresholdChange = vi.fn()
    function ThresholdHarness() {
      const [threshold, setThreshold] = useState('1.25')
      return (
        <EventSequenceInput
          events={['', '']}
          onEventChange={vi.fn()}
          onAddEvent={vi.fn()}
          onRemoveEvent={vi.fn()}
          threshold={threshold}
          onThresholdChange={(value) => {
            setThreshold(value)
            onThresholdChange(value)
          }}
          onSubmit={vi.fn()}
        />
      )
    }
    render(<ThresholdHarness />)

    await user.clear(screen.getByLabelText('threshold'))
    await user.type(screen.getByLabelText('threshold'), '0.8')

    expect(onThresholdChange).toHaveBeenLastCalledWith('0.8')
  })

  it('calls onSubmit when the form is submitted', async () => {
    const user = userEvent.setup()
    const onSubmit = vi.fn()
    renderInput({ events: ['a', 'b'], onSubmit })

    await user.click(screen.getByRole('button', { name: /tìm chuỗi/i }))

    expect(onSubmit).toHaveBeenCalledTimes(1)
  })

  it('disables the submit button while a search is in flight', () => {
    renderInput({ events: ['a', 'b'], isLoading: true })

    expect(screen.getByRole('button', { name: /tìm chuỗi/i })).toBeDisabled()
  })
})
