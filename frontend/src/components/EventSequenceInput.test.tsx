import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { EventSequenceInput } from './EventSequenceInput'

// RED reproducer for DANTE/TRAKE's input UI (plan §P2.2, PRD §F6/US-4):
// "tôi nhập N mô tả sự kiện theo thứ tự". Each event is its own labeled
// row so the operator can see and edit the time order directly.
describe('EventSequenceInput', () => {
  it('renders one labeled textbox per event, in order', () => {
    render(
      <EventSequenceInput
        events={['người đàn ông mở cửa', 'xe máy chạy qua']}
        onEventChange={vi.fn()}
        onAddEvent={vi.fn()}
        onRemoveEvent={vi.fn()}
        onSubmit={vi.fn()}
      />,
    )

    expect(screen.getByLabelText('event 1')).toHaveValue('người đàn ông mở cửa')
    expect(screen.getByLabelText('event 2')).toHaveValue('xe máy chạy qua')
  })

  it('calls onEventChange with the row index when typing', async () => {
    const user = userEvent.setup()
    const onEventChange = vi.fn()
    render(
      <EventSequenceInput
        events={['', '']}
        onEventChange={onEventChange}
        onAddEvent={vi.fn()}
        onRemoveEvent={vi.fn()}
        onSubmit={vi.fn()}
      />,
    )

    await user.type(screen.getByLabelText('event 2'), 'a')

    expect(onEventChange).toHaveBeenCalledWith(1, 'a')
  })

  it('calls onAddEvent when the add-row button is clicked', async () => {
    const user = userEvent.setup()
    const onAddEvent = vi.fn()
    render(
      <EventSequenceInput
        events={['', '']}
        onEventChange={vi.fn()}
        onAddEvent={onAddEvent}
        onRemoveEvent={vi.fn()}
        onSubmit={vi.fn()}
      />,
    )

    await user.click(screen.getByRole('button', { name: /thêm sự kiện/i }))

    expect(onAddEvent).toHaveBeenCalledTimes(1)
  })

  it('calls onRemoveEvent with the correct index', async () => {
    const user = userEvent.setup()
    const onRemoveEvent = vi.fn()
    render(
      <EventSequenceInput
        events={['a', 'b', 'c']}
        onEventChange={vi.fn()}
        onAddEvent={vi.fn()}
        onRemoveEvent={onRemoveEvent}
        onSubmit={vi.fn()}
      />,
    )

    await user.click(screen.getAllByRole('button', { name: /xoá sự kiện/i })[1] as HTMLElement)

    expect(onRemoveEvent).toHaveBeenCalledWith(1)
  })

  it('hides the remove button when only one event row remains', () => {
    render(
      <EventSequenceInput
        events={['a']}
        onEventChange={vi.fn()}
        onAddEvent={vi.fn()}
        onRemoveEvent={vi.fn()}
        onSubmit={vi.fn()}
      />,
    )

    expect(screen.queryByRole('button', { name: /xoá sự kiện/i })).not.toBeInTheDocument()
  })

  it('calls onSubmit when the form is submitted', async () => {
    const user = userEvent.setup()
    const onSubmit = vi.fn()
    render(
      <EventSequenceInput
        events={['a', 'b']}
        onEventChange={vi.fn()}
        onAddEvent={vi.fn()}
        onRemoveEvent={vi.fn()}
        onSubmit={onSubmit}
      />,
    )

    await user.click(screen.getByRole('button', { name: /tìm chuỗi/i }))

    expect(onSubmit).toHaveBeenCalledTimes(1)
  })

  it('disables the submit button while a search is in flight', () => {
    render(
      <EventSequenceInput
        events={['a', 'b']}
        onEventChange={vi.fn()}
        onAddEvent={vi.fn()}
        onRemoveEvent={vi.fn()}
        onSubmit={vi.fn()}
        isLoading
      />,
    )

    expect(screen.getByRole('button', { name: /tìm chuỗi/i })).toBeDisabled()
  })
})
