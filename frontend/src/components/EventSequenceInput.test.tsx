import { render, screen } from '@testing-library/react'
import { useState } from 'react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { EMPTY_TRAKE_EVENT, type TrakeEventDraft } from '../hooks/useTrake'
import { EventSequenceInput, type EventSequenceInputProps } from './EventSequenceInput'

// TRAKE's input UI (PRD §F6/US-4): "tôi nhập N mô tả sự kiện theo thứ tự".
// Each event is its own block in fixed position order, and each block carries
// all four KIS channels because the backend runs one Stage 1 call per event.
describe('EventSequenceInput', () => {
  const TWO_BLANK: TrakeEventDraft[] = [{ ...EMPTY_TRAKE_EVENT }, { ...EMPTY_TRAKE_EVENT }]

  function renderInput(overrides: Partial<EventSequenceInputProps> = {}) {
    return render(
      <EventSequenceInput
        events={TWO_BLANK}
        onEventFieldChange={vi.fn()}
        onLocalPhraseChange={vi.fn()}
        onAddLocalPhrase={vi.fn()}
        onRemoveLocalPhrase={vi.fn()}
        onAddEvent={vi.fn()}
        onRemoveEvent={vi.fn()}
        threshold="0"
        onThresholdChange={vi.fn()}
        onSubmit={vi.fn()}
        {...overrides}
      />,
    )
  }

  it('renders a labeled block per event, in order', () => {
    renderInput({
      events: [
        { ...EMPTY_TRAKE_EVENT, text: 'người đàn ông mở cửa' },
        { ...EMPTY_TRAKE_EVENT, text: 'xe máy chạy qua' },
      ],
    })

    expect(screen.getByText('Event 1')).toBeInTheDocument()
    expect(screen.getByText('Event 2')).toBeInTheDocument()
    expect(screen.getByLabelText('event 1 text')).toHaveValue('người đàn ông mở cửa')
    expect(screen.getByLabelText('event 2 text')).toHaveValue('xe máy chạy qua')
  })

  it('renders Text, OCR, Title and ASR boxes for every event', () => {
    renderInput()

    for (const index of [1, 2]) {
      expect(screen.getByLabelText(`event ${index} text`)).toBeInTheDocument()
      expect(screen.getByLabelText(`event ${index} ocr`)).toBeInTheDocument()
      expect(screen.getByLabelText(`event ${index} title`)).toBeInTheDocument()
      expect(screen.getByLabelText(`event ${index} asr`)).toBeInTheDocument()
    }
  })

  it('shows each channel s current value independently', () => {
    renderInput({
      events: [
        {
          text: 'chay len',
          ocr: 'ty so 0-0',
          title: 'chung ket',
          asr: 'mo man',
          textTopK: '250',
          localPhrases: [],
        },
      ],
    })

    expect(screen.getByLabelText('event 1 text')).toHaveValue('chay len')
    expect(screen.getByLabelText('event 1 ocr')).toHaveValue('ty so 0-0')
    expect(screen.getByLabelText('event 1 title')).toHaveValue('chung ket')
    expect(screen.getByLabelText('event 1 asr')).toHaveValue('mo man')
    expect(screen.getByLabelText('event 1 text top k')).toHaveValue(250)
  })

  it('lets the operator set a per-event Text Top K instead of the fixed default', async () => {
    const user = userEvent.setup()
    const onEventFieldChange = vi.fn()
    renderInput({ onEventFieldChange })

    await user.type(screen.getByLabelText('event 1 text top k'), '3')

    expect(onEventFieldChange).toHaveBeenCalledWith(0, 'textTopK', '3')
  })

  it('shows the default Top K as a placeholder, not a forced value', () => {
    renderInput()

    expect(screen.getByLabelText('event 1 text top k')).toHaveValue(null)
    expect(screen.getByLabelText('event 1 text top k')).toHaveAttribute('placeholder', '300')
  })

  it('reports the row index and the channel when typing', async () => {
    const user = userEvent.setup()
    const onEventFieldChange = vi.fn()
    renderInput({ onEventFieldChange })

    await user.type(screen.getByLabelText('event 2 ocr'), 'a')

    expect(onEventFieldChange).toHaveBeenCalledWith(1, 'ocr', 'a')
  })

  it('disables only the ASR box when ASR is not ready', () => {
    renderInput({ asrDisabled: true, asrReadinessMessage: 'ASR chưa sẵn sàng' })

    expect(screen.getByLabelText('event 1 asr')).toBeDisabled()
    expect(screen.getByLabelText('event 1 text')).toBeEnabled()
    expect(screen.getByLabelText('event 1 ocr')).toBeEnabled()
    expect(screen.getByText('ASR chưa sẵn sàng')).toBeInTheDocument()
  })

  it('calls onAddEvent when the add-row button is clicked', async () => {
    const user = userEvent.setup()
    const onAddEvent = vi.fn()
    renderInput({ onAddEvent })

    await user.click(screen.getByRole('button', { name: /\+ Add event/i }))

    expect(onAddEvent).toHaveBeenCalledTimes(1)
  })

  it('calls onRemoveEvent with the correct index', async () => {
    const user = userEvent.setup()
    const onRemoveEvent = vi.fn()
    renderInput({
      events: [
        { ...EMPTY_TRAKE_EVENT, text: 'a' },
        { ...EMPTY_TRAKE_EVENT, text: 'b' },
        { ...EMPTY_TRAKE_EVENT, text: 'c' },
      ],
      onRemoveEvent,
    })

    await user.click(screen.getAllByRole('button', { name: /remove event/i })[1] as HTMLElement)

    expect(onRemoveEvent).toHaveBeenCalledWith(1)
  })

  it('hides the remove button when only one event row remains', () => {
    renderInput({ events: [{ ...EMPTY_TRAKE_EVENT, text: 'a' }] })

    expect(screen.queryByRole('button', { name: /remove event/i })).not.toBeInTheDocument()
  })

  it('shows per-event candidate counts once a search has returned', () => {
    renderInput({
      eventDiagnostics: [
        { event_index: 0, kis_candidate_count: 120, kis_video_count: 34, has_text: true },
      ],
      candidateVideoCount: 42,
      droppedVideoCount: 7,
    })

    expect(screen.getByText(/120 keyframes/)).toBeInTheDocument()
    expect(screen.getByText(/42 video ứng viên/)).toBeInTheDocument()
    expect(screen.getByText(/7 bị loại/)).toBeInTheDocument()
  })

  it('renders and edits the score threshold input', async () => {
    const user = userEvent.setup()
    const onThresholdChange = vi.fn()
    function ThresholdHarness() {
      const [threshold, setThreshold] = useState('1.25')
      return (
        <EventSequenceInput
          events={TWO_BLANK}
          onEventFieldChange={vi.fn()}
          onLocalPhraseChange={vi.fn()}
          onAddLocalPhrase={vi.fn()}
          onRemoveLocalPhrase={vi.fn()}
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
    renderInput({ onSubmit })

    await user.click(screen.getByRole('button', { name: /Search TRAKE/i }))

    expect(onSubmit).toHaveBeenCalledTimes(1)
  })

  it('disables the submit button while a search is in flight', () => {
    renderInput({ isLoading: true })

    expect(screen.getByRole('button', { name: /Search TRAKE/i })).toBeDisabled()
  })
})
