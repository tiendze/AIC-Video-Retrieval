import { fireEvent, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import type { ComponentProps } from 'react'
import { describe, expect, it, vi } from 'vitest'
import type { AsrGate } from '../api/types'
import { SearchBar } from './SearchBar'

const ASR_GATE: AsrGate = {
  query: 'thủy lợi',
  queries: ['thủy lợi'],
  match_mode: 'any',
  detected: true,
  asr_detected_any: true,
  matched_segment_count: 1,
  scanned_segment_count: 57,
  candidate_video_count: 1,
  asr_matches: [
    {
      query: 'thủy lợi',
      detected: true,
      matches: [
        {
          segment_id: 'V1_seg1',
          text: 'công trình thủy lợi',
          start_time: 151.2,
          end_time: 158.4,
        },
      ],
    },
  ],
  videos: [
    {
      video_id: 'V1',
      detected: true,
      asr_matches: [
        {
          query: 'thủy lợi',
          detected: true,
          matches: [
            {
              segment_id: 'V1_seg1',
              text: 'công trình thủy lợi',
              start_time: 151.2,
              end_time: 158.4,
            },
          ],
        },
      ],
      segments: [
        {
          segment_id: 'V1_seg1',
          text: 'công trình thủy lợi',
          start_time: 151.2,
          end_time: 158.4,
        },
      ],
    },
  ],
}

describe('SearchBar', () => {
  function renderSearchBar(overrides: Partial<ComponentProps<typeof SearchBar>> = {}) {
    const props: ComponentProps<typeof SearchBar> = {
      textQueries: [''],
      onTextQueryChange: vi.fn(),
      onAddTextQuery: vi.fn(),
      onRemoveTextQuery: vi.fn(),
      localPhrasesByQuery: [[]],
      onLocalPhraseChange: vi.fn(),
      onAddLocalPhrase: vi.fn(),
      onRemoveLocalPhrase: vi.fn(),
      globalTextQuery: '',
      textCandidateK: 300,
      ocrQuery: '',
      ocrCandidateK: 100,
      titleQuery: '',
      titleCandidateK: 100,
      asrQuery: '',
      asrVideoCandidateK: 20,
      asrGate: null,
      asrDisabled: false,
      asrReadinessMessage: null,
      candidatePoolSize: null,
      textCandidateCount: 0,
      ocrCandidateCount: 0,
      titleCandidateCount: 0,
      onGlobalTextQueryChange: vi.fn(),
      onTextCandidateKChange: vi.fn(),
      onOcrQueryChange: vi.fn(),
      onOcrCandidateKChange: vi.fn(),
      onTitleQueryChange: vi.fn(),
      onTitleCandidateKChange: vi.fn(),
      onAsrQueryChange: vi.fn(),
      onAsrVideoCandidateKChange: vi.fn(),
      onRetrieveCandidates: vi.fn(),
      onEnhance: vi.fn(),
      ...overrides,
    }
    if (overrides.globalTextQuery !== undefined && overrides.textQueries === undefined) {
      props.textQueries = [overrides.globalTextQuery]
    }
    render(<SearchBar {...props} />)
    return props
  }

  it('calls onGlobalTextQueryChange when typing Text Q0', async () => {
    const user = userEvent.setup()
    const props = renderSearchBar()

    await user.type(screen.getByLabelText('text q0'), 'a')

    expect(props.onGlobalTextQueryChange).toHaveBeenCalledWith('a')
  })

  it('calls onRetrieveCandidates when Stage 1 form is submitted', async () => {
    const user = userEvent.setup()
    const props = renderSearchBar({ globalTextQuery: 'boat' })

    await user.click(screen.getByRole('button', { name: 'Retrieve Candidates' }))

    expect(props.onRetrieveCandidates).toHaveBeenCalledTimes(1)
  })

  it('allows Stage 1 retrieval with only ASR input when transcripts are ready', async () => {
    const user = userEvent.setup()
    const props = renderSearchBar({ asrQuery: 'thủy lợi' })
    const retrieveButton = screen.getByRole('button', { name: 'Retrieve Candidates' })

    expect(retrieveButton).toBeEnabled()
    await user.click(retrieveButton)

    expect(props.onRetrieveCandidates).toHaveBeenCalledTimes(1)
  })

  it('keeps Text, OCR, and Title candidate counts independent', () => {
    const props = renderSearchBar()

    fireEvent.change(screen.getByLabelText('text candidates'), { target: { value: '250' } })
    fireEvent.change(screen.getByLabelText('ocr candidates'), { target: { value: '75' } })
    fireEvent.change(screen.getByLabelText('title candidates'), { target: { value: '30' } })

    expect(props.onTextCandidateKChange).toHaveBeenLastCalledWith(250)
    expect(props.onOcrCandidateKChange).toHaveBeenLastCalledWith(75)
    expect(props.onTitleCandidateKChange).toHaveBeenLastCalledWith(30)
  })

  it('does not cap Text candidates while keeping OCR and Title capped', () => {
    renderSearchBar()

    expect(screen.getByLabelText('text candidates')).not.toHaveAttribute('max')
    expect(screen.getByLabelText('ocr candidates')).toHaveAttribute('max', '1000')
    expect(screen.getByLabelText('title candidates')).toHaveAttribute('max', '1000')
  })

  it('adds and removes Stage 1 text query rows', async () => {
    const user = userEvent.setup()
    const props = renderSearchBar({ textQueries: ['dam', 'reservoir'] })

    await user.type(screen.getByLabelText('text q2'), '!')
    await user.click(screen.getByRole('button', { name: /\+ Add query/i }))
    await user.click(screen.getByRole('button', { name: /remove text query 2/i }))

    expect(props.onTextQueryChange).toHaveBeenCalledWith(1, 'reservoir!')
    expect(props.onAddTextQuery).toHaveBeenCalledTimes(1)
    expect(props.onRemoveTextQuery).toHaveBeenCalledWith(1)
  })

  it('edits local phrases for the matching text query row', async () => {
    const user = userEvent.setup()
    const props = renderSearchBar({
      textQueries: ['man holding a packet', 'woman beside him'],
      localPhrasesByQuery: [['man'], ['woman', 'red packet']],
    })

    await user.type(screen.getByLabelText('local phrase 2.2'), '!')
    await user.click(screen.getAllByRole('button', { name: /\+ Add local phrase/i })[1])
    await user.click(screen.getByRole('button', { name: /remove local phrase 1.1/i }))

    expect(props.onLocalPhraseChange).toHaveBeenCalledWith(1, 1, 'red packet!')
    expect(props.onAddLocalPhrase).toHaveBeenCalledWith(1)
    expect(props.onRemoveLocalPhrase).toHaveBeenCalledWith(0, 0)
    expect(screen.getByText(/Local phrases score visual regions/i)).toBeInTheDocument()
  })

  it('renders ASR controls and reports the active gated video count', () => {
    const props = renderSearchBar({
      asrQuery: 'thủy lợi',
      asrVideoCandidateK: 20,
      asrGate: { ...ASR_GATE, candidate_video_count: 12 },
      candidatePoolSize: 300,
      textCandidateCount: 300,
      ocrCandidateCount: 0,
      titleCandidateCount: 0,
    })

    expect(screen.getByLabelText('stage 1 asr')).toHaveValue('thủy lợi')
    expect(screen.getByLabelText('stage 1 asr')).toHaveAttribute('maxlength', '1000')
    expect(screen.getByLabelText('asr candidate videos')).toHaveValue(20)
    expect(screen.getByLabelText('asr candidate videos')).toHaveAttribute('max', '100')
    expect(screen.getByText('Text 300 · OCR 0 · Title OCR 0 · ASR 12 detected')).toBeInTheDocument()

    fireEvent.change(screen.getByLabelText('stage 1 asr'), {
      target: { value: 'new topic' },
    })
    fireEvent.change(screen.getByLabelText('asr candidate videos'), {
      target: { value: '10' },
    })
    expect(props.onAsrQueryChange).toHaveBeenLastCalledWith('new topic')
    expect(props.onAsrVideoCandidateKChange).toHaveBeenLastCalledWith(10)
  })

  it('keeps an active zero-hit ASR gate visible with revision guidance', () => {
    renderSearchBar({
      asrQuery: 'không có kết quả',
      asrGate: { ...ASR_GATE, candidate_video_count: 0, videos: [] },
      candidatePoolSize: 0,
    })

    expect(screen.getByText('Text 0 · OCR 0 · Title OCR 0 · ASR 0 detected')).toBeInTheDocument()
    expect(
      screen.getByText('No transcript video matched - clear or revise the ASR query.'),
    ).toBeInTheDocument()
  })

  it('disables only ASR controls with their readiness reason', () => {
    renderSearchBar({
      asrDisabled: true,
      asrReadinessMessage: 'The ASR transcript index is empty.',
    })

    expect(screen.getByLabelText('stage 1 asr')).toBeDisabled()
    expect(screen.getByLabelText('asr candidate videos')).toBeDisabled()
    expect(screen.getByText('The ASR transcript index is empty.')).toBeInTheDocument()
    expect(screen.getByLabelText('stage 1 ocr')).toBeEnabled()
    expect(screen.getByLabelText('ocr candidates')).toBeEnabled()
  })

  it('calls onEnhance with Text Q0 when the AI enhancer button is clicked', async () => {
    const user = userEvent.setup()
    const onEnhance = vi.fn()
    renderSearchBar({ globalTextQuery: 'Labubu', onEnhance })

    await user.click(screen.getByRole('button', { name: /AI enhancer/i }))

    expect(onEnhance).toHaveBeenCalledWith('Labubu')
  })

  it('disables the AI enhancer button when Text Q0 is blank', () => {
    renderSearchBar({ globalTextQuery: '   ' })

    expect(screen.getByRole('button', { name: /AI enhancer/i })).toBeDisabled()
  })

  it('disables Stage 1 OCR input when Elasticsearch is unavailable', () => {
    renderSearchBar({ ocrDisabled: true, candidatePoolSize: 3 })

    expect(screen.getByLabelText('stage 1 ocr')).toBeDisabled()
    expect(screen.queryByText(/Stage 2/i)).not.toBeInTheDocument()
  })

  it('keeps the active KIS flow on Stage 1 after candidate retrieval', () => {
    renderSearchBar({
      candidatePoolSize: 347,
      textCandidateCount: 300,
      ocrCandidateCount: 100,
    })

    expect(screen.getByText('Candidate Pool: 347 unique images')).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Rerank Candidates' })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /\+ Add subquery/i })).not.toBeInTheDocument()
  })
})
