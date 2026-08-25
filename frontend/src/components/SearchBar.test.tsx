import { fireEvent, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import type { ComponentProps } from 'react'
import { describe, expect, it, vi } from 'vitest'
import type { AsrGate, KisQueryRow } from '../api/types'
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
      stage2Queries: [],
      finalTopK: 20,
      onGlobalTextQueryChange: vi.fn(),
      onTextCandidateKChange: vi.fn(),
      onOcrQueryChange: vi.fn(),
      onOcrCandidateKChange: vi.fn(),
      onTitleQueryChange: vi.fn(),
      onTitleCandidateKChange: vi.fn(),
      onAsrQueryChange: vi.fn(),
      onAsrVideoCandidateKChange: vi.fn(),
      onStage2QueryTextChange: vi.fn(),
      onStage2QueryTypeChange: vi.fn(),
      onAbePairAttributeChange: vi.fn(),
      onAbePairObjectChange: vi.fn(),
      onAddAbePair: vi.fn(),
      onRemoveAbePair: vi.fn(),
      onAddStage2Query: vi.fn(),
      onRemoveStage2Query: vi.fn(),
      onFinalTopKChange: vi.fn(),
      onRetrieveCandidates: vi.fn(),
      onRerankCandidates: vi.fn(),
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

  it('disables Stage 1 OCR input and Stage 2 OCR option when Elasticsearch is unavailable', () => {
    const rows: KisQueryRow[] = [{ id: 'q1', type: 'text', text: '', pairs: [] }]
    renderSearchBar({ ocrDisabled: true, candidatePoolSize: 3, stage2Queries: rows })

    expect(screen.getByLabelText('stage 1 ocr')).toBeDisabled()
    expect(screen.getByRole('option', { name: 'OCR' })).toBeDisabled()
  })

  it('reveals Stage 2 after candidate retrieval and wires subquery controls', async () => {
    const user = userEvent.setup()
    const rows: KisQueryRow[] = [
      { id: 'q1', type: 'text', text: 'night scene', pairs: [] },
      { id: 'q2', type: 'ocr', text: 'SAIGON', pairs: [] },
    ]
    const props = renderSearchBar({
      candidatePoolSize: 347,
      textCandidateCount: 300,
      ocrCandidateCount: 100,
      stage2Queries: rows,
    })

    expect(screen.getByText('Candidate Pool: 347')).toBeInTheDocument()
    expect(screen.getByLabelText('stage 2 query 1')).toHaveValue('night scene')
    expect(screen.getByLabelText('stage 2 query 2')).toHaveValue('SAIGON')

    await user.selectOptions(screen.getByLabelText('stage 2 query type 1'), 'ocr')
    await user.type(screen.getByLabelText('stage 2 query 1'), '!')
    await user.click(screen.getByRole('button', { name: /\+ Add subquery/i }))
    await user.click(screen.getByRole('button', { name: /remove stage 2 query 2/i }))

    expect(props.onStage2QueryTypeChange).toHaveBeenCalledWith('q1', 'ocr')
    expect(props.onStage2QueryTextChange).toHaveBeenCalledWith('q1', 'night scene!')
    expect(props.onAddStage2Query).toHaveBeenCalledTimes(1)
    expect(props.onRemoveStage2Query).toHaveBeenCalledWith('q2')
  })

  it('renders ABE pair controls and wires pair edits', async () => {
    const user = userEvent.setup()
    const rows: KisQueryRow[] = [
      {
        id: 'q1',
        type: 'abe',
        text: '',
        pairs: [{ id: 'p1', attribute: 'red', object: 'packet' }],
      },
    ]
    const props = renderSearchBar({
      candidatePoolSize: 10,
      stage2Queries: rows,
    })

    await user.type(screen.getByLabelText('abe attribute 1.1'), '!')
    await user.type(screen.getByLabelText('abe object 1.1'), '!')
    await user.click(screen.getByRole('button', { name: /\+ Add pair/i }))
    await user.click(screen.getByRole('button', { name: /remove abe pair 1.1/i }))

    expect(props.onAbePairAttributeChange).toHaveBeenCalledWith('q1', 'p1', 'red!')
    expect(props.onAbePairObjectChange).toHaveBeenCalledWith('q1', 'p1', 'packet!')
    expect(props.onAddAbePair).toHaveBeenCalledWith('q1')
    expect(props.onRemoveAbePair).toHaveBeenCalledWith('q1', 'p1')
  })

  it('disables rerank and shows guidance when no Stage 2 subquery is active', () => {
    renderSearchBar({ candidatePoolSize: 10, stage2Queries: [] })

    expect(screen.getByRole('button', { name: 'Rerank Candidates' })).toBeDisabled()
    expect(
      screen.getByText('Add at least one Stage 2 constraint to calculate final reranking.'),
    ).toBeInTheDocument()
  })
})
