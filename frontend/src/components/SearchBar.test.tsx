import { fireEvent, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import type { ComponentProps } from 'react'
import { describe, expect, it, vi } from 'vitest'
import type { KisQueryRow } from '../api/types'
import { SearchBar } from './SearchBar'

describe('SearchBar', () => {
  function renderSearchBar(overrides: Partial<ComponentProps<typeof SearchBar>> = {}) {
    const props: ComponentProps<typeof SearchBar> = {
      globalTextQuery: '',
      textCandidateK: 300,
      ocrQuery: '',
      ocrCandidateK: 100,
      candidatePoolSize: null,
      textCandidateCount: 0,
      ocrCandidateCount: 0,
      stage2Queries: [],
      finalTopK: 20,
      onGlobalTextQueryChange: vi.fn(),
      onTextCandidateKChange: vi.fn(),
      onOcrQueryChange: vi.fn(),
      onOcrCandidateKChange: vi.fn(),
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

  it('keeps Text and OCR candidate counts independent', () => {
    const props = renderSearchBar()

    fireEvent.change(screen.getByLabelText('text candidates'), { target: { value: '250' } })
    fireEvent.change(screen.getByLabelText('ocr candidates'), { target: { value: '75' } })

    expect(props.onTextCandidateKChange).toHaveBeenLastCalledWith(250)
    expect(props.onOcrCandidateKChange).toHaveBeenLastCalledWith(75)
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
