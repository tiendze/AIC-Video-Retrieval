import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { MAX_OCR_QUERIES } from '../api/constants'
import { OcrQueryList } from './OcrQueryList'

function queriesOf(...texts: string[]) {
  return texts.map((text, index) => ({ id: `ocr-${index}`, text }))
}

// RED reproducer for the supplementary OCR conditions of Semantic mode:
// one primary Semantic Query plus N optional OCR rows the operator can add
// and remove. Rows are keyed by a stable id (not array index) so removing a
// middle row never re-associates the surviving rows' DOM nodes.
describe('OcrQueryList', () => {
  it('renders one input per query', () => {
    render(
      <OcrQueryList
        queries={queriesOf('VINMART', 'KHUYEN MAI')}
        onChange={vi.fn()}
        onAdd={vi.fn()}
        onRemove={vi.fn()}
      />,
    )

    expect(screen.getByLabelText('ocr query 1')).toHaveValue('VINMART')
    expect(screen.getByLabelText('ocr query 2')).toHaveValue('KHUYEN MAI')
  })

  it('calls onChange with the row id when typing', async () => {
    const user = userEvent.setup()
    const onChange = vi.fn()
    render(
      <OcrQueryList
        queries={queriesOf('')}
        onChange={onChange}
        onAdd={vi.fn()}
        onRemove={vi.fn()}
      />,
    )

    await user.type(screen.getByLabelText('ocr query 1'), 'V')

    expect(onChange).toHaveBeenCalledWith('ocr-0', 'V')
  })

  it('calls onAdd when the add button is clicked', async () => {
    const user = userEvent.setup()
    const onAdd = vi.fn()
    render(
      <OcrQueryList queries={queriesOf('')} onChange={vi.fn()} onAdd={onAdd} onRemove={vi.fn()} />,
    )

    await user.click(screen.getByRole('button', { name: /add ocr query/i }))

    expect(onAdd).toHaveBeenCalledTimes(1)
  })

  it('calls onRemove with the id of the clicked row, not its index', async () => {
    const user = userEvent.setup()
    const onRemove = vi.fn()
    render(
      <OcrQueryList
        queries={queriesOf('VINMART', 'KHUYEN MAI', '50%')}
        onChange={vi.fn()}
        onAdd={vi.fn()}
        onRemove={onRemove}
      />,
    )

    await user.click(screen.getByRole('button', { name: /xoá ocr query 2/i }))

    expect(onRemove).toHaveBeenCalledWith('ocr-1')
  })

  it('hides the remove button when only one row is left', () => {
    render(
      <OcrQueryList
        queries={queriesOf('VINMART')}
        onChange={vi.fn()}
        onAdd={vi.fn()}
        onRemove={vi.fn()}
      />,
    )

    expect(screen.queryByRole('button', { name: /xoá ocr query/i })).not.toBeInTheDocument()
  })

  it('disables the add button once the maximum number of rows is reached', () => {
    const queries = queriesOf(...Array.from({ length: MAX_OCR_QUERIES }, () => ''))
    render(
      <OcrQueryList queries={queries} onChange={vi.fn()} onAdd={vi.fn()} onRemove={vi.fn()} />,
    )

    expect(screen.getByRole('button', { name: /add ocr query/i })).toBeDisabled()
  })

  it('focuses the newly appended row', () => {
    const props = { onChange: vi.fn(), onAdd: vi.fn(), onRemove: vi.fn() }
    const { rerender } = render(<OcrQueryList queries={queriesOf('VINMART')} {...props} />)

    rerender(<OcrQueryList queries={queriesOf('VINMART', '')} {...props} />)

    expect(screen.getByLabelText('ocr query 2')).toHaveFocus()
  })

  it('does not add a row when Enter is pressed inside an OCR input', async () => {
    const user = userEvent.setup()
    const onAdd = vi.fn()
    render(
      <OcrQueryList queries={queriesOf('')} onChange={vi.fn()} onAdd={onAdd} onRemove={vi.fn()} />,
    )

    await user.type(screen.getByLabelText('ocr query 1'), 'VINMART{Enter}')

    expect(onAdd).not.toHaveBeenCalled()
  })
})
