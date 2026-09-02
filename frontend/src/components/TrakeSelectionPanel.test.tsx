import { fireEvent, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import type { TrakeVideoResultItem } from '../api/types'
import { createVisualTrakeSelection } from '../utils/submissionExport'
import { TrakeSelectionPanel } from './TrakeSelectionPanel'

function sequence(videoId: string, frames: number[]): TrakeVideoResultItem {
  return {
    video_id: videoId,
    score: 0.5,
    keyframes: frames.map((frame) => ({
      keyframe_id: `${videoId}_f${frame}`,
      frame_idx: frame,
      timestamp: frame / 25,
      thumbnail_url: `/api/keyframes/${videoId}_f${frame}/thumbnail`,
      image_url: `/api/keyframes/${videoId}_f${frame}/image`,
    })),
  }
}

describe('TrakeSelectionPanel', () => {
  it('adds a manual TRAKE sequence from video id and comma-separated keyframes', async () => {
    const user = userEvent.setup()
    const onAddManualSelection = vi.fn()
    render(
      <TrakeSelectionPanel
        results={[]}
        selection={[]}
        exportName="query-4-trake"
        eventCount={2}
        onExportNameChange={vi.fn()}
        onAddManualSelection={onAddManualSelection}
        onUpdateSelection={vi.fn()}
        onReorderSelection={vi.fn()}
        onRemoveSelection={vi.fn()}
        onClearSelection={vi.fn()}
        onExport={vi.fn()}
      />,
    )

    expect(screen.queryByLabelText(/margin/i)).not.toBeInTheDocument()
    await user.type(screen.getByLabelText('Video ID'), 'L30_V046')
    await user.type(screen.getByLabelText('Keyframes'), '894, 969')
    await user.click(screen.getByRole('button', { name: 'Add Sequence' }))

    expect(onAddManualSelection).toHaveBeenCalledWith(
      expect.objectContaining({
        source: 'manual',
        result: expect.objectContaining({
          video_id: 'L30_V046',
          keyframes: [
            expect.objectContaining({ frame_idx: 894 }),
            expect.objectContaining({ frame_idx: 969 }),
          ],
        }),
      }),
    )
  })

  it('requires the manual keyframe count to match the TRAKE event count', async () => {
    const user = userEvent.setup()
    const onAddManualSelection = vi.fn()
    render(
      <TrakeSelectionPanel
        results={[]}
        selection={[]}
        exportName="query-4-trake"
        eventCount={3}
        onExportNameChange={vi.fn()}
        onAddManualSelection={onAddManualSelection}
        onUpdateSelection={vi.fn()}
        onReorderSelection={vi.fn()}
        onRemoveSelection={vi.fn()}
        onClearSelection={vi.fn()}
        onExport={vi.fn()}
      />,
    )

    await user.type(screen.getByLabelText('Video ID'), 'L30_V046')
    await user.type(screen.getByLabelText('Keyframes'), '894, 969')
    await user.click(screen.getByRole('button', { name: 'Add Sequence' }))

    expect(screen.getByText('Enter exactly 3 keyframes for this TRAKE query.')).toBeInTheDocument()
    expect(onAddManualSelection).not.toHaveBeenCalled()
  })

  it('shows margin sweep controls for the primary selection and updates step/margin', () => {
    const result = sequence('L30_V031', [2099, 2190])
    const selection = [createVisualTrakeSelection(result, 0)]
    const onUpdateSelection = vi.fn()
    render(
      <TrakeSelectionPanel
        results={[result]}
        selection={selection}
        exportName="query-4-trake"
        eventCount={2}
        onExportNameChange={vi.fn()}
        onAddManualSelection={vi.fn()}
        onUpdateSelection={onUpdateSelection}
        onReorderSelection={vi.fn()}
        onRemoveSelection={vi.fn()}
        onClearSelection={vi.fn()}
        onExport={vi.fn()}
      />,
    )

    expect(screen.getByLabelText('Margin step (frames)')).toHaveValue(5)
    expect(screen.getByLabelText('Max margin (frames)')).toHaveValue(null)

    fireEvent.change(screen.getByLabelText('Margin step (frames)'), { target: { value: '10' } })
    expect(onUpdateSelection).toHaveBeenLastCalledWith(selection[0].key, { step: 10 })

    fireEvent.change(screen.getByLabelText('Max margin (frames)'), { target: { value: '40' } })
    expect(onUpdateSelection).toHaveBeenLastCalledWith(selection[0].key, { margin: 40 })
  })

  it('hides margin controls for selections after the first -- only sequence #1 is submitted', () => {
    const first = sequence('L30_V031', [2099, 2190])
    const second = sequence('L30_V032', [10, 20])
    const selection = [
      createVisualTrakeSelection(first, 0),
      createVisualTrakeSelection(second, 1),
    ]
    render(
      <TrakeSelectionPanel
        results={[first, second]}
        selection={selection}
        exportName="query-4-trake"
        eventCount={2}
        onExportNameChange={vi.fn()}
        onAddManualSelection={vi.fn()}
        onUpdateSelection={vi.fn()}
        onReorderSelection={vi.fn()}
        onRemoveSelection={vi.fn()}
        onClearSelection={vi.fn()}
        onExport={vi.fn()}
      />,
    )

    expect(screen.getAllByLabelText('Margin step (frames)')).toHaveLength(1)
    expect(
      screen.getByText('Not used for CSV export - only sequence #1 is submitted.'),
    ).toBeInTheDocument()
  })
})
