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

  it('shows synchronized target controls and updates only the edited event target', () => {
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

    expect(screen.getByText('Event 1 start')).toBeInTheDocument()
    expect(screen.getByText('Event 2 start')).toBeInTheDocument()
    expect(screen.getByLabelText('Number of Steps')).toHaveValue(5)

    const targetInputs = screen.getAllByLabelText('Target frame')
    fireEvent.change(targetInputs[1], { target: { value: '2280' } })

    expect(onUpdateSelection).toHaveBeenLastCalledWith(selection[0].key, {
      stepTargetFrames: [2099, 2280],
    })
  })
})
