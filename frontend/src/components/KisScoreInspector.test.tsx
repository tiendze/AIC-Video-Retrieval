import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import type { SearchHit } from '../api/types'
import { KisScoreInspector } from './KisScoreInspector'

describe('KisScoreInspector', () => {
  it('renders ABE score details and toggles the patch heatmap', () => {
    const hit: SearchHit = {
      keyframe_id: 'A',
      video_id: 'V1',
      score: 1,
      final_score: 1,
      active_modalities: 1,
      frame_idx: 10,
      timestamp: 0.4,
      thumbnail_path: 'a.jpg',
      thumbnail_url: '/api/keyframes/A/thumbnail',
      image_url: '/api/keyframes/A/image',
      modalities: {
        abe: {
          geometric_mean: 1,
          score: 1,
          queries: [
            {
              query_index: 0,
              pairs: [
                {
                  attribute: 'red',
                  object: 'packet',
                  positive_attributes: ['red'],
                  negative_attributes: ['white'],
                },
              ],
              base_local: 0.2,
              refined_local: 0.4,
              binding_delta: 0.2,
              local_score: 0.6,
              global_abe: 0.7,
              raw_score: 0.63,
              rank: 1,
              rank_percentile: 1,
              patch_heatmap: {
                patch_count: 4,
                grid_width: 2,
                grid_height: 2,
                base_top_patches: [{ label: 'red', patch_index: 0, score: 0.9 }],
                refined_top_patches: [{ label: 'red -> packet', patch_index: 1, score: 0.8 }],
              },
            },
          ],
        },
      },
    }

    render(<KisScoreInspector hit={hit} />)

    expect(screen.getByText('ABE - FG-CLIP')).toBeInTheDocument()
    expect(screen.getByText('Positive attrs')).toBeInTheDocument()
    expect(screen.getByText('Raw ABE Score')).toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: 'Show ABE Patch Heatmap' }))

    expect(screen.getByTitle(/Patch 0/)).toBeInTheDocument()
    expect(screen.getByTitle(/Patch 1/)).toBeInTheDocument()
  })
})
