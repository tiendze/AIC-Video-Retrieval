import { render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { ResultGrid } from './ResultGrid'
import type { SearchHit } from '../api/types'

// RED reproducer for plan §P1.6's virtualization requirement. `max_top_k`
// caps a *live* grid at 100 (references.md §1.2 submission limit), so this
// uses a synthetic 1,000-hit fixture to prove the grid itself scales --
// the plan's own resolution of the max_top_k=100 vs "≥1.000 thumbnail"
// contradiction (see plan §P1.6 finding 2).

function makeHits(count: number): SearchHit[] {
  return Array.from({ length: count }, (_, i) => ({
    keyframe_id: `L21_V001_f${i}`,
    video_id: 'L21_V001',
    score: 1 - i / count,
    frame_idx: i,
    timestamp: i * 0.1,
    thumbnail_path: `t${i}.jpg`,
    thumbnail_url: `/api/keyframes/L21_V001_f${i}/thumbnail`,
    image_url: `/api/keyframes/L21_V001_f${i}/image`,
  }))
}

// jsdom reports 0 for every layout measurement, so @tanstack/react-virtual
// needs an explicit viewport size to compute how many rows fit -- without
// this the virtualizer collapses to zero estimated rows instead of a
// bounded window. @tanstack/virtual-core measures via offsetWidth/
// offsetHeight (not clientWidth/clientHeight); ResultGrid's own column-count
// math uses clientWidth, so both are stubbed. Applied on the prototype (not
// the instance) so the virtualizer's first-mount measurement -- which runs
// in a layout effect before the test can reach in and set an instance
// property -- already sees a nonzero size.
const STUBBED_PROPS = ['clientHeight', 'clientWidth', 'offsetHeight', 'offsetWidth'] as const

function stubViewportSizeGlobally(size: number): () => void {
  const originals = STUBBED_PROPS.map(
    (prop) => [prop, Object.getOwnPropertyDescriptor(HTMLElement.prototype, prop)] as const,
  )
  for (const prop of STUBBED_PROPS) {
    Object.defineProperty(HTMLElement.prototype, prop, { configurable: true, value: size })
  }
  return () => {
    for (const [prop, descriptor] of originals) {
      if (descriptor) Object.defineProperty(HTMLElement.prototype, prop, descriptor)
    }
  }
}

describe('ResultGrid', () => {
  it('mounts a bounded number of DOM nodes for 1,000 hits (virtualized)', () => {
    const restore = stubViewportSizeGlobally(600)
    const hits = makeHits(1000)
    const { container } = render(
      <ResultGrid hits={hits} onSelect={vi.fn()} onFindSimilar={vi.fn()} />,
    )
    restore()

    const renderedCards = container.querySelectorAll('button[aria-label*="L21_V001"]')
    // Never mounts anywhere close to all 1,000 -- a handful of screens'
    // worth is the point of virtualization.
    expect(renderedCards.length).toBeGreaterThan(0)
    expect(renderedCards.length).toBeLessThan(100)
  })

  it('renders nothing but an empty state for zero hits', () => {
    render(<ResultGrid hits={[]} onSelect={vi.fn()} onFindSimilar={vi.fn()} />)

    expect(screen.getByText(/no results/i)).toBeInTheDocument()
  })

  it('shows local and global rank badges when local reranking is present', () => {
    const restore = stubViewportSizeGlobally(600)
    render(
      <ResultGrid
        hits={[
          {
            keyframe_id: 'L26_V035_f5056',
            video_id: 'L26_V035',
            score: 0.31,
            fused_image_rank: 687,
            local_score: 0.31,
            local_retrieval_used: true,
            frame_idx: 5056,
            timestamp: 202.24,
            thumbnail_path: '',
            thumbnail_url: '/api/keyframes/L26_V035_f5056/thumbnail',
            image_url: '/api/keyframes/L26_V035_f5056/image',
          },
        ]}
        onSelect={vi.fn()}
        onFindSimilar={vi.fn()}
      />,
    )
    restore()

    expect(screen.getByText('L#1 G#687')).toBeInTheDocument()
  })
})
