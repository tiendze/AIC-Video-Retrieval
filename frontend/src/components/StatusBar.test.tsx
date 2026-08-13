import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { StatusBar } from './StatusBar'

// RED reproducer for plan §P1.8 (PRD T5): StatusBar's own docstring named
// this exact gap -- "a degraded-mode banner (ES down -> semantic only) is
// P1.8 scope, not this one".
describe('StatusBar', () => {
  it('shows the degraded-mode banner when degradedMessage is set', () => {
    render(
      <StatusBar
        resultCount={0}
        latencyMs={null}
        isLoading={false}
        error={null}
        degradedMessage="Elasticsearch không khả dụng — chỉ tìm kiếm Semantic"
      />,
    )

    expect(
      screen.getByText('Elasticsearch không khả dụng — chỉ tìm kiếm Semantic'),
    ).toBeInTheDocument()
  })

  it('shows no banner when degradedMessage is null', () => {
    render(<StatusBar resultCount={0} latencyMs={null} isLoading={false} error={null} degradedMessage={null} />)

    expect(screen.queryByText(/không khả dụng/)).not.toBeInTheDocument()
  })

  it('still reports the result count when degraded', () => {
    render(
      <StatusBar
        resultCount={3}
        latencyMs={120}
        isLoading={false}
        error={null}
        degradedMessage="Elasticsearch không khả dụng — chỉ tìm kiếm Semantic"
      />,
    )

    expect(screen.getByText(/3 results/)).toBeInTheDocument()
  })
})
