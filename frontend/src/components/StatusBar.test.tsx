import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { StatusBar } from './StatusBar'

const ES_DOWN_MESSAGE = 'Elasticsearch không khả dụng — OCR trong KIS tạm thời không hoạt động'

describe('StatusBar', () => {
  it('shows the degraded-mode banner when degradedMessage is set', () => {
    render(
      <StatusBar
        resultCount={0}
        latencyMs={null}
        isLoading={false}
        error={null}
        degradedMessage={ES_DOWN_MESSAGE}
      />,
    )

    expect(screen.getByText(ES_DOWN_MESSAGE)).toBeInTheDocument()
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
        degradedMessage={ES_DOWN_MESSAGE}
      />,
    )

    expect(screen.getByText(/3 results/)).toBeInTheDocument()
  })
})
