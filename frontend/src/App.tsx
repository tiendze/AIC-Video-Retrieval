import { useRef, useState } from 'react'
import { fetchImageBlob } from './api/client'
import type { SearchHit, SearchMode } from './api/types'
import { FilterPanel } from './components/FilterPanel'
import { ImageUpload } from './components/ImageUpload'
import { KeyframeDetail } from './components/KeyframeDetail'
import { ModeSelector } from './components/ModeSelector'
import { ResultGrid } from './components/ResultGrid'
import { SearchBar } from './components/SearchBar'
import { StatusBar } from './components/StatusBar'
import { TopKControl } from './components/TopKControl'
import { useKeyboardShortcuts } from './hooks/useKeyboardShortcuts'
import { useSearch } from './hooks/useSearch'
import './App.css'

// Index maps 1/2/3 shortcut keys onto modes -- keeps useKeyboardShortcuts
// mode-agnostic (plan §P1.6: "1/2/3 đổi mode").
const MODE_ORDER: SearchMode[] = ['semantic', 'ocr']

function App() {
  const {
    query,
    setQuery,
    mode,
    setMode,
    topK,
    setTopK,
    results,
    isLoading,
    error,
    latencyMs,
    runSearch,
    runImageSearch,
  } = useSearch()
  const [selectedHit, setSelectedHit] = useState<SearchHit | null>(null)
  const queryInputRef = useRef<HTMLInputElement>(null)

  // Shared by ImageUpload's manual upload and every "Find Similar Images"
  // trigger (result card + detail panel) -- fetches the target's own
  // full-res bytes and re-submits through the same POST /api/search/image
  // path (plan §P1.6, CLAUDE.md §3: one encoder, one preprocessing).
  function findSimilar(hit: SearchHit): void {
    fetchImageBlob(hit.image_url).then((blob) => runImageSearch(blob))
  }

  useKeyboardShortcuts({
    onSubmit: runSearch,
    onFocusQuery: () => queryInputRef.current?.focus(),
    onSetMode: (index) => {
      const next = MODE_ORDER[index]
      if (next) setMode(next)
    },
    onClose: () => setSelectedHit(null),
  })

  return (
    <div className="app">
      <header className="app-header">
        <SearchBar ref={queryInputRef} query={query} onQueryChange={setQuery} onSubmit={runSearch} />
        <div className="app-toolbar">
          <ModeSelector mode={mode} onChange={setMode} />
          <TopKControl topK={topK} onChange={setTopK} />
          <FilterPanel />
        </div>
        <ImageUpload onUpload={runImageSearch} />
        <StatusBar
          resultCount={results.length}
          latencyMs={latencyMs}
          isLoading={isLoading}
          error={error}
        />
      </header>
      <main className="app-grid">
        <ResultGrid hits={results} onSelect={setSelectedHit} onFindSimilar={findSimilar} />
      </main>
      {selectedHit && (
        <KeyframeDetail
          hit={selectedHit}
          onClose={() => setSelectedHit(null)}
          onFindSimilar={findSimilar}
        />
      )}
    </div>
  )
}

export default App
