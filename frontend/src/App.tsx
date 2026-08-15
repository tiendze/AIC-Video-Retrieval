import { useRef, useState } from 'react'
import { fetchImageBlob } from './api/client'
import type { SearchHit, SearchMode } from './api/types'
import { FilterPanel } from './components/FilterPanel'
import { ImageCandidatePicker } from './components/ImageCandidatePicker'
import { ImageUpload } from './components/ImageUpload'
import { KeyframeDetail } from './components/KeyframeDetail'
import { ModeSelector } from './components/ModeSelector'
import { QueryRewritePanel } from './components/QueryRewritePanel'
import { ResultGrid } from './components/ResultGrid'
import { SearchBar } from './components/SearchBar'
import { StatusBar } from './components/StatusBar'
import { TopKControl } from './components/TopKControl'
import { WebImageSearch } from './components/WebImageSearch'
import { useHealth } from './hooks/useHealth'
import { useImageCandidates } from './hooks/useImageCandidates'
import { useKeyboardShortcuts } from './hooks/useKeyboardShortcuts'
import { useQueryRewrite } from './hooks/useQueryRewrite'
import { useSearch } from './hooks/useSearch'
import './App.css'

// plan §P1.8, PRD T5: shown in StatusBar and used to disable OCR mode
// whenever GET /health reports elasticsearch_ok=false.
const ES_DOWN_MESSAGE = 'Elasticsearch không khả dụng — chỉ tìm kiếm Semantic'

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
    setResults,
    isLoading,
    error,
    latencyMs,
    setLatencyMs,
    runSearch,
    runImageSearch,
  } = useSearch()
  const [selectedHit, setSelectedHit] = useState<SearchHit | null>(null)
  const queryInputRef = useRef<HTMLInputElement>(null)
  const health = useHealth()
  const esDown = health?.elasticsearch_ok === false

  // QUEST (plan §P2.5, PRD §F5). Both branches push hits into the exact
  // same grid a regular search populates -- latencyMs resets to null
  // because P1.7's measurement is specific to POST /api/search, not these
  // endpoints (useSearch.ts's own comment on setLatencyMs).
  function applyQuestHits(hits: SearchHit[]): void {
    setResults(hits)
    setLatencyMs(null)
  }
  const queryRewrite = useQueryRewrite()
  const imageCandidates = useImageCandidates(topK, applyQuestHits)

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
    onClose: () => {
      setSelectedHit(null)
      queryRewrite.dismiss()
      imageCandidates.dismiss()
    },
  })

  return (
    <div className="app">
      <header className="app-header">
        <SearchBar
          ref={queryInputRef}
          query={query}
          onQueryChange={setQuery}
          onSubmit={runSearch}
          onEnhance={queryRewrite.rewrite}
          enhanceLoading={queryRewrite.isLoading}
        />
        {queryRewrite.result && (
          <QueryRewritePanel
            result={queryRewrite.result}
            onSearch={(text) => {
              setQuery(text)
              runSearch(text)
              queryRewrite.dismiss()
            }}
            onDismiss={queryRewrite.dismiss}
          />
        )}
        {queryRewrite.error && <span className="quest-error">{queryRewrite.error}</span>}
        <div className="app-toolbar">
          <ModeSelector mode={mode} onChange={setMode} ocrDisabled={esDown} />
          <TopKControl topK={topK} onChange={setTopK} />
          <FilterPanel />
        </div>
        <ImageUpload onUpload={runImageSearch} />
        <WebImageSearch
          query={query}
          onSearch={imageCandidates.findCandidates}
          disabled={imageCandidates.isLoading}
        />
        {imageCandidates.error && <span className="quest-error">{imageCandidates.error}</span>}
        <StatusBar
          resultCount={results.length}
          latencyMs={latencyMs}
          isLoading={isLoading}
          error={error}
          degradedMessage={esDown ? ES_DOWN_MESSAGE : null}
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
      {imageCandidates.result?.status === 'candidates' && (
        <ImageCandidatePicker
          candidates={imageCandidates.result.candidates}
          onPick={imageCandidates.pickCandidate}
          onDismiss={imageCandidates.dismiss}
        />
      )}
    </div>
  )
}

export default App
