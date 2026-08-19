import { useEffect, useRef, useState } from 'react'
import { fetchImageBlob } from './api/client'
import type { AppMode, SearchHit } from './api/types'
import { EventSequenceInput } from './components/EventSequenceInput'
import { FilterPanel } from './components/FilterPanel'
import { ImageCandidatePicker } from './components/ImageCandidatePicker'
import { ImageUpload } from './components/ImageUpload'
import { KeyframeDetail } from './components/KeyframeDetail'
import { KisScoreInspector } from './components/KisScoreInspector'
import { ModeSelector } from './components/ModeSelector'
import { QueryRewritePanel } from './components/QueryRewritePanel'
import { ResultGrid } from './components/ResultGrid'
import { SearchBar } from './components/SearchBar'
import { StatusBar } from './components/StatusBar'
import { TopKControl } from './components/TopKControl'
import { TrakeResultList } from './components/TrakeResultList'
import { WebImageSearch } from './components/WebImageSearch'
import { useHealth } from './hooks/useHealth'
import { useImageCandidates } from './hooks/useImageCandidates'
import { useKeyboardShortcuts } from './hooks/useKeyboardShortcuts'
import { useQueryRewrite } from './hooks/useQueryRewrite'
import { useSearch } from './hooks/useSearch'
import { useTrake } from './hooks/useTrake'
import './App.css'

// Shown in StatusBar and used to disable OCR row selection while
// Elasticsearch is known to be down.
const ES_DOWN_MESSAGE = 'Elasticsearch không khả dụng — OCR trong KIS tạm thời không hoạt động'

// Index maps 1/2 shortcut keys onto modes -- keeps useKeyboardShortcuts
// mode-agnostic.
const MODE_ORDER: AppMode[] = ['kis', 'dante']

function App() {
  const {
    globalTextQuery,
    setGlobalTextQuery,
    textCandidateK,
    setTextCandidateK,
    ocrQuery,
    setOcrQuery,
    ocrCandidateK,
    setOcrCandidateK,
    candidatePool,
    stage2Queries,
    setStage2QueryText,
    setStage2QueryType,
    setAbePairAttribute,
    setAbePairObject,
    addAbePair,
    removeAbePair,
    addStage2Query,
    removeStage2Query,
    finalTopK,
    setFinalTopK,
    query,
    setQuery,
    topK,
    setTopK,
    results,
    setResults,
    isLoading,
    error,
    latencyMs,
    setLatencyMs,
    runSearch,
    rerankCandidates,
    runImageSearch,
  } = useSearch()
  const [selectedHit, setSelectedHit] = useState<SearchHit | null>(null)
  const [detailHit, setDetailHit] = useState<SearchHit | null>(null)
  const queryInputRef = useRef<HTMLInputElement>(null)
  const kisLeftScrollRef = useRef<HTMLDivElement>(null)
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

  // DANTE/TRAKE (plan §P2.2, PRD §F6/US-4) is a different request shape
  // (N ordered events, not KIS query rows) and a different endpoint.
  const [activeMode, setActiveMode] = useState<AppMode>('kis')
  const trake = useTrake(topK)
  function handleModeChange(next: AppMode): void {
    setActiveMode(next)
  }

  useEffect(() => {
    setSelectedHit(results[0] ?? null)
  }, [results])

  // Shared by ImageUpload's manual upload and every "Find Similar Images"
  // trigger (result card + detail panel) -- fetches the target's own
  // full-res bytes and re-submits through the same POST /api/search/image
  // path (plan §P1.6, CLAUDE.md §3: one encoder, one preprocessing).
  function findSimilar(hit: SearchHit): void {
    findSimilarByImageUrl(hit.image_url)
  }

  function findSimilarByImageUrl(imageUrl: string): void {
    fetchImageBlob(imageUrl).then((blob) => runImageSearch(blob))
  }

  function openDetail(hit: SearchHit): void {
    setSelectedHit(hit)
    setDetailHit(hit)
  }

  useKeyboardShortcuts({
    onSubmit: runSearch,
    onFocusQuery: () => queryInputRef.current?.focus(),
    onSetMode: (index) => {
      const next = MODE_ORDER[index]
      if (next) handleModeChange(next)
    },
    onClose: () => {
      if (detailHit) {
        setDetailHit(null)
      } else {
        setSelectedHit(null)
      }
      queryRewrite.dismiss()
      imageCandidates.dismiss()
    },
  })

  return (
    <div className={`app ${activeMode === 'kis' ? 'app-kis' : 'app-stack'}`}>
      <div className="app-left" ref={activeMode === 'kis' ? kisLeftScrollRef : undefined}>
        <header className="app-header">
          {activeMode === 'dante' ? (
            <EventSequenceInput
              events={trake.events}
              onEventChange={trake.setEvent}
              onAddEvent={trake.addEvent}
              onRemoveEvent={trake.removeEvent}
              threshold={trake.threshold}
              onThresholdChange={trake.setThreshold}
              onSubmit={trake.search}
              isLoading={trake.isLoading}
            />
          ) : (
            <>
              <SearchBar
                ref={queryInputRef}
                globalTextQuery={globalTextQuery}
                textCandidateK={textCandidateK}
                ocrQuery={ocrQuery}
                ocrCandidateK={ocrCandidateK}
                candidatePoolSize={candidatePool?.candidatePoolSize ?? null}
                textCandidateCount={candidatePool?.textCandidateCount ?? 0}
                ocrCandidateCount={candidatePool?.ocrCandidateCount ?? 0}
                stage2Queries={stage2Queries}
                finalTopK={finalTopK}
                onGlobalTextQueryChange={setGlobalTextQuery}
                onTextCandidateKChange={setTextCandidateK}
                onOcrQueryChange={setOcrQuery}
                onOcrCandidateKChange={setOcrCandidateK}
                onStage2QueryTextChange={setStage2QueryText}
                onStage2QueryTypeChange={setStage2QueryType}
                onAbePairAttributeChange={setAbePairAttribute}
                onAbePairObjectChange={setAbePairObject}
                onAddAbePair={addAbePair}
                onRemoveAbePair={removeAbePair}
                onAddStage2Query={addStage2Query}
                onRemoveStage2Query={removeStage2Query}
                onFinalTopKChange={setFinalTopK}
                onRetrieveCandidates={runSearch}
                onRerankCandidates={rerankCandidates}
                onEnhance={queryRewrite.rewrite}
                enhanceLoading={queryRewrite.isLoading}
                ocrDisabled={esDown}
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
            </>
          )}
          <div className="app-toolbar">
            <ModeSelector mode={activeMode} onChange={handleModeChange} />
            {activeMode === 'dante' && <TopKControl topK={topK} onChange={setTopK} />}
            <FilterPanel />
          </div>
          {activeMode !== 'dante' && (
            <>
              <ImageUpload onUpload={runImageSearch} />
              <WebImageSearch
                query={query}
                onSearch={imageCandidates.findCandidates}
                disabled={imageCandidates.isLoading}
              />
              {imageCandidates.error && (
                <span className="quest-error">{imageCandidates.error}</span>
              )}
            </>
          )}
          <StatusBar
            resultCount={activeMode === 'dante' ? trake.results.length : results.length}
            latencyMs={activeMode === 'dante' ? null : latencyMs}
            isLoading={activeMode === 'dante' ? trake.isLoading : isLoading}
            error={activeMode === 'dante' ? trake.error : error}
            degradedMessage={esDown ? ES_DOWN_MESSAGE : null}
          />
        </header>
        <main className="app-grid">
          {activeMode === 'dante' ? (
            <TrakeResultList results={trake.results} />
          ) : (
            <div className="result-area">
              <ResultGrid
                hits={results}
                selectedKeyframeId={selectedHit?.keyframe_id}
                scrollElementRef={kisLeftScrollRef}
                onSelect={setSelectedHit}
                onFindSimilar={findSimilar}
                onOpenDetail={openDetail}
              />
            </div>
          )}
        </main>
      </div>
      {activeMode === 'kis' && <KisScoreInspector hit={selectedHit} />}
      {detailHit && (
        <KeyframeDetail
          hit={detailHit}
          onClose={() => setDetailHit(null)}
          onFindSimilar={findSimilarByImageUrl}
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
