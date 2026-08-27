import { useEffect, useMemo, useRef, useState } from 'react'
import { fetchImageBlob } from './api/client'
import { DEFAULT_RRF_K } from './api/constants'
import type {
  AppMode,
  HealthStatus,
  SearchHit,
  TrakeKeyframe,
  TrakeVideoResultItem,
} from './api/types'
import { EventSequenceInput } from './components/EventSequenceInput'
import { FilterPanel } from './components/FilterPanel'
import { FrameChecker } from './components/FrameChecker'
import { GroupedResultList } from './components/GroupedResultList'
import { ImageCandidatePicker } from './components/ImageCandidatePicker'
import { ImageUpload } from './components/ImageUpload'
import { KeyframeDetail, type DetailFrame } from './components/KeyframeDetail'
import { KisSidebar, type KisSidebarTab } from './components/KisSidebar'
import type { KisSelectionMode } from './components/KisSelectionPanel'
import { ModeSelector } from './components/ModeSelector'
import { QueryRewritePanel } from './components/QueryRewritePanel'
import { ResultBrowserControls } from './components/ResultBrowserControls'
import { ResultGrid } from './components/ResultGrid'
import { SearchBar } from './components/SearchBar'
import { StatusBar } from './components/StatusBar'
import { TopKControl } from './components/TopKControl'
import { TrakeResultList } from './components/TrakeResultList'
import { TrakeSelectionPanel } from './components/TrakeSelectionPanel'
import { WebImageSearch } from './components/WebImageSearch'
import { useHealth } from './hooks/useHealth'
import { useImageCandidates } from './hooks/useImageCandidates'
import { useKeyboardShortcuts } from './hooks/useKeyboardShortcuts'
import { useQueryRewrite } from './hooks/useQueryRewrite'
import { useSearch } from './hooks/useSearch'
import { countFilledTrakeEvents, useTrake, type TrakeEventDraft } from './hooks/useTrake'
import {
  filterHits,
  hasLocalRanking,
  sortHitsByImageRankMode,
  type ImageRankSortMode,
  type RetrievalViewMode,
  type VideoGroupSortMode,
} from './utils/retrievalView'
import {
  buildKisSubmissionRows,
  buildTrakeSubmissionPlan,
  buildVqaFinalRows,
  createVisualKisSelection,
  downloadCsv,
  searchHitKey,
  serializeKisCsv,
  serializeTrakeCsv,
  serializeVqaCsv,
  trakeSequenceKey,
  validateCsvFilename,
  validateTrakeEventCount,
  type SelectedKisItem,
  type SelectedTrakeSequence,
  type SelectedVqaItem,
} from './utils/submissionExport'
import './App.css'

// Shown in StatusBar and used to disable OCR row selection while
// Elasticsearch is known to be down.
const ES_DOWN_MESSAGE = 'Elasticsearch không khả dụng — OCR trong KIS tạm thời không hoạt động'

// Index maps 1/2 shortcut keys onto modes -- keeps useKeyboardShortcuts
// mode-agnostic.
const MODE_ORDER: AppMode[] = ['kis', 'dante']

function asrReadinessMessage(health: HealthStatus | null): string | null {
  if (!health || health.asr_status === 'ready') return null
  if (health.asr_status === 'unreachable') {
    return 'Elasticsearch is unreachable; ASR is unavailable.'
  }
  if (health.asr_status === 'unavailable') {
    return 'The ASR transcript index is unavailable.'
  }
  return 'The ASR transcript index is empty.'
}

function App() {
  const health = useHealth()
  const {
    textQueries,
    setTextQuery,
    addTextQuery,
    removeTextQuery,
    localPhrasesByQuery,
    setLocalPhrase,
    addLocalPhrase,
    removeLocalPhrase,
    globalTextQuery,
    setGlobalTextQuery,
    textCandidateK,
    setTextCandidateK,
    ocrQuery,
    setOcrQuery,
    ocrCandidateK,
    setOcrCandidateK,
    titleQuery,
    setTitleQuery,
    titleCandidateK,
    setTitleCandidateK,
    asrQuery,
    setAsrQuery,
    asrVideoCandidateK,
    setAsrVideoCandidateK,
    candidatePool,
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
    runImageSearch,
  } = useSearch({ asrReady: health?.asr_ready !== false })
  const [selectedHit, setSelectedHit] = useState<SearchHit | null>(null)
  const [detailHit, setDetailHit] = useState<SearchHit | null>(null)
  const [detailHighlightFrames, setDetailHighlightFrames] = useState<DetailFrame[]>([])
  const queryInputRef = useRef<HTMLInputElement>(null)
  const kisLeftScrollRef = useRef<HTMLDivElement>(null)
  const esDown = health?.elasticsearch_ok === false
  const [viewMode, setViewMode] = useState<RetrievalViewMode>('ranked')
  const [imageSortMode, setImageSortMode] = useState<ImageRankSortMode>('local')
  const [groupSortMode, setGroupSortMode] = useState<VideoGroupSortMode>('local')
  const [videoFilter, setVideoFilter] = useState('')
  const [excludedVideoIds, setExcludedVideoIds] = useState<Set<string>>(() => new Set())
  const [kisFrameSweepStep, setKisFrameSweepStep] = useState(10)
  const [kisFrameSweepMargin, setKisFrameSweepMargin] = useState(0)
  const [trakeFrameSweepStep, setTrakeFrameSweepStep] = useState(2)
  const [trakeFrameSweepMargin, setTrakeFrameSweepMargin] = useState(0)
  const hasLocalResultRanking = useMemo(() => hasLocalRanking(results), [results])
  const activeImageSortMode: ImageRankSortMode =
    hasLocalResultRanking && imageSortMode === 'local' ? 'local' : 'global'
  const activeGroupSortMode: VideoGroupSortMode =
    hasLocalResultRanking || groupSortMode !== 'local' ? groupSortMode : 'global'
  const visibleResults = useMemo(() => {
    const filtered = filterHits(results, videoFilter)
    const visible =
      excludedVideoIds.size === 0
        ? filtered
        : filtered.filter((hit) => !excludedVideoIds.has(hit.video_id))
    return sortHitsByImageRankMode(visible, activeImageSortMode)
  }, [activeImageSortMode, excludedVideoIds, results, videoFilter])
  const asrMessage = asrReadinessMessage(health)

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
  const [kisSidebarTab, setKisSidebarTab] = useState<KisSidebarTab>('score')
  const [kisSelectionMode, setKisSelectionMode] = useState<KisSelectionMode>('kis')
  const [kisExportName, setKisExportName] = useState('query-1-kis')
  const [vqaExportName, setVqaExportName] = useState('query-3-qa')
  const [trakeExportName, setTrakeExportName] = useState('query-4-trake')
  const [kisSelection, setKisSelection] = useState<SelectedKisItem[]>([])
  const [vqaSelection, setVqaSelection] = useState<SelectedVqaItem[]>([])
  const [trakeSelection, setTrakeSelection] = useState<SelectedTrakeSequence[]>([])
  function handleModeChange(next: AppMode): void {
    setActiveMode(next)
  }

  useEffect(() => {
    setSelectedHit(visibleResults[0] ?? null)
  }, [visibleResults])

  useEffect(() => {
    setExcludedVideoIds(new Set())
    setKisSelection((prev) => syncSearchSelectionWithResults(prev, results))
    setVqaSelection((prev) => syncVqaSelectionWithResults(prev, results))
  }, [results])

  useEffect(() => {
    setTrakeSelection((prev) => syncTrakeSelectionWithResults(prev, trake.results))
  }, [trake.results])

  const manualSearchSelectionKeys = useMemo(() => {
    const source = kisSelectionMode === 'kis' ? kisSelection : vqaSelection
    return new Set(source.map((item) => searchHitKey(item.hit)))
  }, [kisSelection, kisSelectionMode, vqaSelection])

  const selectedTrakeSequenceKeys = useMemo(
    () => new Set(trakeSelection.map((item) => item.key)),
    [trakeSelection],
  )

  function toggleSearchManualSelection(hit: SearchHit): void {
    const key = searchHitKey(hit)
    const originalRank = Math.max(0, results.findIndex((result) => searchHitKey(result) === key))
    if (kisSelectionMode === 'kis') {
      setKisSelection((prev) =>
        prev.some((item) => searchHitKey(item.hit) === key)
          ? prev.filter((item) => searchHitKey(item.hit) !== key)
          : [
              ...prev,
              createVisualKisSelection(hit, originalRank, kisFrameSweepStep, kisFrameSweepMargin),
            ],
      )
      return
    }
    setVqaSelection((prev) =>
      prev.some((item) => item.key === key)
        ? prev.filter((item) => item.key !== key)
        : [...prev, { key, hit, originalRank, answer: '' }],
    )
  }

  function toggleTrakeSequence(result: TrakeVideoResultItem, originalRank: number): void {
    const key = trakeSequenceKey(result)
    setTrakeSelection((prev) =>
      prev.some((item) => item.key === key)
        ? prev.filter((item) => item.key !== key)
        : [...prev, { key, result, originalRank }],
    )
  }

  function exportKisCsv(): void {
    const validation = validateCsvFilename(kisExportName, 'kis')
    if (!validation.isValid || kisSelection.length === 0) return
    const rows = buildKisSubmissionRows(kisSelection, visibleResults)
    if (rows.length === 0) return
    downloadCsv(validation.filename, serializeKisCsv(rows))
  }

  function excludeVideo(videoId: string): void {
    setExcludedVideoIds((prev) => {
      const next = new Set(prev)
      next.add(videoId)
      return next
    })
  }

  function exportVqaCsv(): void {
    const validation = validateCsvFilename(vqaExportName, 'qa')
    const hasInvalidAnswer = vqaSelection.some(
      (item) => item.answer.length === 0 || item.answer.length > 100,
    )
    if (!validation.isValid || hasInvalidAnswer) return
    const rows = buildVqaFinalRows(vqaSelection)
    if (rows.length === 0) return
    downloadCsv(validation.filename, serializeVqaCsv(rows))
  }

  function exportTrakeCsv(): void {
    const validation = validateCsvFilename(trakeExportName, 'trake')
    if (!validation.isValid) return
    const rows = buildTrakeSubmissionPlan(trakeSelection, trake.results, {
      step: trakeFrameSweepStep,
      margin: trakeFrameSweepMargin,
    }).rows
    const eventCount = currentTrakeEventCount(trake.events, trake.results)
    if (rows.length === 0 || validateTrakeEventCount(rows, eventCount)) return
    downloadCsv(validation.filename, serializeTrakeCsv(rows))
  }

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
    setDetailHighlightFrames([])
    setDetailHit(hit)
  }

  function trakeFrameToDetailFrame(
    result: TrakeVideoResultItem,
    keyframe: TrakeKeyframe,
  ): DetailFrame {
    return {
      keyframe_id: keyframe.keyframe_id,
      video_id: result.video_id,
      frame_idx: keyframe.frame_idx,
      timestamp: keyframe.timestamp,
      thumbnail_url: keyframe.thumbnail_url,
      image_url: keyframe.image_url,
    }
  }

  function trakeFrameToSearchHit(
    result: TrakeVideoResultItem,
    keyframe: TrakeKeyframe,
  ): SearchHit {
    return {
      ...trakeFrameToDetailFrame(result, keyframe),
      score: result.score,
      thumbnail_path: '',
    }
  }

  function openTrakeDetail(result: TrakeVideoResultItem, keyframeIndex: number): void {
    const keyframe = result.keyframes[keyframeIndex]
    if (!keyframe) return
    setDetailHighlightFrames(
      result.keyframes.map((eventKeyframe) => trakeFrameToDetailFrame(result, eventKeyframe)),
    )
    setDetailHit(trakeFrameToSearchHit(result, keyframe))
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
        setDetailHighlightFrames([])
      } else {
        setSelectedHit(null)
      }
      queryRewrite.dismiss()
      imageCandidates.dismiss()
    },
  })

  return (
    <div className={`app ${activeMode === 'kis' ? 'app-kis' : 'app-trake'}`}>
      <div className="app-left" ref={activeMode === 'kis' ? kisLeftScrollRef : undefined}>
        <header className="app-header">
          {activeMode === 'dante' ? (
            <EventSequenceInput
              events={trake.events}
              onEventFieldChange={trake.setEventField}
              onLocalPhraseChange={trake.setLocalPhrase}
              onAddLocalPhrase={trake.addLocalPhrase}
              onRemoveLocalPhrase={trake.removeLocalPhrase}
              onAddEvent={trake.addEvent}
              onRemoveEvent={trake.removeEvent}
              threshold={trake.threshold}
              onThresholdChange={trake.setThreshold}
              onSubmit={trake.search}
              isLoading={trake.isLoading}
              asrDisabled={health?.asr_ready === false}
              asrReadinessMessage={asrMessage}
              eventDiagnostics={trake.eventDiagnostics}
              candidateVideoCount={trake.candidateVideoCount}
              droppedVideoCount={trake.droppedVideoCount}
              localRerankVideoCount={trake.localRerankVideoCount}
              localMissingFrameCount={trake.localMissingFrameCount}
            />
          ) : (
            <>
              <SearchBar
                ref={queryInputRef}
                textQueries={textQueries}
                onTextQueryChange={setTextQuery}
                onAddTextQuery={addTextQuery}
                onRemoveTextQuery={removeTextQuery}
                localPhrasesByQuery={localPhrasesByQuery}
                onLocalPhraseChange={setLocalPhrase}
                onAddLocalPhrase={addLocalPhrase}
                onRemoveLocalPhrase={removeLocalPhrase}
                globalTextQuery={globalTextQuery}
                textCandidateK={textCandidateK}
                ocrQuery={ocrQuery}
                ocrCandidateK={ocrCandidateK}
                titleQuery={titleQuery}
                titleCandidateK={titleCandidateK}
                asrQuery={asrQuery}
                asrVideoCandidateK={asrVideoCandidateK}
                asrGate={candidatePool?.asrGate ?? null}
                asrDisabled={health?.asr_ready === false}
                asrReadinessMessage={asrMessage}
                candidatePoolSize={candidatePool?.candidatePoolSize ?? null}
                textCandidateCount={candidatePool?.textCandidateCount ?? 0}
                ocrCandidateCount={candidatePool?.ocrCandidateCount ?? 0}
                titleCandidateCount={candidatePool?.titleCandidateCount ?? 0}
                onGlobalTextQueryChange={setGlobalTextQuery}
                onTextCandidateKChange={setTextCandidateK}
                onOcrQueryChange={setOcrQuery}
                onOcrCandidateKChange={setOcrCandidateK}
                onTitleQueryChange={setTitleQuery}
                onTitleCandidateKChange={setTitleCandidateK}
                onAsrQueryChange={setAsrQuery}
                onAsrVideoCandidateKChange={setAsrVideoCandidateK}
                onRetrieveCandidates={runSearch}
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
              <FrameChecker onOpenDetail={openDetail} />
            </>
          )}
          <StatusBar
            resultCount={activeMode === 'dante' ? trake.results.length : visibleResults.length}
            latencyMs={activeMode === 'dante' ? trake.latencyMs : latencyMs}
            isLoading={activeMode === 'dante' ? trake.isLoading : isLoading}
            error={activeMode === 'dante' ? trake.error : error}
            degradedMessage={esDown ? ES_DOWN_MESSAGE : null}
          />
        </header>
        <main className="app-grid">
          {activeMode === 'dante' ? (
            <TrakeResultList
              results={trake.results}
              sequencesByVideo={trake.sequencesByVideo}
              selectedSequenceKeys={selectedTrakeSequenceKeys}
              onToggleSequence={toggleTrakeSequence}
              onOpenDetail={openTrakeDetail}
            />
          ) : (
            <div className="result-area">
              <ResultBrowserControls
                viewMode={viewMode}
                onViewModeChange={setViewMode}
                imageSortMode={activeImageSortMode}
                onImageSortModeChange={setImageSortMode}
                groupSortMode={activeGroupSortMode}
                onGroupSortModeChange={setGroupSortMode}
                hasLocalRanking={hasLocalResultRanking}
                videoFilter={videoFilter}
                onVideoFilterChange={setVideoFilter}
                excludedCount={excludedVideoIds.size}
                onClearExcludedVideos={() => setExcludedVideoIds(new Set())}
                filteredCount={visibleResults.length}
                totalCount={results.length}
              />
              {viewMode === 'ranked' ? (
                <ResultGrid
                  hits={visibleResults}
                  rankSourceHits={results}
                  selectedKeyframeId={selectedHit?.keyframe_id}
                  manuallySelectedKeys={manualSearchSelectionKeys}
                  manualSelectionLabel={
                    kisSelectionMode === 'kis' ? 'Select KIS' : 'Select VQA'
                  }
                  scrollElementRef={kisLeftScrollRef}
                  onSelect={setSelectedHit}
                  onFindSimilar={findSimilar}
                  onToggleManualSelection={toggleSearchManualSelection}
                  onOpenDetail={openDetail}
                />
              ) : (
                <GroupedResultList
                  hits={visibleResults}
                  rankSourceHits={results}
                  sortMode={activeGroupSortMode}
                  rrfK={candidatePool?.rrfK ?? DEFAULT_RRF_K}
                  selectedKeyframeId={selectedHit?.keyframe_id}
                  manuallySelectedKeys={manualSearchSelectionKeys}
                  manualSelectionLabel={
                    kisSelectionMode === 'kis' ? 'Select KIS' : 'Select VQA'
                  }
                  onSelect={setSelectedHit}
                  onFindSimilar={findSimilar}
                  onToggleManualSelection={toggleSearchManualSelection}
                  onOpenDetail={openDetail}
                  onExcludeVideo={excludeVideo}
                />
              )}
            </div>
          )}
        </main>
      </div>
      {activeMode === 'kis' && (
        <KisSidebar
          activeTab={kisSidebarTab}
          onTabChange={setKisSidebarTab}
          hit={selectedHit}
          selectionMode={kisSelectionMode}
          onSelectionModeChange={setKisSelectionMode}
          results={visibleResults}
          kisSelection={kisSelection}
          vqaSelection={vqaSelection}
          kisExportName={kisExportName}
          vqaExportName={vqaExportName}
          onKisExportNameChange={setKisExportName}
          onVqaExportNameChange={setVqaExportName}
          onMoveKisSelection={(index, direction) =>
            setKisSelection((prev) => moveSelection(prev, index, direction))
          }
          onMoveVqaSelection={(index, direction) =>
            setVqaSelection((prev) => moveSelection(prev, index, direction))
          }
          onRemoveKisSelection={(key) =>
            setKisSelection((prev) => prev.filter((item) => item.key !== key))
          }
          onAddManualKisSelection={(item) =>
            setKisSelection((prev) =>
              prev.some((existing) => searchHitKey(existing.hit) === searchHitKey(item.hit))
                ? prev
                : [...prev, item],
            )
          }
          onRemoveVqaSelection={(key) =>
            setVqaSelection((prev) => prev.filter((item) => item.key !== key))
          }
          onClearKisSelection={() => setKisSelection([])}
          onClearVqaSelection={() => setVqaSelection([])}
          onClearVqaAnswers={() =>
            setVqaSelection((prev) => prev.map((item) => ({ ...item, answer: '' })))
          }
          onVqaAnswerChange={(key, answer) =>
            setVqaSelection((prev) =>
              prev.map((item) => (item.key === key ? { ...item, answer } : item)),
            )
          }
          kisFrameSweepStep={kisFrameSweepStep}
          kisFrameSweepMargin={kisFrameSweepMargin}
          onKisFrameSweepStepChange={setKisFrameSweepStep}
          onKisFrameSweepMarginChange={setKisFrameSweepMargin}
          onExportKis={exportKisCsv}
          onExportVqa={exportVqaCsv}
        />
      )}
      {activeMode === 'dante' && (
        <TrakeSelectionPanel
          results={trake.results}
          selection={trakeSelection}
          exportName={trakeExportName}
          eventCount={currentTrakeEventCount(trake.events, trake.results)}
          frameSweepStep={trakeFrameSweepStep}
          frameSweepMargin={trakeFrameSweepMargin}
          onFrameSweepStepChange={setTrakeFrameSweepStep}
          onFrameSweepMarginChange={setTrakeFrameSweepMargin}
          onExportNameChange={setTrakeExportName}
          onMoveSelection={(index, direction) =>
            setTrakeSelection((prev) => moveSelection(prev, index, direction))
          }
          onRemoveSelection={(key) =>
            setTrakeSelection((prev) => prev.filter((item) => item.key !== key))
          }
          onClearSelection={() => setTrakeSelection([])}
          onExport={exportTrakeCsv}
        />
      )}
      {detailHit && (
        <KeyframeDetail
          hit={detailHit}
          highlightedFrames={detailHighlightFrames}
          onClose={() => {
            setDetailHit(null)
            setDetailHighlightFrames([])
          }}
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

function moveSelection<T>(items: T[], index: number, direction: -1 | 1): T[] {
  const nextIndex = index + direction
  if (nextIndex < 0 || nextIndex >= items.length) return items
  const next = [...items]
  const [item] = next.splice(index, 1)
  next.splice(nextIndex, 0, item)
  return next
}

function syncSearchSelectionWithResults(
  selection: SelectedKisItem[],
  results: SearchHit[],
): SelectedKisItem[] {
  const resultByKey = new Map(results.map((hit, index) => [searchHitKey(hit), { hit, index }]))
  return selection.map((item) => {
    if (item.source === 'manual') return item
    const current = resultByKey.get(item.key)
    if (!current) return item
    return { ...item, hit: current.hit, originalRank: current.index }
  })
}

function syncVqaSelectionWithResults(
  selection: SelectedVqaItem[],
  results: SearchHit[],
): SelectedVqaItem[] {
  const resultByKey = new Map(results.map((hit, index) => [searchHitKey(hit), { hit, index }]))
  return selection.flatMap((item) => {
    const current = resultByKey.get(item.key)
    if (!current) return []
    return [{ ...item, hit: current.hit, originalRank: current.index }]
  })
}

function syncTrakeSelectionWithResults(
  selection: SelectedTrakeSequence[],
  results: TrakeVideoResultItem[],
): SelectedTrakeSequence[] {
  const resultByKey = new Map(
    results.map((result, index) => [trakeSequenceKey(result), { result, index }]),
  )
  return selection.flatMap((item) => {
    const current = resultByKey.get(item.key)
    if (!current) return []
    return [{ ...item, result: current.result, originalRank: current.index }]
  })
}

function currentTrakeEventCount(
  events: TrakeEventDraft[],
  results: TrakeVideoResultItem[],
): number {
  // An event row counts when ANY of its four boxes is filled -- an OCR-only or
  // ASR-only row is a legitimate event, and the submitted sequence will have
  // one frame for it like any other.
  return countFilledTrakeEvents(events) || results[0]?.keyframes.length || 0
}

export default App
