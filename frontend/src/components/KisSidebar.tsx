import type { SearchHit } from '../api/types'
import type { SelectedSearchItem, SelectedVqaItem } from '../utils/submissionExport'
import { KisScoreInspector } from './KisScoreInspector'
import scoreStyles from './KisScoreInspector.module.css'
import { KisSelectionPanel, type KisSelectionMode } from './KisSelectionPanel'

export type KisSidebarTab = 'score' | 'selection'

export interface KisSidebarProps {
  activeTab: KisSidebarTab
  onTabChange: (tab: KisSidebarTab) => void
  hit: SearchHit | null
  selectionMode: KisSelectionMode
  onSelectionModeChange: (mode: KisSelectionMode) => void
  results: SearchHit[]
  kisSelection: SelectedSearchItem[]
  vqaSelection: SelectedVqaItem[]
  kisExportName: string
  vqaExportName: string
  onKisExportNameChange: (value: string) => void
  onVqaExportNameChange: (value: string) => void
  onMoveKisSelection: (index: number, direction: -1 | 1) => void
  onMoveVqaSelection: (index: number, direction: -1 | 1) => void
  onRemoveKisSelection: (key: string) => void
  onRemoveVqaSelection: (key: string) => void
  onClearKisSelection: () => void
  onClearVqaSelection: () => void
  onClearVqaAnswers: () => void
  onVqaAnswerChange: (key: string, answer: string) => void
  kisFrameSweepStep: number
  kisFrameSweepMargin: number
  onKisFrameSweepStepChange: (value: number) => void
  onKisFrameSweepMarginChange: (value: number) => void
  onExportKis: () => void
  onExportVqa: () => void
}

export function KisSidebar({
  activeTab,
  onTabChange,
  hit,
  selectionMode,
  onSelectionModeChange,
  results,
  kisSelection,
  vqaSelection,
  kisExportName,
  vqaExportName,
  onKisExportNameChange,
  onVqaExportNameChange,
  onMoveKisSelection,
  onMoveVqaSelection,
  onRemoveKisSelection,
  onRemoveVqaSelection,
  onClearKisSelection,
  onClearVqaSelection,
  onClearVqaAnswers,
  onVqaAnswerChange,
  kisFrameSweepStep,
  kisFrameSweepMargin,
  onKisFrameSweepStepChange,
  onKisFrameSweepMarginChange,
  onExportKis,
  onExportVqa,
}: KisSidebarProps) {
  return (
    <aside
      className={scoreStyles.panel}
      aria-label={activeTab === 'score' ? 'KIS score details' : 'KIS selection export'}
    >
      <div className={scoreStyles.tabs} role="tablist" aria-label="KIS sidebar">
        <button
          type="button"
          role="tab"
          aria-selected={activeTab === 'score'}
          className={`${scoreStyles.tab} ${activeTab === 'score' ? scoreStyles.tabActive : ''}`}
          onClick={() => onTabChange('score')}
        >
          Score
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={activeTab === 'selection'}
          className={`${scoreStyles.tab} ${
            activeTab === 'selection' ? scoreStyles.tabActive : ''
          }`}
          onClick={() => onTabChange('selection')}
        >
          Selection
        </button>
      </div>
      {activeTab === 'score' ? (
        <KisScoreInspector hit={hit} embedded />
      ) : (
        <KisSelectionPanel
          mode={selectionMode}
          onModeChange={onSelectionModeChange}
          results={results}
          kisSelection={kisSelection}
          vqaSelection={vqaSelection}
          kisExportName={kisExportName}
          vqaExportName={vqaExportName}
          onKisExportNameChange={onKisExportNameChange}
          onVqaExportNameChange={onVqaExportNameChange}
          onMoveKisSelection={onMoveKisSelection}
          onMoveVqaSelection={onMoveVqaSelection}
          onRemoveKisSelection={onRemoveKisSelection}
          onRemoveVqaSelection={onRemoveVqaSelection}
          onClearKisSelection={onClearKisSelection}
          onClearVqaSelection={onClearVqaSelection}
          onClearVqaAnswers={onClearVqaAnswers}
          onVqaAnswerChange={onVqaAnswerChange}
          kisFrameSweepStep={kisFrameSweepStep}
          kisFrameSweepMargin={kisFrameSweepMargin}
          onKisFrameSweepStepChange={onKisFrameSweepStepChange}
          onKisFrameSweepMarginChange={onKisFrameSweepMarginChange}
          onExportKis={onExportKis}
          onExportVqa={onExportVqa}
        />
      )}
    </aside>
  )
}
