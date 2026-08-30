import type { SearchHit } from '../api/types'
import type { SelectedKisItem, SelectedVqaItem } from '../utils/submissionExport'
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
  kisSelection: SelectedKisItem[]
  vqaSelection: SelectedVqaItem[]
  kisExportName: string
  vqaExportName: string
  onKisExportNameChange: (value: string) => void
  onVqaExportNameChange: (value: string) => void
  onReorderKisSelection: (fromIndex: number, toIndex: number) => void
  onReorderVqaSelection: (fromIndex: number, toIndex: number) => void
  onRemoveKisSelection: (key: string) => void
  onAddManualKisSelection: (item: SelectedKisItem) => void
  onUpdateKisSelection: (key: string, patch: Partial<SelectedKisItem>) => void
  onAddManualVqaSelection: (item: SelectedVqaItem) => void
  onUpdateVqaSelection: (key: string, patch: Partial<SelectedVqaItem>) => void
  onRemoveVqaSelection: (key: string) => void
  onClearKisSelection: () => void
  onClearVqaSelection: () => void
  onClearVqaAnswers: () => void
  onVqaAnswerChange: (key: string, answer: string) => void
  kisFrameSweepStep: number
  onKisFrameSweepStepChange: (value: number) => void
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
  onReorderKisSelection,
  onReorderVqaSelection,
  onRemoveKisSelection,
  onAddManualKisSelection,
  onUpdateKisSelection,
  onAddManualVqaSelection,
  onUpdateVqaSelection,
  onRemoveVqaSelection,
  onClearKisSelection,
  onClearVqaSelection,
  onClearVqaAnswers,
  onVqaAnswerChange,
  kisFrameSweepStep,
  onKisFrameSweepStepChange,
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
          onReorderKisSelection={onReorderKisSelection}
          onReorderVqaSelection={onReorderVqaSelection}
          onRemoveKisSelection={onRemoveKisSelection}
          onAddManualKisSelection={onAddManualKisSelection}
          onUpdateKisSelection={onUpdateKisSelection}
          onAddManualVqaSelection={onAddManualVqaSelection}
          onUpdateVqaSelection={onUpdateVqaSelection}
          onRemoveVqaSelection={onRemoveVqaSelection}
          onClearKisSelection={onClearKisSelection}
          onClearVqaSelection={onClearVqaSelection}
          onClearVqaAnswers={onClearVqaAnswers}
          onVqaAnswerChange={onVqaAnswerChange}
          kisFrameSweepStep={kisFrameSweepStep}
          onKisFrameSweepStepChange={onKisFrameSweepStepChange}
          onExportKis={onExportKis}
          onExportVqa={onExportVqa}
        />
      )}
    </aside>
  )
}
