import { forwardRef } from 'react'
import { MAX_CANDIDATE_K, MAX_TOP_K, MIN_TOP_K } from '../api/constants'
import type { KisQueryRow, KisQueryType } from '../api/types'
import styles from './SearchBar.module.css'

export interface SearchBarProps {
  globalTextQuery: string
  textCandidateK: number
  ocrQuery: string
  ocrCandidateK: number
  candidatePoolSize: number | null
  textCandidateCount: number
  ocrCandidateCount: number
  stage2Queries: KisQueryRow[]
  finalTopK: number
  onGlobalTextQueryChange: (value: string) => void
  onTextCandidateKChange: (value: number) => void
  onOcrQueryChange: (value: string) => void
  onOcrCandidateKChange: (value: number) => void
  onStage2QueryTextChange: (id: string, value: string) => void
  onStage2QueryTypeChange: (id: string, type: KisQueryType) => void
  onAddStage2Query: () => void
  onRemoveStage2Query: (id: string) => void
  onFinalTopKChange: (value: number) => void
  onRetrieveCandidates: () => void
  onRerankCandidates: () => void
  onEnhance: (query: string) => void
  enhanceLoading?: boolean
  ocrDisabled?: boolean
}

export const SearchBar = forwardRef<HTMLInputElement, SearchBarProps>(function SearchBar(
  {
    globalTextQuery,
    textCandidateK,
    ocrQuery,
    ocrCandidateK,
    candidatePoolSize,
    textCandidateCount,
    ocrCandidateCount,
    stage2Queries,
    finalTopK,
    onGlobalTextQueryChange,
    onTextCandidateKChange,
    onOcrQueryChange,
    onOcrCandidateKChange,
    onStage2QueryTextChange,
    onStage2QueryTypeChange,
    onAddStage2Query,
    onRemoveStage2Query,
    onFinalTopKChange,
    onRetrieveCandidates,
    onRerankCandidates,
    onEnhance,
    enhanceLoading = false,
    ocrDisabled = false,
  },
  ref,
) {
  const hasStage1Input = globalTextQuery.trim().length > 0 || ocrQuery.trim().length > 0
  const activeStage2Queries = stage2Queries.filter((row) => row.text.trim().length > 0)
  const canRerank =
    candidatePoolSize !== null &&
    candidatePoolSize > 0 &&
    activeStage2Queries.length > 0 &&
    finalTopK >= 1 &&
    finalTopK <= candidatePoolSize
  const finalTopKMax = Math.min(candidatePoolSize ?? MAX_TOP_K, MAX_TOP_K)

  return (
    <div className={styles.flow}>
      <form
        className={styles.stage}
        onSubmit={(event) => {
          event.preventDefault()
          onRetrieveCandidates()
        }}
      >
        <div className={styles.stageHeader}>
          <span>Stage 1 — Global Retrieval</span>
          {candidatePoolSize !== null && (
            <strong>Candidate Pool: {candidatePoolSize} unique images</strong>
          )}
        </div>
        <div className={styles.stageGrid}>
          <label className={styles.fieldWide}>
            <span>Text Q0</span>
            <input
              ref={ref}
              aria-label="text q0"
              className={styles.input}
              type="text"
              placeholder="Five men sitting on a boat at night..."
              value={globalTextQuery}
              onChange={(event) => onGlobalTextQueryChange(event.target.value)}
            />
          </label>
          <NumberField
            label="Text candidates"
            ariaLabel="text candidates"
            value={textCandidateK}
            min={1}
            max={MAX_CANDIDATE_K}
            onChange={onTextCandidateKChange}
          />
          <label className={styles.fieldWide}>
            <span>Optional OCR</span>
            <input
              aria-label="stage 1 ocr"
              className={styles.input}
              type="text"
              placeholder="SAIGON"
              value={ocrQuery}
              disabled={ocrDisabled}
              onChange={(event) => onOcrQueryChange(event.target.value)}
            />
          </label>
          <NumberField
            label="OCR candidates"
            ariaLabel="ocr candidates"
            value={ocrCandidateK}
            min={1}
            max={MAX_CANDIDATE_K}
            disabled={ocrDisabled}
            onChange={onOcrCandidateKChange}
          />
        </div>
        <div className={styles.actions}>
          <button type="submit" className={styles.submit} disabled={!hasStage1Input}>
            Retrieve Candidates
          </button>
          <button
            type="button"
            className={styles.enhancer}
            disabled={!globalTextQuery.trim() || enhanceLoading}
            title="Viết lại Text Q0 bằng AI (QUEST nhánh 1)"
            onClick={() => onEnhance(globalTextQuery)}
          >
            AI enhancer{enhanceLoading ? '...' : ''}
          </button>
          {candidatePoolSize !== null && (
            <span className={styles.poolBreakdown}>
              Text {textCandidateCount} · OCR {ocrCandidateCount}
            </span>
          )}
        </div>
      </form>

      {candidatePoolSize !== null && (
        <section className={styles.stage}>
          <div className={styles.stageHeader}>
            <span>Stage 2 — Constraint Reranking</span>
            <strong>Candidate Pool: {candidatePoolSize}</strong>
          </div>
          <div className={styles.queryList}>
            {stage2Queries.map((row, index) => (
              <div className={styles.queryRow} key={row.id}>
                <select
                  aria-label={`stage 2 query type ${index + 1}`}
                  className={styles.typeSelect}
                  value={row.type}
                  onChange={(event) =>
                    onStage2QueryTypeChange(row.id, event.currentTarget.value as KisQueryType)
                  }
                >
                  <option value="text">Text</option>
                  <option value="ocr" disabled={ocrDisabled}>
                    OCR
                  </option>
                </select>
                <input
                  aria-label={`stage 2 query ${index + 1}`}
                  className={styles.input}
                  type="text"
                  placeholder={
                    row.type === 'ocr'
                      ? 'Text visible in the frame...'
                      : 'Constraint to rerank the candidate pool...'
                  }
                  value={row.text}
                  onChange={(event) => onStage2QueryTextChange(row.id, event.target.value)}
                />
                <button
                  type="button"
                  className={styles.remove}
                  aria-label={`remove stage 2 query ${index + 1}`}
                  title="Remove query"
                  onClick={() => onRemoveStage2Query(row.id)}
                >
                  ×
                </button>
              </div>
            ))}
          </div>
          <div className={styles.actions}>
            <button type="button" className={styles.addButton} onClick={onAddStage2Query}>
              + Add subquery
            </button>
            <NumberField
              label="Final Top-K"
              ariaLabel="final top k"
              value={finalTopK}
              min={MIN_TOP_K}
              max={Math.max(MIN_TOP_K, finalTopKMax)}
              onChange={onFinalTopKChange}
              compact
            />
            <button
              type="button"
              className={styles.submit}
              disabled={!canRerank}
              onClick={onRerankCandidates}
            >
              Rerank Candidates
            </button>
            {activeStage2Queries.length === 0 && (
              <span className={styles.hint}>
                Add at least one Stage 2 constraint to calculate final reranking.
              </span>
            )}
          </div>
        </section>
      )}
    </div>
  )
})

function NumberField({
  label,
  ariaLabel,
  value,
  min,
  max,
  disabled = false,
  compact = false,
  onChange,
}: {
  label: string
  ariaLabel: string
  value: number
  min: number
  max: number
  disabled?: boolean
  compact?: boolean
  onChange: (value: number) => void
}) {
  return (
    <label className={compact ? styles.fieldCompact : styles.field}>
      <span>{label}</span>
      <input
        aria-label={ariaLabel}
        className={styles.numberInput}
        type="number"
        min={min}
        max={max}
        value={value}
        disabled={disabled}
        onChange={(event) => onChange(Number(event.currentTarget.value))}
      />
    </label>
  )
}
