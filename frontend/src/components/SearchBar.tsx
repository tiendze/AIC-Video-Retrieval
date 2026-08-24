import { forwardRef } from 'react'
import {
  MAX_ASR_QUERY_LENGTH,
  MAX_ASR_VIDEO_CANDIDATE_K,
  MAX_CANDIDATE_K,
  MAX_TOP_K,
  MIN_TOP_K,
} from '../api/constants'
import type { AsrGate, KisQueryRow, KisQueryType } from '../api/types'
import { QueryInput } from './QueryInput'
import styles from './SearchBar.module.css'

export interface SearchBarProps {
  textQueries: string[]
  onTextQueryChange: (index: number, value: string) => void
  onAddTextQuery: () => void
  onRemoveTextQuery: (index: number) => void
  globalTextQuery: string
  textCandidateK: number
  ocrQuery: string
  ocrCandidateK: number
  titleQuery: string
  titleCandidateK: number
  asrQuery: string
  asrVideoCandidateK: number
  asrGate: AsrGate | null
  asrDisabled: boolean
  asrReadinessMessage: string | null
  candidatePoolSize: number | null
  textCandidateCount: number
  ocrCandidateCount: number
  titleCandidateCount: number
  stage2Queries: KisQueryRow[]
  finalTopK: number
  onGlobalTextQueryChange: (value: string) => void
  onTextCandidateKChange: (value: number) => void
  onOcrQueryChange: (value: string) => void
  onOcrCandidateKChange: (value: number) => void
  onTitleQueryChange: (value: string) => void
  onTitleCandidateKChange: (value: number) => void
  onAsrQueryChange: (value: string) => void
  onAsrVideoCandidateKChange: (value: number) => void
  onStage2QueryTextChange: (id: string, value: string) => void
  onStage2QueryTypeChange: (id: string, type: KisQueryType) => void
  onAbePairAttributeChange: (queryId: string, pairId: string, value: string) => void
  onAbePairObjectChange: (queryId: string, pairId: string, value: string) => void
  onAddAbePair: (queryId: string) => void
  onRemoveAbePair: (queryId: string, pairId: string) => void
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
    textQueries,
    onTextQueryChange,
    onAddTextQuery,
    onRemoveTextQuery,
    globalTextQuery,
    textCandidateK,
    ocrQuery,
    ocrCandidateK,
    titleQuery,
    titleCandidateK,
    asrQuery,
    asrVideoCandidateK,
    asrGate,
    asrDisabled,
    asrReadinessMessage,
    candidatePoolSize,
    textCandidateCount,
    ocrCandidateCount,
    titleCandidateCount,
    stage2Queries,
    finalTopK,
    onGlobalTextQueryChange,
    onTextCandidateKChange,
    onOcrQueryChange,
    onOcrCandidateKChange,
    onTitleQueryChange,
    onTitleCandidateKChange,
    onAsrQueryChange,
    onAsrVideoCandidateKChange,
    onStage2QueryTextChange,
    onStage2QueryTypeChange,
    onAbePairAttributeChange,
    onAbePairObjectChange,
    onAddAbePair,
    onRemoveAbePair,
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
  const hasTextInput = textQueries.some((query) => query.trim().length > 0)
  const hasStage1Input =
    hasTextInput || ocrQuery.trim().length > 0 || titleQuery.trim().length > 0
  const activeStage2Queries = stage2Queries.filter((row) =>
    row.type === 'abe'
      ? row.pairs.some((pair) => pair.attribute.trim() && pair.object.trim())
      : row.text.trim().length > 0,
  )
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
          <span>Stage 1 - Candidate Retrieval</span>
          {candidatePoolSize !== null && (
            <strong>Candidate Pool: {candidatePoolSize} unique images</strong>
          )}
        </div>
        <div className={styles.stageGrid}>
          <div className={styles.fieldWide}>
            <div className={styles.queryListHeader}>
              <span>Text queries (FG-CLIP)</span>
              <button type="button" className={styles.addInlineButton} onClick={onAddTextQuery}>
                + Add query
              </button>
            </div>
            <div className={styles.textQueryList}>
              {textQueries.map((query, index) => (
                <div className={styles.textQueryRow} key={`stage1-text-${index}`}>
                  <QueryInput
                    ref={index === 0 ? ref : undefined}
                    aria-label={index === 0 ? 'text q0' : `text q${index + 1}`}
                    type="text"
                    placeholder={
                      index === 0
                        ? 'Aerial view of a dam and reservoir...'
                        : 'Concrete dam under heavy rain...'
                    }
                    value={query}
                    onValueChange={(value) => {
                      if (index === 0) onGlobalTextQueryChange(value)
                      else onTextQueryChange(index, value)
                    }}
                  />
                  <button
                    type="button"
                    className={styles.removeMini}
                    disabled={textQueries.length === 1}
                    aria-label={`remove text query ${index + 1}`}
                    title="Remove query"
                    onClick={() => onRemoveTextQuery(index)}
                  >
                    x
                  </button>
                </div>
              ))}
            </div>
          </div>
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
            <QueryInput
              aria-label="stage 1 ocr"
              type="text"
              placeholder="SAIGON"
              value={ocrQuery}
              disabled={ocrDisabled}
              onValueChange={onOcrQueryChange}
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
          <label className={styles.fieldWide}>
            <span>Optional Title OCR</span>
            <QueryInput
              aria-label="stage 1 title"
              type="text"
              placeholder="chicken, football, rain..."
              value={titleQuery}
              onValueChange={onTitleQueryChange}
            />
          </label>
          <NumberField
            label="Title OCR candidates"
            ariaLabel="title candidates"
            value={titleCandidateK}
            min={1}
            max={MAX_CANDIDATE_K}
            onChange={onTitleCandidateKChange}
          />
          <label className={styles.fieldWide}>
            <span>Optional ASR topic/dialogue</span>
            <QueryInput
              aria-label="stage 1 asr"
              type="text"
              maxLength={MAX_ASR_QUERY_LENGTH}
              placeholder="cong trinh thuy loi dap ho chua"
              value={asrQuery}
              disabled={asrDisabled}
              onValueChange={onAsrQueryChange}
            />
          </label>
          <NumberField
            label="ASR candidate videos"
            ariaLabel="asr candidate videos"
            value={asrVideoCandidateK}
            min={1}
            max={MAX_ASR_VIDEO_CANDIDATE_K}
            disabled={asrDisabled}
            onChange={onAsrVideoCandidateKChange}
          />
        </div>
        {asrReadinessMessage && (
          <span className={styles.asrReadiness}>{asrReadinessMessage}</span>
        )}
        <div className={styles.actions}>
          <button type="submit" className={styles.submit} disabled={!hasStage1Input}>
            Retrieve Candidates
          </button>
          <button
            type="button"
            className={styles.enhancer}
            disabled={!globalTextQuery.trim() || enhanceLoading}
            title="Viet lai Text Q0 bang AI (QUEST nhanh 1)"
            onClick={() => onEnhance(globalTextQuery)}
          >
            AI enhancer{enhanceLoading ? '...' : ''}
          </button>
          {candidatePoolSize !== null && (
            <span className={styles.poolBreakdown}>
              {`Text ${textCandidateCount} · OCR ${ocrCandidateCount} · Title OCR ${titleCandidateCount}${
                asrGate ? ` · ASR ${asrGate.candidate_video_count} videos` : ''
              }`}
            </span>
          )}
          {asrGate?.candidate_video_count === 0 && (
            <span className={styles.hint}>
              No transcript video matched - clear or revise the ASR query.
            </span>
          )}
        </div>
      </form>

      {candidatePoolSize !== null && (
        <section className={styles.stage}>
          <div className={styles.stageHeader}>
            <span>Stage 2 - Constraint Reranking</span>
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
                  <option value="abe">ABE</option>
                </select>
                {row.type === 'abe' ? (
                  <div className={styles.abeGroup}>
                    <div className={styles.abeHeader} aria-hidden="true">
                      <span>Attribute</span>
                      <span>Object</span>
                      <span />
                    </div>
                    {row.pairs.map((pair, pairIndex) => (
                      <div className={styles.abePair} key={pair.id}>
                        <label className={styles.abeField}>
                          <span className={styles.visuallyHidden}>Attribute {pairIndex + 1}</span>
                          <QueryInput
                            aria-label={`abe attribute ${index + 1}.${pairIndex + 1}`}
                            type="text"
                            placeholder="red"
                            value={pair.attribute}
                            onValueChange={(value) =>
                              onAbePairAttributeChange(row.id, pair.id, value)
                            }
                          />
                        </label>
                        <label className={styles.abeField}>
                          <span className={styles.visuallyHidden}>Object {pairIndex + 1}</span>
                          <QueryInput
                            aria-label={`abe object ${index + 1}.${pairIndex + 1}`}
                            type="text"
                            placeholder="seasoning packet"
                            value={pair.object}
                            onValueChange={(value) =>
                              onAbePairObjectChange(row.id, pair.id, value)
                            }
                          />
                        </label>
                        <button
                          type="button"
                          className={`${styles.remove} ${styles.removePair}`}
                          aria-label={`remove abe pair ${index + 1}.${pairIndex + 1}`}
                          title="Remove pair"
                          onClick={() => onRemoveAbePair(row.id, pair.id)}
                        >
                          x
                        </button>
                      </div>
                    ))}
                    <div className={styles.abeActions}>
                      <button
                        type="button"
                        className={styles.addPairButton}
                        onClick={() => onAddAbePair(row.id)}
                      >
                        + Add pair
                      </button>
                    </div>
                  </div>
                ) : (
                  <QueryInput
                    aria-label={`stage 2 query ${index + 1}`}
                    type="text"
                    placeholder={
                      row.type === 'ocr'
                        ? 'Text visible in the frame...'
                        : 'Constraint to rerank the candidate pool...'
                    }
                    value={row.text}
                    onValueChange={(value) => onStage2QueryTextChange(row.id, value)}
                  />
                )}
                <button
                  type="button"
                  className={`${styles.remove} ${styles.removeQuery}`}
                  aria-label={`remove stage 2 query ${index + 1}`}
                  title="Remove query"
                  onClick={() => onRemoveStage2Query(row.id)}
                >
                  x
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
