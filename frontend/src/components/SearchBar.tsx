import { forwardRef } from 'react'
import {
  MAX_ASR_QUERY_LENGTH,
  MAX_ASR_VIDEO_CANDIDATE_K,
  MAX_CANDIDATE_K,
} from '../api/constants'
import type { AsrGate } from '../api/types'
import { QueryInput } from './QueryInput'
import styles from './SearchBar.module.css'

export interface SearchBarProps {
  textQueries: string[]
  onTextQueryChange: (index: number, value: string) => void
  onAddTextQuery: () => void
  onRemoveTextQuery: (index: number) => void
  localPhrasesByQuery: string[][]
  onLocalPhraseChange: (queryIndex: number, phraseIndex: number, value: string) => void
  onAddLocalPhrase: (queryIndex: number) => void
  onRemoveLocalPhrase: (queryIndex: number, phraseIndex: number) => void
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
  onGlobalTextQueryChange: (value: string) => void
  onTextCandidateKChange: (value: number) => void
  onOcrQueryChange: (value: string) => void
  onOcrCandidateKChange: (value: number) => void
  onTitleQueryChange: (value: string) => void
  onTitleCandidateKChange: (value: number) => void
  onAsrQueryChange: (value: string) => void
  onAsrVideoCandidateKChange: (value: number) => void
  onRetrieveCandidates: () => void
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
    localPhrasesByQuery,
    onLocalPhraseChange,
    onAddLocalPhrase,
    onRemoveLocalPhrase,
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
    onGlobalTextQueryChange,
    onTextCandidateKChange,
    onOcrQueryChange,
    onOcrCandidateKChange,
    onTitleQueryChange,
    onTitleCandidateKChange,
    onAsrQueryChange,
    onAsrVideoCandidateKChange,
    onRetrieveCandidates,
    onEnhance,
    enhanceLoading = false,
    ocrDisabled = false,
  },
  ref,
) {
  const hasTextInput = textQueries.some((query) => query.trim().length > 0)
  const hasAsrInput = !asrDisabled && asrQuery.trim().length > 0
  const hasStage1Input =
    hasTextInput || ocrQuery.trim().length > 0 || titleQuery.trim().length > 0 || hasAsrInput

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
                <div className={styles.textQueryBlock} key={`stage1-text-${index}`}>
                  <div className={styles.textQueryLabel}>Text Query {index + 1}</div>
                  <div className={styles.textQueryRow}>
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
                  <div className={styles.localPhraseGroup}>
                    <div className={styles.localPhraseHeader}>
                      <span>Local phrases</span>
                      <button
                        type="button"
                        className={styles.addInlineButton}
                        onClick={() => onAddLocalPhrase(index)}
                      >
                        + Add local phrase
                      </button>
                    </div>
                    {(localPhrasesByQuery[index] ?? []).map((phrase, phraseIndex) => (
                      <div
                        className={styles.localPhraseRow}
                        key={`local-phrase-${index}-${phraseIndex}`}
                      >
                        <QueryInput
                          aria-label={`local phrase ${index + 1}.${phraseIndex + 1}`}
                          type="text"
                          placeholder="red seasoning packet"
                          value={phrase}
                          onValueChange={(value) =>
                            onLocalPhraseChange(index, phraseIndex, value)
                          }
                        />
                        <button
                          type="button"
                          className={styles.removeMini}
                          aria-label={`remove local phrase ${index + 1}.${phraseIndex + 1}`}
                          title="Remove local phrase"
                          onClick={() => onRemoveLocalPhrase(index, phraseIndex)}
                        >
                          x
                        </button>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
            <p className={styles.localPhraseHint}>
              Local phrases score visual regions: use phrases like "red seasoning packet" or
              "man holding a red seasoning packet".
            </p>
          </div>
          <NumberField
            label="Text candidates"
            ariaLabel="text candidates"
            value={textCandidateK}
            min={1}
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
            <span>Optional ASR keyword/phrase</span>
            <QueryInput
              aria-label="stage 1 asr"
              type="text"
              maxLength={MAX_ASR_QUERY_LENGTH}
              placeholder="con cá, cân, bao nhiêu ký"
              value={asrQuery}
              disabled={asrDisabled}
              onValueChange={onAsrQueryChange}
            />
          </label>
          <NumberField
            label="ASR max videos"
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
                asrGate ? ` · ASR ${asrGate.candidate_video_count} detected` : ''
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
  max?: number
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
        {...(max === undefined ? {} : { max })}
        value={value}
        disabled={disabled}
        onChange={(event) => onChange(Number(event.currentTarget.value))}
      />
    </label>
  )
}
