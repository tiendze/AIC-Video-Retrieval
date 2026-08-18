import type { SearchHit } from '../api/types'
import styles from './KisScoreInspector.module.css'

export interface KisScoreInspectorProps {
  hit: SearchHit | null
}

function formatScore(value: number | undefined, digits = 3): string {
  if (value === undefined || !Number.isFinite(value)) return '-'
  return value.toFixed(digits)
}

export function KisScoreInspector({ hit }: KisScoreInspectorProps) {
  const displayScore = hit?.final_score ?? hit?.score

  if (!hit) {
    return (
      <aside className={styles.panel} aria-label="KIS score details">
        <div className={styles.header}>
          <span>KIS Score</span>
          <span>-</span>
        </div>
      </aside>
    )
  }

  return (
    <aside className={styles.panel} aria-label="KIS score details">
      <div className={styles.header}>
        <span>{hit.final_score === undefined ? 'Stage 1 Candidate' : 'Final Geometric Mean'}</span>
        <span>{hit.final_score === undefined ? formatScore(displayScore) : formatScore(hit.final_score)}</span>
      </div>
      <section className={styles.section}>
        <div className={styles.keyframe}>{hit.keyframe_id}</div>
      </section>
      {hit.stage1 && (
        <section className={styles.section}>
          <div className={styles.sectionHeader}>
            <span>STAGE 1</span>
            <strong>Candidate retrieval only</strong>
          </div>
          <div className={styles.sourceList}>Source: {hit.stage1.source_modalities.join(' + ')}</div>
          {hit.stage1.text && (
            <div className={styles.queryBlock}>
              <div className={styles.queryTitle}>Global Text Q0</div>
              <div className={styles.queryText} title={hit.stage1.text.query}>
                "{hit.stage1.text.query}"
              </div>
              <ScoreRow label="raw cosine" value={hit.stage1.text.raw_cosine} />
              <ScoreRow label="Text rank" value={hit.stage1.text.rank} digits={0} />
            </div>
          )}
          {hit.stage1.ocr && (
            <div className={styles.queryBlock}>
              <div className={styles.queryTitle}>Stage 1 OCR</div>
              <div className={styles.queryText} title={hit.stage1.ocr.query}>
                "{hit.stage1.ocr.query}"
              </div>
              <ScoreRow label="BM25 raw" value={hit.stage1.ocr.raw_bm25} />
              <ScoreRow label="BM25 normalized" value={hit.stage1.ocr.bm25_normalized} />
              <ScoreRow label="Coverage" value={hit.stage1.ocr.coverage} />
              <ScoreRow label="OCR query score" value={hit.stage1.ocr.query_score} />
              <ScoreRow label="OCR rank" value={hit.stage1.ocr.rank} digits={0} />
            </div>
          )}
        </section>
      )}
      {hit.final_score === undefined && (
        <section className={styles.section}>
          <div className={styles.notice}>
            Add at least one Stage 2 constraint to calculate the final reranking.
          </div>
        </section>
      )}
      {hit.modalities?.text && (
        <section className={styles.section}>
          <div className={styles.sectionHeader}>
            <span>STAGE 2 — TEXT</span>
            <strong>{formatScore(hit.modalities.text.score)}</strong>
          </div>
          <ScoreRow label="Geometric Mean" value={hit.modalities.text.geometric_mean} />
          {hit.modalities.text.queries.map((query) => (
            <div className={styles.queryBlock} key={`text-${query.query_index}`}>
              <div className={styles.queryTitle}>Query {query.query_index + 1}</div>
              <div className={styles.queryText} title={query.query}>
                "{query.query}"
              </div>
              <ScoreRow label="raw cosine" value={query.raw_cosine} />
              <ScoreRow label="rank" value={query.rank} digits={0} suffix={rankSuffix(hit)} />
              <ScoreRow label="percentile" value={query.rank_percentile} />
            </div>
          ))}
        </section>
      )}
      {hit.modalities?.ocr && (
        <section className={styles.section}>
          <div className={styles.sectionHeader}>
            <span>STAGE 2 — OCR</span>
            <strong>{formatScore(hit.modalities.ocr.score)}</strong>
          </div>
          <ScoreRow label="Geometric Mean" value={hit.modalities.ocr.geometric_mean} />
          {hit.modalities.ocr.queries.map((query) => (
            <div className={styles.queryBlock} key={`ocr-${query.query_index}`}>
              <div className={styles.queryTitle}>Query {query.query_index + 1}</div>
              <div className={styles.queryText} title={query.query}>
                "{query.query}"
              </div>
              <ScoreRow label="BM25 raw" value={query.raw_bm25} />
              <ScoreRow label="BM25 normalized" value={query.bm25_normalized} />
              <ScoreRow label="Coverage" value={query.coverage} />
              <ScoreRow label="OCR query score" value={query.query_score} />
              <ScoreRow label="rank" value={query.rank} digits={0} suffix={rankSuffix(hit)} />
              <ScoreRow label="percentile" value={query.rank_percentile} />
            </div>
          ))}
        </section>
      )}
      {hit.final_score !== undefined && (
        <section className={styles.section}>
          <div className={styles.sectionHeader}>
            <span>FINAL GM</span>
            <strong>{formatScore(hit.final_score)}</strong>
          </div>
        </section>
      )}
    </aside>
  )
}

function rankSuffix(hit: SearchHit): string {
  return hit.candidate_pool_size ? ` / ${hit.candidate_pool_size}` : ''
}

function ScoreRow({
  label,
  value,
  digits,
  suffix = '',
}: {
  label: string
  value: number
  digits?: number
  suffix?: string
}) {
  return (
    <div className={styles.row}>
      <span>{label}</span>
      <span>
        {formatScore(value, digits)}
        {suffix}
      </span>
    </div>
  )
}
