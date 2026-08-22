import { useState } from 'react'
import type { ReactNode } from 'react'
import type { AbePatchHeatmap, SearchHit } from '../api/types'
import styles from './KisScoreInspector.module.css'

export interface KisScoreInspectorProps {
  hit: SearchHit | null
  embedded?: boolean
}

function formatScore(value: number | undefined, digits = 3): string {
  if (value === undefined || !Number.isFinite(value)) return '-'
  return value.toFixed(digits)
}

export function KisScoreInspector({ hit, embedded = false }: KisScoreInspectorProps) {
  const displayScore = hit?.final_score ?? hit?.score
  const [heatmapOpen, setHeatmapOpen] = useState<Record<number, boolean>>({})

  if (!hit) {
    return (
      <ScoreContainer embedded={embedded}>
        <div className={styles.header}>
          <span>KIS Score</span>
          <span>-</span>
        </div>
      </ScoreContainer>
    )
  }

  return (
    <ScoreContainer embedded={embedded}>
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
          {hit.stage1.fused_image_score !== undefined && (
            <div className={styles.queryBlock}>
              <div className={styles.queryTitle}>Fused Image Rank</div>
              <ScoreRow label="sum cosine" value={hit.stage1.fused_image_score} />
              <ScoreRow label="global rank" value={hit.stage1.fused_image_rank} digits={0} />
            </div>
          )}
          {hit.stage1.text && (
            <div className={styles.queryBlock}>
              <div className={styles.queryTitle}>Global Text Q0 - FG-CLIP</div>
              <div className={styles.queryText} title={hit.stage1.text.query}>
                "{hit.stage1.text.query}"
              </div>
              <ScoreRow label="FG-CLIP raw cosine" value={hit.stage1.text.raw_cosine} />
              <ScoreRow label="Text rank" value={hit.stage1.text.rank} digits={0} />
            </div>
          )}
          {hit.stage1.text_queries && hit.stage1.text_queries.length > 1 && (
            <div className={styles.queryBlock}>
              <div className={styles.queryTitle}>Stage 1 Text Queries</div>
              {hit.stage1.text_queries.map((query) => (
                <div className={styles.pairBlock} key={`stage1-text-${query.query_index}`}>
                  <div className={styles.queryText} title={query.query}>
                    Q{query.query_index + 1}: "{query.query}"
                  </div>
                  <ScoreRow label="cosine" value={query.raw_cosine} />
                  <ScoreRow label="rank in union" value={query.rank} digits={0} />
                  <ScoreTextRow
                    label="original Top-K rank"
                    value={query.original_rank === null ? '-' : `#${query.original_rank}`}
                  />
                </div>
              ))}
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
          {hit.stage1.title && (
            <div className={styles.queryBlock}>
              <div className={styles.queryTitle}>Stage 1 Title</div>
              <div className={styles.queryText} title={hit.stage1.title.video_title}>
                "{hit.stage1.title.video_title}"
              </div>
              <ScoreRow label="Title score" value={hit.stage1.title.title_score} />
              <ScoreRow label="Title rank" value={hit.stage1.title.rank} digits={0} />
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
            <span>TEXT - FG-CLIP</span>
            <strong>{formatScore(hit.modalities.text.score)}</strong>
          </div>
          <ScoreRow label="Text GM" value={hit.modalities.text.geometric_mean} />
          {hit.modalities.text.queries.map((query) => (
            <div className={styles.queryBlock} key={`text-${query.query_index}`}>
              <div className={styles.queryTitle}>Query {query.query_index + 1}</div>
              <div className={styles.queryText} title={query.query}>
                "{query.query}"
              </div>
              <ScoreRow label="FG-CLIP raw cosine" value={query.raw_cosine} />
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
          <ScoreRow label="OCR GM" value={hit.modalities.ocr.geometric_mean} />
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
      {hit.modalities?.abe && (
        <section className={styles.section}>
          <div className={styles.sectionHeader}>
            <span>ABE - FG-CLIP</span>
            <strong>{formatScore(hit.modalities.abe.score)}</strong>
          </div>
          <ScoreRow label="ABE GM" value={hit.modalities.abe.geometric_mean} />
          {hit.modalities.abe.queries.map((query) => (
            <div className={styles.queryBlock} key={`abe-${query.query_index}`}>
              <div className={styles.queryTitle}>ABE Group {query.query_index + 1}</div>
              {query.pairs.map((pair, pairIndex) => (
                <div className={styles.pairBlock} key={`${pair.attribute}-${pair.object}`}>
                  <div className={styles.queryTitle}>Pair {pairIndex + 1}</div>
                  <ScoreTextRow label="Attribute" value={pair.attribute} />
                  <ScoreTextRow label="Object" value={pair.object} />
                  <ScoreTextRow label="Positive attrs" value={pair.positive_attributes.join(', ')} />
                  <ScoreTextRow
                    label="Negative attrs"
                    value={pair.negative_attributes.length ? pair.negative_attributes.join(', ') : '-'}
                  />
                </div>
              ))}
              <ScoreRow label="Base Local" value={query.base_local} />
              <ScoreRow label="Refined Local" value={query.refined_local} />
              <ScoreRow label="Binding Delta" value={query.binding_delta} />
              <ScoreRow label="Local Score" value={query.local_score} />
              <ScoreRow label="FG Global ABE" value={query.global_abe} />
              <ScoreRow label="Raw ABE Score" value={query.raw_score} />
              <ScoreRow label="rank" value={query.rank} digits={0} suffix={rankSuffix(hit)} />
              <ScoreRow label="percentile" value={query.rank_percentile} />
              {query.patch_heatmap && (
                <>
                  <button
                    type="button"
                    className={styles.heatmapToggle}
                    onClick={() =>
                      setHeatmapOpen((prev) => ({
                        ...prev,
                        [query.query_index]: !prev[query.query_index],
                      }))
                    }
                  >
                    {heatmapOpen[query.query_index] ? 'Hide ABE Patch Heatmap' : 'Show ABE Patch Heatmap'}
                  </button>
                  {heatmapOpen[query.query_index] && (
                    <AbePatchHeatmapView heatmap={query.patch_heatmap} />
                  )}
                </>
              )}
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
    </ScoreContainer>
  )
}

function ScoreContainer({
  embedded,
  children,
}: {
  embedded: boolean
  children: ReactNode
}) {
  if (embedded) {
    return <div className={styles.embedded}>{children}</div>
  }
  return (
    <aside className={styles.panel} aria-label="KIS score details">
      {children}
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
  value: number | undefined
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

function ScoreTextRow({ label, value }: { label: string; value: string }) {
  return (
    <div className={styles.row}>
      <span>{label}</span>
      <span title={value}>{value}</span>
    </div>
  )
}

function AbePatchHeatmapView({ heatmap }: { heatmap: AbePatchHeatmap }) {
  const base = new Map(heatmap.base_top_patches.map((hit) => [hit.patch_index, hit.score]))
  const refined = new Map(heatmap.refined_top_patches.map((hit) => [hit.patch_index, hit.score]))
  const cells = Array.from({ length: heatmap.patch_count }, (_, index) => {
    const baseScore = base.get(index)
    const refinedScore = refined.get(index)
    const strength = Math.max(baseScore ?? 0, refinedScore ?? 0, 0)
    const opacity = Math.max(0.12, Math.min(0.95, (strength + 1) / 2))
    return (
      <span
        key={index}
        className={styles.heatmapCell}
        title={`Patch ${index} · base ${formatScore(baseScore)} · refined ${formatScore(refinedScore)}`}
        style={{
          background:
            refinedScore !== undefined
              ? `rgba(16, 185, 129, ${opacity})`
              : baseScore !== undefined
                ? `rgba(59, 130, 246, ${opacity})`
                : 'rgba(71, 85, 105, 0.28)',
        }}
      />
    )
  })

  return (
    <div className={styles.heatmapBlock}>
      <div
        className={styles.heatmapGrid}
        style={{ gridTemplateColumns: `repeat(${heatmap.grid_width}, 1fr)` }}
      >
        {cells}
      </div>
      <div className={styles.heatmapLegend}>
        <span>Base</span>
        <span>Refined</span>
      </div>
    </div>
  )
}
