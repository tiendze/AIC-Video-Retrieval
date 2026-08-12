// Mirrors configs/search/search.yaml -- default_top_k / max_top_k. Kept in
// sync by hand like api/types.ts; max_top_k=100 is the AIC 2026 submission
// cap (references.md §1.2), not an arbitrary frontend choice.
export const DEFAULT_TOP_K = 20
export const MIN_TOP_K = 1
export const MAX_TOP_K = 100
