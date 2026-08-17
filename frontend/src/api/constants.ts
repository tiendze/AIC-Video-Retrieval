// Mirrors configs/search/search.yaml -- default_top_k / max_top_k. Kept in
// sync by hand like api/types.ts; max_top_k=100 is the AIC 2026 submission
// cap (references.md §1.2), not an arbitrary frontend choice.
export const DEFAULT_TOP_K = 20
export const MIN_TOP_K = 1
export const MAX_TOP_K = 100

// Mirrors configs/search/trake.yaml -- POST /api/trake ranks N-keyframe
// sequences, a different unit than /api/search's per-keyframe top_k, so it
// gets its own default even though MAX_TOP_K is currently the same
// submission-cap ceiling for both.
export const TRAKE_DEFAULT_TOP_K = 10
export const TRAKE_DEFAULT_THRESHOLD = 0

// Semantic mode's supplementary OCR conditions (frontend-only for now --
// POST /api/search does not accept ocr_queries yet). Frontend-chosen cap to
// keep the row list usable, not a backend constraint.
export const MAX_OCR_QUERIES = 10
