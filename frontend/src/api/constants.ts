// Mirrors configs/search/search.yaml -- default_top_k / max_top_k. Kept in
// sync by hand like api/types.ts; max_top_k=100 is the AIC 2026 submission
// cap (references.md §1.2), not an arbitrary frontend choice.
export const DEFAULT_TOP_K = 20
export const MIN_TOP_K = 1
export const MAX_TOP_K = 100
export const DEFAULT_TEXT_CANDIDATE_K = 300
export const DEFAULT_OCR_CANDIDATE_K = 100
export const DEFAULT_TITLE_CANDIDATE_K = 100
export const MAX_CANDIDATE_K = 1000
export const DEFAULT_RRF_K = 60
export const DEFAULT_ASR_VIDEO_CANDIDATE_K = 20
export const MAX_ASR_VIDEO_CANDIDATE_K = 100
export const MAX_ASR_QUERY_LENGTH = 1000

// Mirrors configs/search/trake.yaml -- POST /api/trake ranks N-keyframe
// sequences, a different unit than /api/search's per-keyframe top_k, so it
// gets its own default even though MAX_TOP_K is currently the same
// submission-cap ceiling for both.
//
// The default is the full 100 slots: the pipeline lays them out round-robin
// across videos (best sequence of each video first, frame variants after), so
// asking for fewer than 100 throws away the R@50/R@100 coverage that layout
// exists to buy.
export const TRAKE_DEFAULT_TOP_K = 100
// Sequence scores are z-scored, so 0 means "an average frame of the corpus".
// The default sits far below that: the operator raises it to cut weak rows,
// rather than the API silently hiding candidates by default.
export const TRAKE_DEFAULT_THRESHOLD = -100

// Mirrors configs/search/trake.yaml -- event_text_top_k. How many keyframes
// Stage 1 pulls in for one event's Text channel before chaining; this is what
// the per-event "N keyframes / M videos" diagnostic actually counts when only
// Text is filled. Operator-editable per event (EventSequenceInput's Top K
// box) instead of fixed, per TrakeEventRequest.text_top_k's ge=1,le=1000
// bound (app/schemas/request/trake.py, MAX_CANDIDATE_K).
export const TRAKE_DEFAULT_EVENT_TEXT_TOP_K = 1000
export const TRAKE_MIN_EVENT_TEXT_TOP_K = 1
export const TRAKE_MAX_EVENT_TEXT_TOP_K = MAX_CANDIDATE_K
