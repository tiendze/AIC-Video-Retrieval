import { ApiError } from './client'

// Shared by every hook that fires an AbortController-cancellable request
// (useSearch, useQueryRewrite, useImageCandidates) -- extracted once three
// near-identical copies existed (DRY, .claude/rules/ecc/common/coding-style.md).
export function isAbortError(err: unknown): boolean {
  return err instanceof DOMException && err.name === 'AbortError'
}

export function messageFor(err: unknown, fallback: string): string {
  if (err instanceof ApiError) return err.message
  return fallback
}
