// Shared by useSearch.ts (KIS Stage 1, phrases grouped per text query) and
// useTrake.ts (one phrase list per event) -- both dedupe/trim the same way
// before sending to app/schemas/request/search.py: KisCandidateRequest's
// local_phrases_by_query / app/schemas/request/trake.py: TrakeEventRequest's
// local_phrases, which apply the equivalent normalization server-side
// (_normalize_local_phrase_group).
export function cleanLocalPhraseGroup(phrases: string[] | undefined): string[] {
  const cleaned: string[] = []
  const seen = new Set<string>()
  for (const phrase of phrases ?? []) {
    const stripped = phrase.trim()
    if (!stripped) continue
    const key = stripped.toLocaleLowerCase()
    if (seen.has(key)) continue
    seen.add(key)
    cleaned.push(stripped)
  }
  return cleaned
}
