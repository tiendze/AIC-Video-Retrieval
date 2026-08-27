"""split_into_sentences -- LLM-free query decomposition for KIS Stage 1.

Measured on the 25 SOTUYEN1 contest queries (git 5191098): 9/25 exceed
FG-CLIP's 77-token cap after VI->EN translation, and `truncation=True`
(configs/model/fg_clip.yaml) drops the tail silently -- for
query-p1-16-trake.txt the tail is TRAKE event E3, one of three mandatory
keypoints (CLAUDE.md §3: this is exactly the "sai âm thầm" class of bug).

The fix does not need an LLM: these queries are already written as multiple
sentences (periods, question marks, or one clause per line). Splitting on
that existing punctuation and translating each piece separately keeps every
piece under the cap -- measured 0/25 split fragments still over 77 tokens.

`retrieve_kis_candidates` (app/services/search_service.py) calls
`split_into_sentences()` on `text_query`, translates every piece, searches
Milvus once per piece, and OR-unions the resulting candidates by keeping the
highest cosine score seen per keyframe -- a candidate counts if it matches
ANY piece well. This is a first-stage recall widening, the mirror of
`geometric_mean_score` (app/services/kis_scoring.py)'s Stage 2 AND-fusion:
see docs/research-query-refinement.md §4-§5 for why Stage 1 must be OR and
Stage 2 must stay AND.

A single-sentence query (no terminal punctuation, no newline) round-trips to
exactly `[text.strip()]` -- unchanged behaviour, so every existing Stage 1
test keeps passing byte-for-byte.
"""

from __future__ import annotations

import re

# Splits after a Vietnamese sentence terminator (kept with the sentence) or on
# a blank/single newline (dropped) -- covers both the punctuated queries
# (KIS/QA) and the one-clause-per-line queries (some multi-shot KIS queries).
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")


def split_into_sentences(text: str) -> list[str]:
    """Splits `text` into non-empty, whitespace-trimmed sentence pieces.

    Returns `[]` for empty/whitespace-only input, and `[text.strip()]`
    (one piece) when no boundary is found -- callers must not special-case
    the single-sentence case."""
    return [piece.strip() for piece in _SENTENCE_BOUNDARY.split(text) if piece.strip()]
