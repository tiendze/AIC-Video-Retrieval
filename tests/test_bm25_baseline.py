"""RED reproducer for eval/baselines/bm25.py (plan §P0.9).

Deliberately a thin delegation, not a second lexical index: Elasticsearch's
default similarity *is* BM25, so this is mechanistically identical to the
`ocr` search channel (`SearchService._search_ocr` -> `EsStore.search`).
Confirmed with the project owner (2026-08-10): the `bm25` baseline row and
the `ocr` channel row in eval/run.py's report are expected to carry the same
numbers -- that equality is the point (a sanity floor for the `ocr` channel
to clear), not a bug to fix.
"""

from __future__ import annotations

from typing import Any

from eval.baselines.bm25 import BM25Baseline


class _FakeEsStore:
    def __init__(self, hits: list[dict[str, Any]]) -> None:
        self._hits = hits
        self.search_calls: list[tuple[str, int]] = []

    async def search(self, query_text: str, top_k: int) -> list[dict]:
        self.search_calls.append((query_text, top_k))
        return self._hits


async def test_search_delegates_to_es_store_search_unchanged():
    hits = [{"keyframe_id": "k1", "video_id": "v1", "timestamp": 1.0, "score": 3.2}]
    es_store = _FakeEsStore(hits)
    baseline = BM25Baseline(es_store)

    results = await baseline.search("tin tuc", top_k=5)

    assert es_store.search_calls == [("tin tuc", 5)]
    assert results == hits
