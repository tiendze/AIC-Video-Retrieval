"""BM25 baseline (plan §P0.9) -- deliberately a thin delegation, not a
second lexical index: Elasticsearch's default similarity *is* BM25, so this
is mechanistically identical to the `ocr` search channel
(`SearchService._search_ocr` -> `EsStore.search`). It exists so
`eval/run.py` can drive `random`/`bm25` through one uniform "baseline" loop,
separate from the per-channel (`semantic`/`ocr`) reporting loop, without
duplicating any search logic.

Confirmed with the project owner (2026-08-10): the `bm25` baseline row and
the `ocr` channel row in the report are expected to carry the SAME numbers
-- that equality is the point (a sanity floor for `ocr` to clear), not a
bug. "Với bộ nhãn tự sinh, bm25.py đổi vai từ phụ thành chính" (plan §P0.9):
an unusually high `bm25`/`ocr` score is the label-leakage signal, not
evidence the system is good (CLAUDE.md invariant: model sinh nhãn khác model
bị đánh giá) -- see `eval/metrics/leakage.py` for the direct measurement.
"""

from __future__ import annotations

from typing import Protocol


class _OcrSearchStore(Protocol):
    async def search(self, query_text: str, top_k: int) -> list[dict]: ...


class BM25Baseline:
    def __init__(self, es_store: _OcrSearchStore) -> None:
        self._es_store = es_store

    async def search(self, query: str, top_k: int) -> list[dict]:
        return await self._es_store.search(query, top_k)
