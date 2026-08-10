"""Random baseline (plan §P0.9) -- the absolute floor every real channel
must clear: "hệ thống phải vượt xa baseline random. Không vượt ⇒ có bug,
không phải cần tối ưu."

`search()` takes and ignores a query string so `eval/run.py` can drive this
baseline through the exact same per-query loop as `SearchService`/
`BM25Baseline` (uniform call shape), even though a random baseline by
definition never looks at the query.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class KeyframeRecord:
    """One row of the full-corpus sampling pool -- built from
    `MongoStore.scroll_all_keyframes()` by `eval/run.py`."""

    keyframe_id: str
    video_id: str
    timestamp: float


class RandomBaseline:
    def __init__(self, pool: Sequence[KeyframeRecord], seed: int) -> None:
        if not pool:
            raise ValueError("RandomBaseline: pool is empty")
        self._pool = list(pool)
        self._rng = random.Random(seed)

    async def search(self, query: str, top_k: int) -> list[KeyframeRecord]:
        k = min(top_k, len(self._pool))
        return self._rng.sample(self._pool, k)
