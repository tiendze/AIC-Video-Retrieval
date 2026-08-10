"""RED reproducer for eval/baselines/random.py (plan §P0.9) -- the absolute
floor every real channel must clear ("hệ thống phải vượt xa baseline
random. Không vượt ⇒ có bug, không phải cần tối ưu").
"""

from __future__ import annotations

import pytest

from eval.baselines.random import KeyframeRecord, RandomBaseline

_POOL = [
    KeyframeRecord(keyframe_id=f"k{i}", video_id=f"v{i}", timestamp=float(i))
    for i in range(10)
]


async def test_search_samples_top_k_items_from_the_pool():
    baseline = RandomBaseline(pool=_POOL, seed=42)

    results = await baseline.search("any query", top_k=3)

    assert len(results) == 3
    assert all(r in _POOL for r in results)


async def test_search_ignores_the_query_text():
    baseline_a = RandomBaseline(pool=_POOL, seed=42)
    baseline_b = RandomBaseline(pool=_POOL, seed=42)

    results_a = await baseline_a.search("caption A", top_k=3)
    results_b = await baseline_b.search("a totally different caption B", top_k=3)

    assert results_a == results_b


async def test_search_is_deterministic_given_the_same_seed():
    baseline_a = RandomBaseline(pool=_POOL, seed=7)
    baseline_b = RandomBaseline(pool=_POOL, seed=7)

    assert await baseline_a.search("q", top_k=5) == await baseline_b.search("q", top_k=5)


async def test_search_returns_the_whole_pool_when_top_k_exceeds_pool_size():
    baseline = RandomBaseline(pool=_POOL, seed=1)

    results = await baseline.search("q", top_k=1000)

    assert len(results) == len(_POOL)


def test_raises_on_empty_pool():
    with pytest.raises(ValueError):
        RandomBaseline(pool=[], seed=1)
