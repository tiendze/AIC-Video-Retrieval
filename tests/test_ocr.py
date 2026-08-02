"""RED reproducer for pipelines/ocr.py (P0.7 TEXT branch).

Uses a fake OcrClient/EsStore (Protocol test doubles) — no real Gemini API
key or Elasticsearch needed. `has_likely_text`/`deduplicate_by_phash` are
pure and tested directly against synthetic images.
"""

from __future__ import annotations

from PIL import Image, ImageDraw

from pipelines.ocr import (
    OcrCandidate,
    OcrConfig,
    deduplicate_by_phash,
    has_likely_text,
    run_ocr,
)


def _text_image() -> Image.Image:
    image = Image.new("RGB", (200, 100), color="white")
    draw = ImageDraw.Draw(image)
    draw.text((10, 40), "Breaking News 12:00 Tin Tuc Moi Nhat", fill="black")
    return image


def _blank_image() -> Image.Image:
    return Image.new("RGB", (200, 100), color="white")


def _different_text_image() -> Image.Image:
    """Deliberately far from `_text_image()` in both color and content —
    solid-color images are a poor dedupe fixture since phash is driven by
    texture/frequency, not hue (verified: phash(solid red) == phash(solid
    blue), Hamming distance 0)."""
    image = Image.new("RGB", (200, 100), color="black")
    draw = ImageDraw.Draw(image)
    draw.text((10, 10), "XYZ 99:99 Completely Different Layout", fill="white")
    draw.rectangle([0, 60, 200, 100], fill="red")
    return image


def _candidate(keyframe_id: str, image: Image.Image, timestamp: float = 1.0) -> OcrCandidate:
    return OcrCandidate(
        keyframe_id=keyframe_id, video_id="video1", timestamp=timestamp, image=image
    )


class _FakeOcrClient:
    def __init__(self, fail_first_n: int = 0):
        self.calls = 0
        self._fail_first_n = fail_first_n

    def extract_text(self, image: Image.Image) -> str:
        self.calls += 1
        if self.calls <= self._fail_first_n:
            raise RuntimeError("simulated OCR failure")
        return "recognized text"


class _FakeEsStore:
    def __init__(self):
        self.bulk_index_calls: list[list] = []

    async def bulk_index(self, docs):
        self.bulk_index_calls.append(list(docs))


def test_has_likely_text_true_for_image_with_text():
    assert has_likely_text(_text_image()) is True


def test_has_likely_text_false_for_blank_image():
    assert has_likely_text(_blank_image()) is False


def test_deduplicate_by_phash_collapses_near_identical_frames():
    candidates = [
        _candidate("k1", _text_image()),
        _candidate("k2", _text_image()),  # near-identical to k1
        _candidate("k3", _different_text_image()),  # clearly different
    ]

    kept = deduplicate_by_phash(candidates)

    assert [c.keyframe_id for c in kept] == ["k1", "k3"]


async def test_ocr_skips_frames_without_text():
    candidates = [_candidate("k1", _text_image()), _candidate("k2", _blank_image())]
    client = _FakeOcrClient()
    es = _FakeEsStore()

    summary = await run_ocr(candidates, ocr_client=client, es=es, token_budget=10_000)

    assert summary.after_text_filter == 1
    assert client.calls == 1


async def test_ocr_deduplicates_near_identical_frames_before_api_call():
    candidates = [
        _candidate("k1", _text_image()),
        _candidate("k2", _text_image()),  # same rendered text -> near-identical
        _candidate("k3", _blank_image()),  # filtered out at step [1] anyway
    ]
    client = _FakeOcrClient()
    es = _FakeEsStore()

    summary = await run_ocr(candidates, ocr_client=client, es=es, token_budget=10_000)

    assert summary.after_dedupe == 1
    assert client.calls == 1


async def test_ocr_failure_does_not_abort_run():
    candidates = [
        _candidate("k1", _text_image()),
        _candidate("k2", _different_text_image()),
    ]
    client = _FakeOcrClient(fail_first_n=1)
    es = _FakeEsStore()

    summary = await run_ocr(candidates, ocr_client=client, es=es, token_budget=10_000)

    assert summary.failed_keyframe_ids == ["k1"]
    assert summary.gemini_calls == 1
    assert len(es.bulk_index_calls[0]) == 1


async def test_ocr_stops_when_token_budget_exceeded():
    candidates = [
        _candidate("k1", _text_image()),
        _candidate("k2", _different_text_image()),
    ]
    client = _FakeOcrClient()
    es = _FakeEsStore()

    summary = await run_ocr(candidates, ocr_client=client, es=es, token_budget=258)

    assert summary.stopped_on_budget is True
    assert summary.gemini_calls == 1
    assert client.calls == 1


async def test_run_ocr_writes_recognized_text_to_es():
    candidates = [_candidate("k1", _text_image(), timestamp=3.5)]
    es = _FakeEsStore()

    await run_ocr(candidates, ocr_client=_FakeOcrClient(), es=es, token_budget=10_000)

    doc = es.bulk_index_calls[0][0]
    assert doc.keyframe_id == "k1"
    assert doc.video_id == "video1"
    assert doc.timestamp == 3.5
    assert doc.ocr_text == "recognized text"


async def test_run_ocr_with_no_text_candidates_writes_nothing():
    candidates = [_candidate("k1", _blank_image())]
    es = _FakeEsStore()

    summary = await run_ocr(candidates, ocr_client=_FakeOcrClient(), es=es, token_budget=10_000)

    assert summary.gemini_calls == 0
    assert es.bulk_index_calls == [[]]


def test_has_likely_text_respects_custom_edge_density_threshold():
    # An impossibly high threshold makes even real text fail the filter —
    # proves the value comes from config, not a hardcoded constant.
    strict_config = OcrConfig(edge_density_threshold=0.99)
    assert has_likely_text(_text_image(), strict_config) is False


async def test_run_ocr_respects_custom_hamming_dedupe_threshold():
    # threshold=0 (only exact-hash matches dedupe) keeps both near-identical
    # frames instead of collapsing them — proves the value comes from config.
    candidates = [_candidate("k1", _text_image()), _candidate("k2", _text_image())]
    permissive_config = OcrConfig(hamming_dedupe_threshold=-1)

    summary = await run_ocr(
        candidates,
        ocr_client=_FakeOcrClient(),
        es=_FakeEsStore(),
        token_budget=10_000,
        config=permissive_config,
    )

    assert summary.after_dedupe == 2
