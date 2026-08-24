import pytest
from elasticsearch import ApiError
from elasticsearch import ConnectionError as EsConnectionError
from elasticsearch import ConnectionTimeout as EsConnectionTimeout

from app.services.asr_retrieval_service import (
    AsrRetrievalConfig,
    AsrRetrievalService,
    AsrRetrievalUnavailable,
    asr_query_detected,
    normalize_asr_text,
)


def row(
    segment_id: str,
    video_id: str,
    text: str,
    start_time: float = 0.0,
    end_time: float = 1.0,
) -> dict:
    return {
        "segment_id": segment_id,
        "video_id": video_id,
        "text": text,
        "start_time": start_time,
        "end_time": end_time,
    }


class FakeAsrStore:
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows
        self.scroll_all_calls = 0

    async def scroll_all(self):
        self.scroll_all_calls += 1
        for item in self.rows:
            yield item


class FailingAsrStore:
    def __init__(self, error: Exception) -> None:
        self._error = error

    async def scroll_all(self):
        raise self._error
        yield


def test_normalize_asr_text_keeps_vietnamese_diacritics_and_removes_punctuation():
    assert normalize_asr_text("  Giá Vàng,   Hôm Nay! ") == "giá vàng hôm nay"
    assert normalize_asr_text("ca cá bán ban") == "ca cá bán ban"


@pytest.mark.parametrize(
    ("query", "transcript", "detected"),
    [
        ("cá", "người đàn ông cầm một con cá", True),
        ("cá", "một chiếc camera", False),
        ("giá vàng", "hôm nay giá vàng tiếp tục tăng", True),
        ("CON CÁ", "người đàn ông đang cầm con cá", True),
        ("một con cá lớn", "một con cá rất lớn được đặt lên cân", True),
        ("", "người đàn ông cầm một con cá", False),
    ],
)
def test_asr_query_detected_uses_boolean_lexical_matching(query, transcript, detected):
    assert asr_query_detected(query, transcript) is detected


async def test_detects_videos_and_preserves_matching_timestamps_without_bm25_scores():
    store = FakeAsrStore(
        [
            row("s1", "V2", "không liên quan", 0.0, 1.0),
            row("s2", "V1", "người đàn ông đang cầm con cá", 34.2, 37.8),
            row("s3", "V1", "ông đặt con cá lên cân", 40.0, 42.0),
            row("s4", "V3", "một chiếc camera", 5.0, 6.0),
        ]
    )
    service = AsrRetrievalService(store, AsrRetrievalConfig(evidence_per_video=2))

    result = await service.retrieve("con cá", video_top_k=20)

    assert result.query == "con cá"
    assert result.detected is True
    assert result.asr_detected_any is True
    assert result.candidate_video_count == 1
    assert result.scanned_segment_count == 4
    assert result.matched_segment_count == 2
    assert result.video_ids == ["V1"]
    assert result.asr_matches[0].detected is True
    assert [segment.segment_id for segment in result.videos[0].segments] == ["s2", "s3"]
    assert result.videos[0].segments[0].start_time == 34.2
    assert store.scroll_all_calls == 1


async def test_multiple_asr_queries_support_any_and_all_modes():
    store = FakeAsrStore(
        [
            row("s1", "V1", "con cá được đặt lên cân"),
            row("s2", "V2", "con cá bơi trong ao"),
        ]
    )
    service = AsrRetrievalService(store, AsrRetrievalConfig())

    any_result = await service.retrieve(["con cá", "cân"], video_top_k=20, match_mode="any")
    all_result = await service.retrieve(["con cá", "cân"], video_top_k=20, match_mode="all")

    assert any_result.video_ids == ["V1", "V2"]
    assert all_result.video_ids == ["V1"]
    assert {match.query: match.detected for match in all_result.asr_matches} == {
        "con cá": True,
        "cân": True,
    }


async def test_zero_detected_segments_returns_an_active_empty_gate():
    service = AsrRetrievalService(
        FakeAsrStore([row("s1", "V1", "một chiếc camera")]),
        AsrRetrievalConfig(),
    )

    result = await service.retrieve("cá", video_top_k=20)

    assert result.detected is False
    assert result.candidate_video_count == 0
    assert result.videos == []
    assert result.video_ids == []
    assert result.asr_matches[0].detected is False


async def test_blank_query_is_rejected_before_transcript_scan():
    store = FakeAsrStore([])
    service = AsrRetrievalService(store, AsrRetrievalConfig())

    with pytest.raises(ValueError, match="ASR query must not be blank"):
        await service.retrieve("  ", video_top_k=20)

    assert store.scroll_all_calls == 0


@pytest.mark.parametrize(
    "error",
    [
        EsConnectionError("offline"),
        EsConnectionTimeout("timed out"),
        ApiError("503", {}, {}),
    ],
)
async def test_elasticsearch_failures_become_asr_unavailable(error):
    service = AsrRetrievalService(FailingAsrStore(error), AsrRetrievalConfig())

    with pytest.raises(AsrRetrievalUnavailable, match="ASR transcript detection is unavailable"):
        await service.retrieve("query", video_top_k=20)


async def test_non_elasticsearch_failures_propagate_unchanged():
    service = AsrRetrievalService(FailingAsrStore(RuntimeError("unexpected")), AsrRetrievalConfig())

    with pytest.raises(RuntimeError, match="unexpected"):
        await service.retrieve("query", video_top_k=20)


def test_config_reads_the_operational_asr_mapping_values():
    config = AsrRetrievalConfig.from_mapping(
        {
            "asr_evidence_per_video": 2,
            "asr_token_coverage_threshold": 0.75,
            "asr_match_mode": "all",
        }
    )

    assert config == AsrRetrievalConfig(
        evidence_per_video=2,
        token_coverage_threshold=0.75,
        default_match_mode="all",
    )
