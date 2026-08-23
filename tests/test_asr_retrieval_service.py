import pytest
from elasticsearch import ApiError
from elasticsearch import ConnectionError as EsConnectionError

from app.services.asr_retrieval_service import (
    AsrRetrievalConfig,
    AsrRetrievalService,
    AsrRetrievalUnavailable,
)
from app.stores.es_client import AsrSegmentSearchHit, AsrSegmentSearchResult


def hit(
    segment_id: str,
    video_id: str,
    score: float,
    start_time: float,
    end_time: float,
) -> AsrSegmentSearchHit:
    return AsrSegmentSearchHit(
        segment_id=segment_id,
        video_id=video_id,
        text=f"text for {segment_id}",
        start_time=start_time,
        end_time=end_time,
        score=score,
    )


class FakeAsrStore:
    def __init__(self, total_hits: int, hits: list[AsrSegmentSearchHit]) -> None:
        self.result = AsrSegmentSearchResult(total_hits=total_hits, hits=hits)
        self.search_calls: list[tuple[str, int]] = []

    async def search(self, query_text: str, top_k: int) -> AsrSegmentSearchResult:
        self.search_calls.append((query_text, top_k))
        return self.result


class FailingAsrStore:
    def __init__(self, error: Exception) -> None:
        self._error = error

    async def search(self, query_text: str, top_k: int) -> AsrSegmentSearchResult:
        raise self._error


async def test_groups_segments_into_deterministically_ranked_videos():
    store = FakeAsrStore(
        total_hits=8,
        hits=[
            hit("s1", "V2", 9.0, 20.0, 24.0),
            hit("s2", "V1", 9.0, 10.0, 14.0),
            hit("s1", "V2", 9.0, 20.0, 24.0),
            hit("s3", "V1", 7.0, 30.0, 34.0),
            hit("s4", "V3", 9.0, 40.0, 44.0),
        ],
    )
    service = AsrRetrievalService(
        store,
        AsrRetrievalConfig(segment_retrieval_k=200, evidence_per_video=1),
    )

    result = await service.retrieve("  thuy loi  ", video_top_k=2)

    assert result.query == "thuy loi"
    assert result.segment_hit_count == 8
    assert [video.video_id for video in result.videos] == ["V2", "V1"]
    assert [video.video_rank for video in result.videos] == [1, 2]
    assert [segment.segment_id for segment in result.videos[0].segments] == ["s1"]
    assert store.search_calls == [("thuy loi", 200)]


async def test_evidence_orders_by_score_then_original_segment_rank():
    service = AsrRetrievalService(
        FakeAsrStore(
            total_hits=3,
            hits=[
                hit("first", "V1", 4.0, 0.0, 1.0),
                hit("second", "V1", 7.0, 2.0, 3.0),
                hit("third", "V1", 7.0, 4.0, 5.0),
            ],
        ),
        AsrRetrievalConfig(segment_retrieval_k=200, evidence_per_video=2),
    )

    result = await service.retrieve("query", video_top_k=1)

    assert [segment.segment_id for segment in result.videos[0].segments] == ["second", "third"]


async def test_zero_segment_hits_returns_an_active_empty_gate():
    service = AsrRetrievalService(
        FakeAsrStore(total_hits=0, hits=[]),
        AsrRetrievalConfig(segment_retrieval_k=200, evidence_per_video=3),
    )

    result = await service.retrieve("dap ho chua", video_top_k=20)

    assert result.candidate_video_count == 0
    assert result.videos == []
    assert result.video_ids == []


async def test_blank_query_is_rejected_before_transcript_search():
    store = FakeAsrStore(total_hits=0, hits=[])
    service = AsrRetrievalService(
        store,
        AsrRetrievalConfig(segment_retrieval_k=200, evidence_per_video=3),
    )

    with pytest.raises(ValueError, match="ASR query must not be blank"):
        await service.retrieve("  ", video_top_k=20)

    assert store.search_calls == []


@pytest.mark.parametrize("error", [EsConnectionError("offline"), ApiError("503", {}, {})])
async def test_elasticsearch_failures_become_asr_unavailable(error):
    service = AsrRetrievalService(
        FailingAsrStore(error),
        AsrRetrievalConfig(segment_retrieval_k=200, evidence_per_video=3),
    )

    with pytest.raises(AsrRetrievalUnavailable, match="ASR transcript search is unavailable"):
        await service.retrieve("query", video_top_k=20)


async def test_non_elasticsearch_failures_propagate_unchanged():
    service = AsrRetrievalService(
        FailingAsrStore(RuntimeError("unexpected")),
        AsrRetrievalConfig(segment_retrieval_k=200, evidence_per_video=3),
    )

    with pytest.raises(RuntimeError, match="unexpected"):
        await service.retrieve("query", video_top_k=20)


def test_config_reads_the_operational_asr_mapping_values():
    config = AsrRetrievalConfig.from_mapping(
        {"asr_segment_retrieval_k": 200, "asr_evidence_per_video": 3}
    )

    assert config == AsrRetrievalConfig(segment_retrieval_k=200, evidence_per_video=3)
