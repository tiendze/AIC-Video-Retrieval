"""TrakeService tests -- the candidate-chaining pipeline behind POST /api/trake.

The service no longer runs a corpus-wide DP. It asks `SearchService` for each
event's KIS candidates (Text / OCR / Title / ASR, per event), keeps the union of
videos those pools touch, backfills each event's bucket locally from the in-RAM
embedding cache, then chains and reranks.

The tests below pin the behaviours that would otherwise fail silently: the union
(not intersection) over events, the local backfill, `frame_idx` provenance, and
loud failure when an operator's OCR/ASR filter cannot be applied.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from app.services.search_service import KisCandidatePool, SearchHit, SearchServiceError
from app.services.trake_embedding_cache import (
    TrakeCachedKeyframe,
    TrakeEmbeddingCacheStats,
    TrakeVideoSlice,
)
from app.services.trake_service import (
    TrakeEventQuery,
    TrakeService,
    TrakeServiceError,
    load_trake_config,
)

_DIM = 4


def _unit(hot: int, dim: int = _DIM) -> np.ndarray:
    vector = np.zeros(dim, dtype=np.float32)
    vector[hot] = 1.0
    return vector


class _FakeEncoder:
    """Maps a query string onto a basis vector, so a keyframe embedded on the
    same basis vector is its perfect match."""

    embedding_dim = _DIM

    def __init__(self, vector_by_text: dict[str, np.ndarray]) -> None:
        self._vector_by_text = vector_by_text
        self.encode_text_calls: list[list[str]] = []

    @property
    def model_id(self) -> str:
        return "fake-encoder"

    def encode_text(self, texts):
        self.encode_text_calls.append(list(texts))
        return np.stack(
            [self._vector_by_text.get(text, np.zeros(_DIM, dtype=np.float32)) for text in texts]
        ).astype(np.float32)

    def encode_image(self, images):  # pragma: no cover - refinement is off here
        raise AssertionError("encode_image must not be called when refinement is disabled")


class _FakeTrakeEmbeddingCache:
    def __init__(self, videos: dict[str, list[dict[str, Any]]]) -> None:
        self.load_count = 1
        self.ensure_ready_calls = 0
        embeddings: list[np.ndarray] = []
        metadata: list[TrakeCachedKeyframe] = []
        timestamps: list[float] = []
        slices: list[TrakeVideoSlice] = []
        for video_id, rows in videos.items():
            start = len(metadata)
            for row in rows:
                embeddings.append(np.asarray(row["embedding"], dtype=np.float32))
                timestamps.append(float(row["timestamp"]))
                metadata.append(
                    TrakeCachedKeyframe(
                        keyframe_id=row["keyframe_id"],
                        video_id=video_id,
                        global_index=row["global_index"],
                        frame_idx=row["frame_idx"],
                        timestamp=float(row["timestamp"]),
                    )
                )
            slices.append(
                TrakeVideoSlice(
                    video_id=video_id,
                    s_v=rows[0]["global_index"],
                    e_v=rows[-1]["global_index"],
                    start=start,
                    stop=len(metadata),
                )
            )
        self._embeddings = np.stack(embeddings).astype(np.float32)
        self._timestamps = np.asarray(timestamps, dtype=np.float64)
        self._metadata = tuple(metadata)
        self._video_slices = tuple(slices)

    def ensure_ready(self) -> None:
        self.ensure_ready_calls += 1

    @property
    def embeddings(self) -> np.ndarray:
        return self._embeddings

    @property
    def timestamps(self) -> np.ndarray:
        return self._timestamps

    @property
    def metadata(self) -> tuple[TrakeCachedKeyframe, ...]:
        return self._metadata

    @property
    def video_slices(self) -> tuple[TrakeVideoSlice, ...]:
        return self._video_slices

    @property
    def stats(self) -> TrakeEmbeddingCacheStats:
        return TrakeEmbeddingCacheStats(
            num_keyframes=self._embeddings.shape[0],
            num_videos=len(self._video_slices),
            embedding_dim=_DIM,
            dtype="float32",
            embedding_memory_bytes=int(self._embeddings.nbytes),
            load_time_ms=0.0,
            source="test-cache",
        )


class _FakeSearchService:
    """Returns a scripted `KisCandidatePool` per event index and records the
    exact kwargs it was called with -- that is how the per-event OCR / Title /
    ASR boxes are verified to reach Stage 1."""

    def __init__(
        self,
        pools_by_event: list[list[SearchHit]] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._pools_by_event = pools_by_event or []
        self._error = error
        self.calls: list[dict[str, Any]] = []

    async def retrieve_kis_candidates(self, **kwargs):
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        index = len(self.calls) - 1
        hits = self._pools_by_event[index] if index < len(self._pools_by_event) else []
        return KisCandidatePool(
            candidate_ids=[hit.keyframe_id for hit in hits],
            text_candidate_count=len(hits),
            ocr_candidate_count=0,
            results=list(hits),
        )


def _hit(keyframe_id: str, video_id: str, frame_idx: int, timestamp: float) -> SearchHit:
    return SearchHit(
        keyframe_id=keyframe_id,
        video_id=video_id,
        score=1.0,
        frame_idx=frame_idx,
        timestamp=timestamp,
        thumbnail_path="",
    )


def _row(keyframe_id: str, global_index: int, frame_idx: int, timestamp: float, hot: int):
    return {
        "keyframe_id": keyframe_id,
        "global_index": global_index,
        "frame_idx": frame_idx,
        "timestamp": timestamp,
        "embedding": _unit(hot),
    }


# V1 is the intended answer: keyframe A matches event 0, keyframe C matches
# event 1, and C is 20s after A so it clears min_gap without hitting max_gap.
# V2 matches neither event on any keyframe.
def _corpus() -> dict[str, list[dict[str, Any]]]:
    return {
        "L21_V001": [
            _row("V1_a", 0, 250, 10.0, hot=0),
            _row("V1_b", 1, 375, 15.0, hot=3),
            _row("V1_c", 2, 750, 30.0, hot=1),
            _row("V1_d", 3, 1000, 40.0, hot=3),
        ],
        "L21_V002": [
            _row("V2_a", 4, 100, 4.0, hot=2),
            _row("V2_b", 5, 500, 20.0, hot=2),
        ],
    }


def _service(
    search_service: _FakeSearchService,
    encoder: _FakeEncoder | None = None,
    config_overrides: dict[str, Any] | None = None,
    cache: _FakeTrakeEmbeddingCache | None = None,
) -> TrakeService:
    config = load_trake_config()
    config.update(config_overrides or {})
    return TrakeService(
        encoder=encoder or _FakeEncoder({"event one": _unit(0), "event two": _unit(1)}),
        trake_cache=cache or _FakeTrakeEmbeddingCache(_corpus()),
        search_service=search_service,  # type: ignore[arg-type]
        query_translator=None,
        config=config,
    )


def _events() -> list[TrakeEventQuery]:
    return [
        TrakeEventQuery(text_queries=("event one",)),
        TrakeEventQuery(text_queries=("event two",)),
    ]


class TestHappyPath:
    async def test_returns_a_time_ordered_sequence_from_the_matching_video(self) -> None:
        search = _FakeSearchService(
            [
                [_hit("V1_a", "L21_V001", 250, 10.0)],
                [_hit("V1_c", "L21_V001", 750, 30.0)],
            ]
        )
        result = await _service(search).search(_events(), top_k=10)

        # Only the video KIS touched becomes a candidate; it contributes
        # several sequence variants, capped by max_sequences_per_video.
        assert {candidate.video_id for candidate in result.candidates} == {"L21_V001"}
        best = result.candidates[0]
        assert best.frames[0].keyframe_id == "V1_a"
        assert best.frames[1].keyframe_id == "V1_c"
        assert best.timestamps[0] < best.timestamps[1]

    async def test_frame_idx_comes_from_the_cache_not_from_timestamp_times_fps(self) -> None:
        # 10.0s and 30.0s would be frames 250/750 at 25fps and 300/900 at
        # 30fps. The submitted value must be the stored one either way
        # (CLAUDE.md §3) -- this is the invariant a wrong conversion breaks.
        search = _FakeSearchService(
            [
                [_hit("V1_a", "L21_V001", 250, 10.0)],
                [_hit("V1_c", "L21_V001", 750, 30.0)],
            ]
        )
        result = await _service(search).search(_events(), top_k=10)

        assert result.candidates[0].frame_indices == (250, 750)

    async def test_reports_per_event_diagnostics(self) -> None:
        search = _FakeSearchService(
            [
                [_hit("V1_a", "L21_V001", 250, 10.0)],
                [
                    _hit("V1_c", "L21_V001", 750, 30.0),
                    _hit("V2_b", "L21_V002", 500, 20.0),
                ],
            ]
        )
        result = await _service(search).search(_events(), top_k=10)

        assert [event.kis_candidate_count for event in result.events] == [1, 2]
        assert [event.kis_video_count for event in result.events] == [1, 2]
        assert all(event.has_text for event in result.events)


class TestSequencesByVideo:
    async def test_details_view_lists_a_videos_own_sequences(self) -> None:
        search = _FakeSearchService(
            [
                [_hit("V1_a", "L21_V001", 250, 10.0)],
                [_hit("V1_c", "L21_V001", 750, 30.0)],
            ]
        )
        result = await _service(search).search(_events(), top_k=10)

        assert "L21_V001" in result.sequences_by_video
        sequences = result.sequences_by_video["L21_V001"]
        assert sequences  # non-empty
        assert all(sequence.video_id == "L21_V001" for sequence in sequences)
        # Details are sorted best-first, independent of the submission order.
        scores = [sequence.score for sequence in sequences]
        assert scores == sorted(scores, reverse=True)

    async def test_details_view_is_capped_by_details_top_k(self) -> None:
        search = _FakeSearchService(
            [
                [_hit("V1_a", "L21_V001", 250, 10.0)],
                [_hit("V1_c", "L21_V001", 750, 30.0)],
            ]
        )
        service = _service(search, config_overrides={"details_top_k": 1})
        result = await service.search(_events(), top_k=10)

        assert len(result.sequences_by_video["L21_V001"]) == 1


class TestCandidateVideoSelection:
    async def test_a_video_kis_only_found_for_one_event_is_still_chained(self) -> None:
        # The whole point of the union: event 2's KIS pool never mentions
        # L21_V001, but the video is still a candidate and its bucket for
        # event 2 is backfilled locally. Under an intersection rule this
        # correct video would be dropped.
        search = _FakeSearchService(
            [
                [_hit("V1_a", "L21_V001", 250, 10.0)],
                [_hit("V2_b", "L21_V002", 500, 20.0)],
            ]
        )
        result = await _service(search).search(_events(), top_k=10)

        chained = {candidate.video_id for candidate in result.candidates}
        assert "L21_V001" in chained

    async def test_min_event_coverage_can_tighten_the_candidate_set(self) -> None:
        search = _FakeSearchService(
            [
                [_hit("V1_a", "L21_V001", 250, 10.0)],
                [_hit("V2_b", "L21_V002", 500, 20.0)],
            ]
        )
        service = _service(search, config_overrides={"min_event_coverage": 2})
        result = await service.search(_events(), top_k=10)

        assert result.candidate_video_count == 0
        assert result.candidates == ()

    async def test_a_kis_hit_outside_the_trake_cache_is_skipped(self) -> None:
        # Videos that are not `ingest_status == complete`, or whose
        # global_index range has a hole, never enter the cache. A KIS hit on
        # one of them cannot be chained.
        search = _FakeSearchService(
            [
                [_hit("GHOST_a", "L99_V999", 10, 1.0)],
                [_hit("V1_c", "L21_V001", 750, 30.0)],
            ]
        )
        result = await _service(search).search(_events(), top_k=10)

        assert all(candidate.video_id != "L99_V999" for candidate in result.candidates)


class TestPerEventChannels:
    async def test_each_event_gets_its_own_ocr_title_and_asr_boxes(self) -> None:
        search = _FakeSearchService([[], []])
        events = [
            TrakeEventQuery(
                text_queries=("event one",),
                ocr_query="ty so 0-0",
                title_query="chung ket",
                asr_queries=("mo man tran dau",),
                asr_match_mode="all",
            ),
            TrakeEventQuery(text_queries=("event two",), ocr_query="ty so 1-0"),
        ]
        await _service(search).search(events, top_k=10)

        assert len(search.calls) == 2
        first, second = search.calls
        assert first["ocr_query"] == "ty so 0-0"
        assert first["title_query"] == "chung ket"
        assert first["asr_queries"] == ["mo man tran dau"]
        assert first["asr_match_mode"] == "all"
        assert second["ocr_query"] == "ty so 1-0"
        assert second["title_query"] is None
        assert second["asr_queries"] == []

    async def test_several_text_boxes_on_one_event_are_prompt_ensembled(self) -> None:
        encoder = _FakeEncoder(
            {"phrasing a": _unit(0), "phrasing b": _unit(0), "event two": _unit(1)}
        )
        search = _FakeSearchService([[], []])
        events = [
            TrakeEventQuery(text_queries=("phrasing a", "phrasing b")),
            TrakeEventQuery(text_queries=("event two",)),
        ]
        await _service(search, encoder=encoder).search(events, top_k=10)

        # One encode call per event, carrying that event's boxes together --
        # they are averaged into a single query vector, not searched separately.
        assert encoder.encode_text_calls == [["phrasing a", "phrasing b"], ["event two"]]

    async def test_an_event_with_no_text_falls_back_to_its_kis_hits(self) -> None:
        search = _FakeSearchService(
            [
                [_hit("V1_a", "L21_V001", 250, 10.0)],
                [_hit("V1_c", "L21_V001", 750, 30.0)],
            ]
        )
        events = [
            TrakeEventQuery(text_queries=("event one",)),
            TrakeEventQuery(ocr_query="ty so 1-0"),
        ]
        result = await _service(search).search(events, top_k=10)

        assert result.candidates
        assert result.candidates[0].frames[1].keyframe_id == "V1_c"
        assert result.events[1].has_text is False

    async def test_a_text_less_event_with_no_kis_hit_drops_the_video_loudly(self) -> None:
        # No text means no semantic axis to backfill with, so this event has no
        # candidate frame at all in that video. The video is dropped and
        # counted, never returned with a shorter sequence.
        search = _FakeSearchService(
            [
                [_hit("V1_a", "L21_V001", 250, 10.0)],
                [],
            ]
        )
        events = [
            TrakeEventQuery(text_queries=("event one",)),
            TrakeEventQuery(ocr_query="never matches"),
        ]
        result = await _service(search).search(events, top_k=10)

        assert result.candidates == ()
        assert result.dropped_video_count == 1


class TestGapConstraints:
    async def test_min_gap_stops_two_events_sharing_one_shot(self) -> None:
        corpus = {
            "L21_V001": [
                _row("V1_a", 0, 250, 10.0, hot=0),
                _row("V1_b", 1, 260, 10.4, hot=1),
            ]
        }
        search = _FakeSearchService([[], []])
        service = _service(
            search,
            cache=_FakeTrakeEmbeddingCache(corpus),
            config_overrides={"min_event_coverage": 0, "bucket_nms_seconds": 0.1},
        )
        # Nothing was retrieved by KIS, so there are no candidate videos at all.
        result = await service.search(_events(), top_k=10)
        assert result.candidates == ()

    async def test_a_video_with_no_ordered_path_is_dropped_and_counted(self) -> None:
        # Event 1's only match sits BEFORE event 0's, so no forward-ordered
        # sequence exists in this video.
        corpus = {
            "L21_V001": [
                _row("V1_c", 0, 250, 10.0, hot=1),
                _row("V1_a", 1, 750, 30.0, hot=0),
            ]
        }
        search = _FakeSearchService(
            [
                [_hit("V1_a", "L21_V001", 750, 30.0)],
                [_hit("V1_c", "L21_V001", 250, 10.0)],
            ]
        )
        service = _service(
            search,
            cache=_FakeTrakeEmbeddingCache(corpus),
            config_overrides={"bucket_top_k": 1, "bucket_nms_seconds": 0.0},
        )
        result = await service.search(_events(), top_k=10)

        assert result.candidates == ()
        assert result.dropped_video_count == 1


class TestFailureModes:
    async def test_an_unavailable_ocr_or_asr_backend_fails_loudly(self) -> None:
        # Running the query anyway, minus the filter the operator asked for,
        # would look like success while returning a different result set.
        search = _FakeSearchService(
            error=SearchServiceError(
                "Elasticsearch unavailable", error_code="ocr_unavailable", status_code=503
            )
        )
        with pytest.raises(TrakeServiceError) as excinfo:
            await _service(search).search(_events(), top_k=10)

        assert excinfo.value.error_code == "ocr_unavailable"
        assert excinfo.value.status_code == 503
        assert "Event 1" in str(excinfo.value)

    async def test_an_event_with_no_channel_at_all_is_rejected(self) -> None:
        search = _FakeSearchService([[], []])
        events = [TrakeEventQuery(text_queries=("event one",)), TrakeEventQuery()]
        with pytest.raises(TrakeServiceError) as excinfo:
            await _service(search).search(events, top_k=10)

        assert excinfo.value.error_code == "empty_event"

    async def test_an_empty_event_list_is_rejected(self) -> None:
        with pytest.raises(TrakeServiceError) as excinfo:
            await _service(_FakeSearchService()).search([], top_k=10)

        assert excinfo.value.error_code == "no_events"


class TestSubmissionShape:
    async def test_top_k_bounds_the_returned_sequence_count(self) -> None:
        search = _FakeSearchService(
            [
                [_hit("V1_a", "L21_V001", 250, 10.0)],
                [_hit("V1_c", "L21_V001", 750, 30.0)],
            ]
        )
        service = _service(search, config_overrides={"bucket_nms_seconds": 0.0})
        result = await service.search(_events(), top_k=1)

        assert len(result.candidates) <= 1

    async def test_every_returned_sequence_has_one_frame_per_event(self) -> None:
        search = _FakeSearchService(
            [
                [_hit("V1_a", "L21_V001", 250, 10.0)],
                [_hit("V1_c", "L21_V001", 750, 30.0)],
            ]
        )
        result = await _service(search).search(_events(), top_k=10)

        for candidate in result.candidates:
            assert len(candidate.frames) == 2

    async def test_video_scores_are_reported_for_every_returned_video(self) -> None:
        search = _FakeSearchService(
            [
                [_hit("V1_a", "L21_V001", 250, 10.0)],
                [_hit("V1_c", "L21_V001", 750, 30.0)],
            ]
        )
        result = await _service(search).search(_events(), top_k=10)

        for candidate in result.candidates:
            assert candidate.video_id in result.video_scores
