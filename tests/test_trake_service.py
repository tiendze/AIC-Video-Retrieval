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

from app.services.kis_local_retrieval import VideoLocalFiles
from app.services.kis_scoring import KisConfig
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
    _VideoEvidence,
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
    ASR boxes are verified to reach Stage 1.

    `local_embedding_store` / `kis_config` stand in for the properties
    `TrakeService.__init__` reads off the real `SearchService` (plan §T2 step
    1: both services share one store)."""

    def __init__(
        self,
        pools_by_event: list[list[SearchHit]] | None = None,
        error: Exception | None = None,
        local_embedding_store: _FakeLocalEmbeddingStore | None = None,
    ) -> None:
        self._pools_by_event = pools_by_event or []
        self._error = error
        self.calls: list[dict[str, Any]] = []
        self.local_embedding_store = local_embedding_store or _FakeLocalEmbeddingStore()
        self.kis_config = KisConfig()

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


class _FakeLocalEmbeddingStore:
    """Stands in for `KISLocalEmbeddingStore.load_video` without touching
    disk -- maps video_id -> VideoLocalFiles, or None for a video with no
    dense features (plan §T2-4: the 20%-of-corpus case)."""

    def __init__(self, files_by_video: dict[str, VideoLocalFiles] | None = None) -> None:
        self._files_by_video = files_by_video or {}

    def load_video(self, video_id: str) -> VideoLocalFiles | None:
        return self._files_by_video.get(video_id)


def _video_local_files(patches_by_frame_idx: dict[int, np.ndarray]) -> VideoLocalFiles:
    frame_indices = sorted(patches_by_frame_idx)
    embeddings = np.stack([patches_by_frame_idx[idx] for idx in frame_indices]).astype(np.float32)
    return VideoLocalFiles(
        embeddings=embeddings,
        row_by_frame_idx={frame_idx: row for row, frame_idx in enumerate(frame_indices)},
    )


def _patches(hot: int, num_patches: int = 2, dim: int = _DIM) -> np.ndarray:
    """`[num_patches, dim]` dense-feature stand-in: every patch is the same
    basis vector, so MaxSim against a phrase on the same basis is exactly 1.0
    and against an orthogonal one is exactly 0.0 -- easy to reason about."""
    return np.stack([_unit(hot, dim) for _ in range(num_patches)])


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


class _FakeEncoderWithTokenCap(_FakeEncoder):
    """Adds the optional `count_text_tokens`/`max_text_tokens` surface
    `_encode_events` duck-types for truncation detection (plan §T3) --
    `_FakeEncoder` itself deliberately omits these so every other test in
    this file exercises the "encoder can't tell" fallback (text_truncated
    always False)."""

    max_text_tokens = 5

    def count_text_tokens(self, texts: list[str]) -> list[int]:
        return [len(text.split()) for text in texts]


class TestLocalPhrasePassthrough:
    """T1: local_phrases reach KIS's Stage 1 call fanned out per text query
    of the event, and are dropped for a text-less event (plan §T1 -- a
    phrase pins a visual region of a moment described by text; without text
    there is no KIS Stage 1 call for it to attach to)."""

    async def test_local_phrases_fan_out_to_every_text_query_of_the_event(self) -> None:
        search = _FakeSearchService([[], []])
        events = [
            TrakeEventQuery(
                text_queries=("event one", "event one alt"),
                local_phrases=("red packet",),
            ),
            TrakeEventQuery(text_queries=("event two",)),
        ]

        await _service(search).search(events, top_k=10)

        assert search.calls[0]["local_phrases_by_query"] == [["red packet"], ["red packet"]]
        assert search.calls[1]["local_phrases_by_query"] == []

    async def test_local_phrases_are_dropped_when_the_event_has_no_text(self) -> None:
        search = _FakeSearchService([[], []])
        events = [
            TrakeEventQuery(ocr_query="scoreboard", local_phrases=("red packet",)),
            TrakeEventQuery(text_queries=("event two",)),
        ]

        await _service(search).search(events, top_k=10)

        assert search.calls[0]["local_phrases_by_query"] == []


class TestLocalRerankScoring:
    """T2: phrase-to-patch MaxSim rescoring inside the bucket (B2b).

    One video, two candidate keyframes for a single event: F1 matches the
    event's TEXT (global cosine) but not the local phrase; F2 is the reverse.
    A large `local_rerank.weight` should be able to flip which one the chain
    picks -- proving the local score actually reaches frame selection, not
    just the diagnostics fields.
    """

    def _corpus_two_frames(self) -> dict[str, list[dict[str, Any]]]:
        return {
            "L30_V001": [
                _row("F1", 0, 100, 5.0, hot=0),  # matches the event's text
                _row("F2", 1, 200, 6.0, hot=1),  # orthogonal to the event's text
            ],
        }

    def _store(self) -> _FakeLocalEmbeddingStore:
        return _FakeLocalEmbeddingStore(
            {
                "L30_V001": _video_local_files(
                    {
                        100: _patches(hot=1),  # orthogonal to the phrase
                        200: _patches(hot=2),  # matches the phrase
                    }
                )
            }
        )

    async def test_local_phrase_can_flip_the_winning_frame(self) -> None:
        search = _FakeSearchService(
            [[_hit("F1", "L30_V001", 100, 5.0), _hit("F2", "L30_V001", 200, 6.0)]],
            local_embedding_store=self._store(),
        )
        encoder = _FakeEncoder({"event one": _unit(0), "red packet": _unit(2)})
        events = [TrakeEventQuery(text_queries=("event one",), local_phrases=("red packet",))]
        service = _service(
            search,
            encoder=encoder,
            cache=_FakeTrakeEmbeddingCache(self._corpus_two_frames()),
            config_overrides={
                "kis_evidence_weight": 0.0,
                "local_rerank": {"enabled": True, "max_videos": 50, "weight": 10.0},
            },
        )

        result = await service.search(events, top_k=10)

        assert result.candidates[0].frames[0].keyframe_id == "F2"
        assert result.local_rerank_video_count == 1

    async def test_without_a_local_phrase_the_global_winner_is_unchanged(self) -> None:
        search = _FakeSearchService(
            [[_hit("F1", "L30_V001", 100, 5.0), _hit("F2", "L30_V001", 200, 6.0)]],
            local_embedding_store=self._store(),
        )
        encoder = _FakeEncoder({"event one": _unit(0)})
        events = [TrakeEventQuery(text_queries=("event one",))]  # no local_phrases
        service = _service(
            search,
            encoder=encoder,
            cache=_FakeTrakeEmbeddingCache(self._corpus_two_frames()),
            config_overrides={"kis_evidence_weight": 0.0},
        )

        result = await service.search(events, top_k=10)

        assert result.candidates[0].frames[0].keyframe_id == "F1"
        assert result.candidates[0].frames[0].local_score is None
        assert result.candidates[0].frames[0].local_z == 0.0
        assert result.local_rerank_video_count == 0

    async def test_a_video_outside_local_rerank_max_videos_is_unaffected(self) -> None:
        # Same setup that flips the winner above, but max_videos=0 excludes
        # every candidate video from B2b -- must reduce to today's
        # global-cosine-only behaviour exactly (plan §"Nguyên tắc bất biến").
        search = _FakeSearchService(
            [[_hit("F1", "L30_V001", 100, 5.0), _hit("F2", "L30_V001", 200, 6.0)]],
            local_embedding_store=self._store(),
        )
        encoder = _FakeEncoder({"event one": _unit(0), "red packet": _unit(2)})
        events = [TrakeEventQuery(text_queries=("event one",), local_phrases=("red packet",))]
        service = _service(
            search,
            encoder=encoder,
            cache=_FakeTrakeEmbeddingCache(self._corpus_two_frames()),
            config_overrides={
                "kis_evidence_weight": 0.0,
                "local_rerank": {"enabled": True, "max_videos": 0, "weight": 10.0},
            },
        )

        result = await service.search(events, top_k=10)

        assert result.candidates[0].frames[0].keyframe_id == "F1"
        assert result.candidates[0].frames[0].local_score is None
        assert result.candidates[0].frames[0].local_z == 0.0
        assert result.local_rerank_video_count == 0


class TestLocalRerankMissingDense:
    async def test_a_video_with_no_dense_features_still_chains_without_raising(self) -> None:
        corpus = {"L30_V001": [_row("F1", 0, 100, 5.0, hot=0)]}
        search = _FakeSearchService(
            [[_hit("F1", "L30_V001", 100, 5.0)]],
            local_embedding_store=_FakeLocalEmbeddingStore(),  # no entry for L30_V001
        )
        encoder = _FakeEncoder({"event one": _unit(0), "red packet": _unit(2)})
        events = [TrakeEventQuery(text_queries=("event one",), local_phrases=("red packet",))]
        service = _service(
            search,
            encoder=encoder,
            cache=_FakeTrakeEmbeddingCache(corpus),
            config_overrides={"local_rerank": {"enabled": True, "max_videos": 50, "weight": 10.0}},
        )

        result = await service.search(events, top_k=10)

        assert len(result.candidates) == 1
        frame = result.candidates[0].frames[0]
        assert frame.local_score is None
        assert frame.local_z == 0.0
        assert result.local_missing_frame_count == 1
        assert result.local_rerank_video_count == 0


class TestLocalRerankZScoreInvariant:
    def test_local_z_is_mean_zero_within_one_video_and_event(self) -> None:
        corpus = {
            "L30_V001": [
                _row("F0", 0, 10, 1.0, hot=0),
                _row("F1", 1, 20, 2.0, hot=0),
                _row("F2", 2, 30, 3.0, hot=0),
                _row("F3", 3, 40, 4.0, hot=0),
            ],
        }
        cache = _FakeTrakeEmbeddingCache(corpus)
        video_slice = cache.video_slices[0]
        files = _video_local_files(
            {10: _patches(0), 20: _patches(1), 30: _patches(2), 40: _patches(3)}
        )
        service = _service(_FakeSearchService(), encoder=_FakeEncoder({}), cache=cache)
        phrase_vectors_by_event = {0: np.stack([_unit(1)])}

        local_by_event, missing = service._score_local_rerank(
            files, video_slice, phrase_vectors_by_event
        )

        assert missing == 0
        z_values = [score.z for score in local_by_event[0].values()]
        assert len(z_values) == 4
        assert abs(sum(z_values) / len(z_values)) < 1e-9


class TestLocalRerankPreservesOrdering:
    async def test_a_local_phrase_cannot_produce_an_out_of_order_chain(self) -> None:
        # 'd' dominates event0's OWN bucket once the phrase is weighted this
        # heavily, but it is the LAST keyframe in the video -- no frame comes
        # after it for event1 to occupy, so it can never appear in a complete
        # ordered chain no matter how high its local score is (plan §T2-5).
        hits = [
            _hit("V1_a", "L21_V001", 250, 10.0),
            _hit("V1_b", "L21_V001", 375, 15.0),
            _hit("V1_c", "L21_V001", 750, 30.0),
            _hit("V1_d", "L21_V001", 1000, 40.0),
        ]
        store = _FakeLocalEmbeddingStore(
            {
                "L21_V001": _video_local_files(
                    {
                        250: _patches(3),
                        375: _patches(3),
                        750: _patches(3),
                        1000: _patches(2),  # matches the phrase; the rest don't
                    }
                )
            }
        )
        search = _FakeSearchService([hits, hits], local_embedding_store=store)
        encoder = _FakeEncoder(
            {"event one": _unit(0), "event two": _unit(1), "prefer the last frame": _unit(2)}
        )
        events = [
            TrakeEventQuery(text_queries=("event one",), local_phrases=("prefer the last frame",)),
            TrakeEventQuery(text_queries=("event two",)),
        ]
        service = _service(
            search,
            encoder=encoder,
            config_overrides={
                "kis_evidence_weight": 0.0,
                "local_rerank": {"enabled": True, "max_videos": 50, "weight": 50.0},
            },
        )

        result = await service.search(events, top_k=10)

        assert result.candidates  # at least one valid ordered chain survives
        for candidate in result.candidates:
            assert candidate.frames[0].timestamp < candidate.frames[1].timestamp
            assert candidate.frames[0].keyframe_id != "V1_d"


class TestTextTruncationDiagnostics:
    async def test_an_event_over_the_token_cap_is_flagged_and_logged(self, caplog) -> None:
        search = _FakeSearchService([[], []])
        encoder = _FakeEncoderWithTokenCap(
            {"one two three four five six seven": _unit(0), "event two": _unit(1)}
        )
        events = [
            TrakeEventQuery(text_queries=("one two three four five six seven",)),
            TrakeEventQuery(text_queries=("event two",)),
        ]
        service = _service(search, encoder=encoder)

        with caplog.at_level("WARNING"):
            result = await service.search(events, top_k=10)

        assert result.events[0].text_truncated is True
        assert result.events[1].text_truncated is False
        assert any("exceeds" in message for message in caplog.messages)


class TestRankEvidenceNormalization:
    """B8: `_rank_evidence` must land on the same [0,1] scale as `coverage`
    regardless of how many events the query has -- fixed 2026-08-28. The
    ONLY existing fixture (`_events()`) has 2 events, the one N where the
    old unbounded-sum formula happened to be exactly 1.0 at full coverage --
    so every test in this file before this class ran with the bug invisible.
    """

    async def test_video_scores_rank_evidence_never_exceeds_coverages_ceiling(self) -> None:
        # 5 events, one video KIS finds at rank 0 for ALL of them. Before the
        # fix, rank_evidence here would have been 5.0 (unbounded sum over 5
        # rank-0 hits), not 1.0 -- see configs/search/trake.yaml:
        # video_rank_evidence_weight.
        num_events = 5
        corpus = {
            "L21_V001": [
                _row(f"V1_e{i}", i, (i + 1) * 100, (i + 1) * 10.0, hot=0) for i in range(num_events)
            ],
        }
        pools = [
            [_hit(f"V1_e{i}", "L21_V001", (i + 1) * 100, (i + 1) * 10.0)] for i in range(num_events)
        ]
        search = _FakeSearchService(pools)
        events = [TrakeEventQuery(text_queries=(f"event {i}",)) for i in range(num_events)]
        service = _service(search, cache=_FakeTrakeEmbeddingCache(corpus))

        result = await service.search(events, top_k=10)

        assert "L21_V001" in result.video_scores
        assert result.video_scores["L21_V001"].rank_evidence == pytest.approx(1.0)

    def test_rank_evidence_is_normalized_by_total_events_not_covered_events(self) -> None:
        # If this divided by covered-event-count instead of total events, it
        # would come out 1.0 here too -- tying a video KIS found for only 1 of
        # 5 events against one it found for 5 of 5 (both at rank 0). Dividing
        # by num_events keeps that distinction (plan §A).
        service = _service(_FakeSearchService())
        item = _VideoEvidence(
            slice=TrakeVideoSlice(video_id="V1", s_v=0, e_v=0, start=0, stop=1),
            best_rank_by_event={0: 0},
        )

        result = service._rank_evidence({"V1": item}, num_events=5)

        assert result["V1"] == pytest.approx(0.2)
