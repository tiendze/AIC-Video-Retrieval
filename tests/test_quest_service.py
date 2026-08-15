"""RED reproducer for `QuestService` (plan §P2.5, branch 2 -- QUEST handling
Out-of-Knowledge queries via a web sample image, PRD §F5).

Flow: `find_image_candidates(q0)` calls `ddgs` (no async client, runs inside
`asyncio.to_thread`) to fetch image candidates, keeping the real URLs in an
in-memory ID->URL cache (SSRF guard -- the client only ever references a
`candidate_id`, never a URL, docs/references.md §5.5). `search_by_candidate(...)`
looks up the cache, downloads exactly the cached URL (the server fetches
only a URL it itself just returned, never one supplied by the client), then
reuses `SearchService.search_by_image()`.

Both steps require timeout + fallback to `SearchService.search(q0,
mode="semantic")` on error (CLAUDE.md §4) -- no Docker/GPU/real network here:
`ddgs`/`httpx` are both Fakes injected via the constructor.
"""

from __future__ import annotations

import io
from typing import Any

import httpx
import pytest
from ddgs.exceptions import DDGSException, RatelimitException, TimeoutException
from PIL import Image

from app.services.quest_service import (
    ImageCandidatesResult,
    ImageSearchResult,
    QuestService,
    QuestServiceError,
)
from app.services.search_service import SearchHit


def _png_bytes(size: tuple[int, int] = (8, 8), color: tuple[int, int, int] = (10, 20, 30)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color=color).save(buf, format="PNG")
    return buf.getvalue()


def _truncated_jpeg_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color=(10, 20, 30)).save(buf, format="JPEG")
    full = buf.getvalue()
    return full[: len(full) // 2]  # header identifies as JPEG; body decode fails with OSError


_HIT = SearchHit(
    keyframe_id="kf1",
    video_id="video1",
    score=0.9,
    frame_idx=42,
    timestamp=1.4,
    thumbnail_path="/thumb/kf1.jpg",
)

_RAW_CANDIDATES: list[dict[str, Any]] = [
    {
        "title": "Labubu doll",
        "image": "https://example.com/full/labubu1.jpg",
        "thumbnail": "https://example.com/thumb/labubu1.jpg",
        "url": "https://example.com/page1",
        "height": 800,
        "width": 600,
        "source": "Bing",
    },
    {
        "title": "Labubu keychain",
        "image": "https://example.com/full/labubu2.jpg",
        "thumbnail": "https://example.com/thumb/labubu2.jpg",
        "url": "https://example.com/page2",
        "height": 900,
        "width": 700,
        "source": "Bing",
    },
]


class _FakeSearchService:
    def __init__(self, hits: list[SearchHit] | None = None) -> None:
        self._hits = hits if hits is not None else [_HIT]
        self.search_calls: list[tuple[str, str, int]] = []
        self.search_by_image_calls: list[tuple[Any, int]] = []

    async def search(self, query: str, mode: str, top_k: int) -> list[SearchHit]:
        self.search_calls.append((query, mode, top_k))
        return self._hits

    async def search_by_image(self, image: Any, top_k: int) -> list[SearchHit]:
        self.search_by_image_calls.append((image, top_k))
        return self._hits


class _FakeImageSearchClient:
    def __init__(
        self,
        results: list[dict[str, Any]] | None = None,
        error: Exception | None = None,
        sleep_seconds: float = 0.0,
    ) -> None:
        self._results = results if results is not None else list(_RAW_CANDIDATES)
        self._error = error
        self._sleep_seconds = sleep_seconds
        self.images_calls: list[tuple[str, int, str]] = []

    def images(self, query: str, max_results: int, safesearch: str) -> list[dict[str, Any]]:
        self.images_calls.append((query, max_results, safesearch))
        if self._sleep_seconds:
            import time

            time.sleep(self._sleep_seconds)
        if self._error is not None:
            raise self._error
        return self._results


class _FakeHttpResponse:
    def __init__(self, content: bytes, status_error: Exception | None = None) -> None:
        self.content = content
        self._status_error = status_error

    def raise_for_status(self) -> None:
        if self._status_error is not None:
            raise self._status_error


class _FakeHttpClient:
    def __init__(
        self,
        content: bytes | None = None,
        error: Exception | None = None,
        status_error: Exception | None = None,
    ) -> None:
        self._content = content if content is not None else _png_bytes()
        self._error = error
        self._status_error = status_error
        self.get_calls: list[tuple[str, float]] = []

    async def get(self, url: str, timeout: float) -> _FakeHttpResponse:
        self.get_calls.append((url, timeout))
        if self._error is not None:
            raise self._error
        return _FakeHttpResponse(self._content, status_error=self._status_error)


def _service(
    search_service: _FakeSearchService | None = None,
    image_search_client: _FakeImageSearchClient | None = None,
    http_client: _FakeHttpClient | None = None,
    max_candidates: int = 10,
    image_candidates_timeout_sec: float = 8.0,
    image_download_timeout_sec: float = 5.0,
    candidate_cache_ttl_sec: float = 300.0,
    clock=None,
    id_factory=None,
) -> tuple[QuestService, _FakeSearchService, _FakeImageSearchClient, _FakeHttpClient]:
    search_service = search_service or _FakeSearchService()
    image_search_client = image_search_client or _FakeImageSearchClient()
    http_client = http_client or _FakeHttpClient()
    service = QuestService(
        search_service=search_service,
        image_search_client=image_search_client,
        http_client=http_client,
        max_candidates=max_candidates,
        image_candidates_timeout_sec=image_candidates_timeout_sec,
        image_download_timeout_sec=image_download_timeout_sec,
        candidate_cache_ttl_sec=candidate_cache_ttl_sec,
        clock=clock,
        id_factory=id_factory,
    )
    return service, search_service, image_search_client, http_client


class TestFindImageCandidates:
    @pytest.mark.asyncio
    async def test_returns_candidates_on_success_without_calling_fallback_search(self) -> None:
        service, search_service, image_client, _ = _service()

        result = await service.find_image_candidates("Labubu", top_k=20)

        assert isinstance(result, ImageCandidatesResult)
        assert result.status == "candidates"
        assert result.candidate_set_id is not None
        assert result.fallback_hits == []
        assert [c.title for c in result.candidates] == ["Labubu doll", "Labubu keychain"]
        assert [c.thumbnail_url for c in result.candidates] == [
            "https://example.com/thumb/labubu1.jpg",
            "https://example.com/thumb/labubu2.jpg",
        ]
        assert search_service.search_calls == []
        assert image_client.images_calls == [("Labubu", 10, "moderate")]

    @pytest.mark.asyncio
    async def test_passes_configured_max_candidates_to_ddgs(self) -> None:
        service, _, image_client, _ = _service(max_candidates=3)

        await service.find_image_candidates("Labubu", top_k=20)

        assert image_client.images_calls == [("Labubu", 3, "moderate")]

    @pytest.mark.asyncio
    async def test_falls_back_to_semantic_search_when_ddgs_raises_base_exception(self) -> None:
        image_client = _FakeImageSearchClient(error=DDGSException("scraper broke"))
        service, search_service, _, _ = _service(image_search_client=image_client)

        result = await service.find_image_candidates("Labubu", top_k=15)

        assert result.status == "fallback"
        assert result.candidate_set_id is None
        assert result.candidates == []
        assert result.fallback_hits == [_HIT]
        assert search_service.search_calls == [("Labubu", "semantic", 15)]

    @pytest.mark.asyncio
    async def test_falls_back_when_ddgs_raises_ratelimit_exception(self) -> None:
        image_client = _FakeImageSearchClient(error=RatelimitException("rate limited"))
        service, search_service, _, _ = _service(image_search_client=image_client)

        result = await service.find_image_candidates("Labubu", top_k=10)

        assert result.status == "fallback"
        assert search_service.search_calls == [("Labubu", "semantic", 10)]

    @pytest.mark.asyncio
    async def test_falls_back_when_ddgs_raises_timeout_exception(self) -> None:
        image_client = _FakeImageSearchClient(error=TimeoutException("timed out"))
        service, search_service, _, _ = _service(image_search_client=image_client)

        result = await service.find_image_candidates("Labubu", top_k=10)

        assert result.status == "fallback"
        assert search_service.search_calls == [("Labubu", "semantic", 10)]

    @pytest.mark.asyncio
    async def test_falls_back_when_ddgs_call_exceeds_configured_timeout(self) -> None:
        image_client = _FakeImageSearchClient(sleep_seconds=0.1)
        service, search_service, _, _ = _service(
            image_search_client=image_client, image_candidates_timeout_sec=0.01
        )

        result = await service.find_image_candidates("Labubu", top_k=10)

        assert result.status == "fallback"
        assert search_service.search_calls == [("Labubu", "semantic", 10)]

    @pytest.mark.asyncio
    async def test_falls_back_when_ddgs_returns_no_results(self) -> None:
        image_client = _FakeImageSearchClient(results=[])
        service, search_service, _, _ = _service(image_search_client=image_client)

        result = await service.find_image_candidates("Labubu", top_k=10)

        assert result.status == "fallback"
        assert result.candidate_set_id is None
        assert search_service.search_calls == [("Labubu", "semantic", 10)]

    @pytest.mark.asyncio
    async def test_two_calls_get_distinct_candidate_set_ids(self) -> None:
        service, _, _, _ = _service()

        first = await service.find_image_candidates("Labubu", top_k=10)
        second = await service.find_image_candidates("Labubu", top_k=10)

        assert first.candidate_set_id != second.candidate_set_id


class TestSearchByCandidate:
    async def _seeded_service(
        self, **service_kwargs
    ) -> tuple[QuestService, _FakeSearchService, _FakeHttpClient, ImageCandidatesResult]:
        service, search_service, _, http_client = _service(**service_kwargs)
        candidates = await service.find_image_candidates("Labubu", top_k=10)
        return service, search_service, http_client, candidates

    @pytest.mark.asyncio
    async def test_downloads_cached_url_and_searches_by_image(self) -> None:
        service, search_service, http_client, candidates = await self._seeded_service()
        chosen = candidates.candidates[0]

        result = await service.search_by_candidate(
            candidates.candidate_set_id, chosen.candidate_id, top_k=25
        )

        assert isinstance(result, ImageSearchResult)
        assert result.status == "ok"
        assert result.hits == [_HIT]
        assert http_client.get_calls == [("https://example.com/full/labubu1.jpg", 5.0)]
        assert len(search_service.search_by_image_calls) == 1
        image_arg, top_k_arg = search_service.search_by_image_calls[0]
        assert isinstance(image_arg, Image.Image)
        assert top_k_arg == 25

    @pytest.mark.asyncio
    async def test_raises_candidate_set_not_found_for_unknown_set_id(self) -> None:
        service, _, _, _ = _service()

        with pytest.raises(QuestServiceError) as exc_info:
            await service.search_by_candidate("nonexistent-set", "nonexistent-candidate", top_k=10)

        assert exc_info.value.error_code == "candidate_set_not_found"
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_raises_candidate_not_found_for_unknown_candidate_id_in_known_set(self) -> None:
        service, _, _, candidates = await self._seeded_service()

        with pytest.raises(QuestServiceError) as exc_info:
            await service.search_by_candidate(candidates.candidate_set_id, "not-a-real-id", top_k=10)

        assert exc_info.value.error_code == "candidate_not_found"
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_expired_candidate_set_is_treated_as_not_found(self) -> None:
        current_time = [0.0]

        def clock() -> float:
            return current_time[0]

        service, _, _, candidates = await self._seeded_service(
            candidate_cache_ttl_sec=60.0, clock=clock
        )
        chosen = candidates.candidates[0]
        current_time[0] = 61.0  # past the 60s TTL

        with pytest.raises(QuestServiceError) as exc_info:
            await service.search_by_candidate(candidates.candidate_set_id, chosen.candidate_id, top_k=10)

        assert exc_info.value.error_code == "candidate_set_not_found"

    @pytest.mark.asyncio
    async def test_falls_back_to_semantic_search_when_download_raises_http_error(self) -> None:
        http_client = _FakeHttpClient(error=httpx.ConnectError("connection refused"))
        service, search_service, _, candidates = await self._seeded_service(http_client=http_client)
        chosen = candidates.candidates[0]

        result = await service.search_by_candidate(candidates.candidate_set_id, chosen.candidate_id, top_k=12)

        assert result.status == "fallback"
        assert result.hits == [_HIT]
        assert search_service.search_calls == [("Labubu", "semantic", 12)]

    @pytest.mark.asyncio
    async def test_falls_back_when_response_status_is_an_error(self) -> None:
        http_client = _FakeHttpClient(
            status_error=httpx.HTTPStatusError("404", request=None, response=None)
        )
        service, search_service, _, candidates = await self._seeded_service(http_client=http_client)
        chosen = candidates.candidates[0]

        result = await service.search_by_candidate(candidates.candidate_set_id, chosen.candidate_id, top_k=10)

        assert result.status == "fallback"
        assert search_service.search_calls == [("Labubu", "semantic", 10)]

    @pytest.mark.asyncio
    async def test_falls_back_when_downloaded_bytes_are_not_a_valid_image(self) -> None:
        http_client = _FakeHttpClient(content=_truncated_jpeg_bytes())
        service, search_service, _, candidates = await self._seeded_service(http_client=http_client)
        chosen = candidates.candidates[0]

        result = await service.search_by_candidate(candidates.candidate_set_id, chosen.candidate_id, top_k=10)

        assert result.status == "fallback"
        assert search_service.search_calls == [("Labubu", "semantic", 10)]

    @pytest.mark.asyncio
    async def test_downloads_the_full_res_url_not_the_thumbnail_url(self) -> None:
        service, _, http_client, candidates = await self._seeded_service()
        second_candidate = candidates.candidates[1]

        await service.search_by_candidate(candidates.candidate_set_id, second_candidate.candidate_id, top_k=5)

        assert http_client.get_calls == [("https://example.com/full/labubu2.jpg", 5.0)]
