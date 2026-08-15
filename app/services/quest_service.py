"""QuestService -- QUEST branch 2 (plan §P2.5, PRD §F5): resolves an
Out-of-Knowledge query by letting the operator pick a web sample image, then
running image-to-image search on it.

Flow: `find_image_candidates(q0)` calls `ddgs` (sync-only, no async client --
docs/references.md §5.5) inside `asyncio.to_thread` to fetch image
candidates, keeps their real URLs in an in-memory TTL cache keyed by a
server-generated `candidate_set_id`/`candidate_id`, and returns only the
`candidate_id` + `thumbnail_url` to the caller. The client never sends a URL
back -- `search_by_candidate()` looks the real URL up in this same cache and
downloads exactly that URL itself, never one supplied by the client
(SSRF guard, CLAUDE.md §7). The chosen image is then run through
`SearchService.search_by_image()`, the same vector space P1.1 already wired.

Both steps require timeout + fallback to `SearchService.search(q0,
mode="semantic")` on error (CLAUDE.md §4: QUEST must never hang the UI) --
`ddgs` has no SLA and this is a third-party scraper, not an official API.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Literal, Protocol

import httpx
from ddgs.exceptions import DDGSException
from PIL import Image

from app.services.search_service import SearchHit

logger = logging.getLogger(__name__)

# Fixed content-safety policy, not a per-deployment tunable -- CLAUDE.md §6's
# "no magic numbers" targets thresholds/timeouts/limits, not a categorical
# safety default like this.
_SAFESEARCH = "moderate"


class _SearchServiceLike(Protocol):
    async def search(self, query: str, mode: str, top_k: int) -> list[SearchHit]: ...
    async def search_by_image(self, image: Image.Image, top_k: int) -> list[SearchHit]: ...


class _ImageSearchClient(Protocol):
    def images(self, query: str, max_results: int, safesearch: str) -> list[dict[str, Any]]: ...


class _HttpResponse(Protocol):
    content: bytes

    def raise_for_status(self) -> None: ...


class _HttpClient(Protocol):
    async def get(self, url: str, timeout: float) -> _HttpResponse: ...


class QuestServiceError(Exception):
    """Raised for an expected caller-facing failure (unknown/expired
    candidate_set_id or candidate_id) -- carries its own error_code/status_code
    so app/api/quest.py can map it onto the unified error schema without
    parsing the message text (CLAUDE.md §7)."""

    def __init__(self, message: str, error_code: str, status_code: int = 404) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.status_code = status_code


@dataclass(frozen=True)
class ImageCandidate:
    candidate_id: str
    thumbnail_url: str
    title: str


@dataclass(frozen=True)
class ImageCandidatesResult:
    """`status="fallback"` means ddgs failed/timed out/returned nothing --
    `fallback_hits` are regular semantic-search results for `q0`, and there
    is no `candidate_set_id` to pick from (CLAUDE.md §4: never hang the UI)."""

    status: Literal["candidates", "fallback"]
    candidate_set_id: str | None
    candidates: list[ImageCandidate]
    fallback_hits: list[SearchHit]


@dataclass(frozen=True)
class ImageSearchResult:
    status: Literal["ok", "fallback"]
    hits: list[SearchHit]


@dataclass(frozen=True)
class _CachedCandidate:
    image_url: str
    thumbnail_url: str
    title: str


@dataclass(frozen=True)
class _CacheEntry:
    query: str
    candidates_by_id: dict[str, _CachedCandidate] = field(default_factory=dict)


class _TtlCache:
    """In-memory candidate_set_id -> _CacheEntry store (SSRF guard: the real
    image URLs never leave the server). `clock` is injected so tests can
    simulate TTL expiry without a real sleep."""

    def __init__(self, ttl_seconds: float, clock) -> None:
        self._ttl_seconds = ttl_seconds
        self._clock = clock
        self._entries: dict[str, tuple[float, _CacheEntry]] = {}

    def put(self, key: str, entry: _CacheEntry) -> None:
        self._entries[key] = (self._clock() + self._ttl_seconds, entry)

    def get(self, key: str) -> _CacheEntry | None:
        item = self._entries.get(key)
        if item is None:
            return None
        expires_at, entry = item
        if self._clock() >= expires_at:
            del self._entries[key]
            return None
        return entry


class QuestService:
    def __init__(
        self,
        search_service: _SearchServiceLike,
        image_search_client: _ImageSearchClient,
        http_client: _HttpClient,
        max_candidates: int,
        image_candidates_timeout_sec: float,
        image_download_timeout_sec: float,
        candidate_cache_ttl_sec: float,
        clock=None,
        id_factory=None,
    ) -> None:
        self._search_service = search_service
        self._image_search_client = image_search_client
        self._http_client = http_client
        self._max_candidates = max_candidates
        self._image_candidates_timeout_sec = image_candidates_timeout_sec
        self._image_download_timeout_sec = image_download_timeout_sec
        self._clock = clock or time.monotonic
        self._id_factory = id_factory or (lambda: str(uuid.uuid4()))
        self._cache = _TtlCache(candidate_cache_ttl_sec, self._clock)

    async def find_image_candidates(self, query: str, top_k: int) -> ImageCandidatesResult:
        try:
            raw_results = await asyncio.wait_for(
                asyncio.to_thread(
                    self._image_search_client.images,
                    query,
                    max_results=self._max_candidates,
                    safesearch=_SAFESEARCH,
                ),
                timeout=self._image_candidates_timeout_sec,
            )
        except (DDGSException, asyncio.TimeoutError) as exc:
            logger.warning("quest image-candidates fallback: query=%r reason=%r", query, exc)
            return await self._fallback_candidates(query, top_k)

        if not raw_results:
            logger.info("quest image-candidates fallback: query=%r reason=no results", query)
            return await self._fallback_candidates(query, top_k)

        candidate_set_id = self._id_factory()
        candidates_by_id: dict[str, _CachedCandidate] = {}
        candidates: list[ImageCandidate] = []
        for raw in raw_results:
            candidate_id = self._id_factory()
            title = raw.get("title", "")
            candidates_by_id[candidate_id] = _CachedCandidate(
                image_url=raw["image"], thumbnail_url=raw["thumbnail"], title=title
            )
            candidates.append(
                ImageCandidate(
                    candidate_id=candidate_id, thumbnail_url=raw["thumbnail"], title=title
                )
            )
        self._cache.put(
            candidate_set_id, _CacheEntry(query=query, candidates_by_id=candidates_by_id)
        )
        return ImageCandidatesResult(
            status="candidates",
            candidate_set_id=candidate_set_id,
            candidates=candidates,
            fallback_hits=[],
        )

    async def _fallback_candidates(self, query: str, top_k: int) -> ImageCandidatesResult:
        hits = await self._search_service.search(query, mode="semantic", top_k=top_k)
        return ImageCandidatesResult(
            status="fallback", candidate_set_id=None, candidates=[], fallback_hits=hits
        )

    async def search_by_candidate(
        self, candidate_set_id: str, candidate_id: str, top_k: int
    ) -> ImageSearchResult:
        entry = self._cache.get(candidate_set_id)
        if entry is None:
            raise QuestServiceError(
                f"Unknown or expired candidate_set_id: {candidate_set_id!r}",
                error_code="candidate_set_not_found",
                status_code=404,
            )
        candidate = entry.candidates_by_id.get(candidate_id)
        if candidate is None:
            raise QuestServiceError(
                f"Unknown candidate_id: {candidate_id!r}",
                error_code="candidate_not_found",
                status_code=404,
            )

        try:
            response = await self._http_client.get(
                candidate.image_url, timeout=self._image_download_timeout_sec
            )
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
            image.load()
        except (httpx.HTTPError, OSError, Image.DecompressionBombError) as exc:
            logger.warning(
                "quest image-search fallback: candidate_id=%s reason=%r", candidate_id, exc
            )
            hits = await self._search_service.search(entry.query, mode="semantic", top_k=top_k)
            return ImageSearchResult(status="fallback", hits=hits)

        hits = await self._search_service.search_by_image(image, top_k=top_k)
        return ImageSearchResult(status="ok", hits=hits)
