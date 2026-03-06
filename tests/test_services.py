from __future__ import annotations

import asyncio
from collections import OrderedDict
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.config import AppConfig
from app.services.cache import CacheService
from app.services.extractor import ContentExtractor, ExtractionResult
from app.services.reranker import RerankerService
from app.services.search_backend import (
    BackendSearchResponse,
    DuckDuckGoBackend,
    FallbackSearchBackend,
    RawSearchResult,
    SearXNGBackend,
)


# ========== Cache service tests ==========

class TestCacheService:
    @pytest.mark.asyncio
    async def test_disabled_cache_returns_none(self):
        config = AppConfig()
        config.cache.enabled = False
        cache = CacheService(config)
        assert await cache.get_search("q", "h") is None
        assert await cache.get_extract("url") is None

    @pytest.mark.asyncio
    async def test_disabled_batch_returns_none(self):
        config = AppConfig()
        config.cache.enabled = False
        cache = CacheService(config)
        result = await cache.get_extract_batch(["u1", "u2"])
        assert result == {"u1": None, "u2": None}

    @pytest.mark.asyncio
    async def test_disabled_set_is_noop(self):
        config = AppConfig()
        config.cache.enabled = False
        cache = CacheService(config)
        # Should not raise
        await cache.set_search("q", "h", {"data": 1})
        await cache.set_extract("url", "content")
        await cache.set_extract_batch([("u1", "c1")])

    def test_hash_key_deterministic(self):
        key1 = CacheService._hash_key("prefix", "value")
        key2 = CacheService._hash_key("prefix", "value")
        assert key1 == key2
        assert key1.startswith("prefix:")

    def test_hash_key_different_values(self):
        key1 = CacheService._hash_key("p", "a")
        key2 = CacheService._hash_key("p", "b")
        assert key1 != key2


# ========== Extractor tests ==========

class TestContentExtractor:
    def _make_extractor(self, **kwargs) -> ContentExtractor:
        config = AppConfig(**kwargs) if kwargs else AppConfig()
        client = MagicMock(spec=httpx.AsyncClient)
        return ContentExtractor(config, client)

    def test_domain_semaphore_lru_eviction(self):
        config = AppConfig()
        config.extraction.domain_semaphore_max_size = 3
        config.extraction.domain_concurrency = 2
        ext = ContentExtractor(config, MagicMock())

        # Fill up
        ext._get_domain_semaphore("https://a.com/1")
        ext._get_domain_semaphore("https://b.com/1")
        ext._get_domain_semaphore("https://c.com/1")
        assert len(ext._domain_semaphores) == 3

        # Adding a 4th should evict the oldest (a.com)
        ext._get_domain_semaphore("https://d.com/1")
        assert len(ext._domain_semaphores) == 3
        assert "a.com" not in ext._domain_semaphores
        assert "d.com" in ext._domain_semaphores

    def test_domain_semaphore_lru_access_refreshes(self):
        config = AppConfig()
        config.extraction.domain_semaphore_max_size = 3
        ext = ContentExtractor(config, MagicMock())

        ext._get_domain_semaphore("https://a.com/1")
        ext._get_domain_semaphore("https://b.com/1")
        ext._get_domain_semaphore("https://c.com/1")

        # Access a.com again to refresh it
        ext._get_domain_semaphore("https://a.com/2")

        # Now adding d.com should evict b.com (oldest non-refreshed)
        ext._get_domain_semaphore("https://d.com/1")
        assert "a.com" in ext._domain_semaphores
        assert "b.com" not in ext._domain_semaphores

    def test_get_headers_returns_valid_headers(self):
        ext = self._make_extractor()
        headers = ext._get_headers()
        assert "User-Agent" in headers
        assert "Accept" in headers
        assert "Mozilla" in headers["User-Agent"]

    def test_to_markdown_with_title(self):
        html = "<html><head><title>Test Page</title></head><body></body></html>"
        result = ContentExtractor._to_markdown("Some text", html, "https://example.com")
        assert "# Test Page" in result
        assert "Source: https://example.com" in result
        assert "---" in result
        assert "Some text" in result

    def test_to_markdown_without_title(self):
        html = "<html><head></head><body></body></html>"
        result = ContentExtractor._to_markdown("Text only", html, "https://example.com")
        assert "# " not in result
        assert "Source: https://example.com" in result
        assert "Text only" in result


# ========== Search backend tests ==========

class TestFallbackSearchBackend:
    @pytest.mark.asyncio
    async def test_fallback_on_primary_failure(self):
        primary = MagicMock(spec=SearXNGBackend)
        primary.search = AsyncMock(side_effect=RuntimeError("SearXNG down"))

        fallback = MagicMock(spec=DuckDuckGoBackend)
        fallback_resp = BackendSearchResponse(
            results=[RawSearchResult(title="DDG", url="https://ddg.com", snippet="fallback", score=0.5)],
        )
        fallback.search = AsyncMock(return_value=fallback_resp)

        backend = FallbackSearchBackend(primary, fallback)
        result = await backend.search("test query", max_results=1)

        primary.search.assert_called_once()
        fallback.search.assert_called_once()
        assert result.results[0].title == "DDG"

    @pytest.mark.asyncio
    async def test_no_fallback_when_primary_succeeds(self):
        primary_resp = BackendSearchResponse(
            results=[RawSearchResult(title="Primary", url="https://p.com", snippet="ok", score=0.9)],
        )
        primary = MagicMock()
        primary.search = AsyncMock(return_value=primary_resp)

        fallback = MagicMock()
        fallback.search = AsyncMock()

        backend = FallbackSearchBackend(primary, fallback)
        result = await backend.search("test", max_results=1)

        primary.search.assert_called_once()
        fallback.search.assert_not_called()
        assert result.results[0].title == "Primary"


# ========== Reranker tests ==========

class TestRerankerService:
    def test_disabled_reranker_passthrough(self):
        reranker = RerankerService()
        # Not initialized (disabled)
        results = [{"title": "A", "content": "a", "score": 0.5}]
        output = reranker.rerank("query", results)
        assert output == results  # unchanged

    def test_empty_results_passthrough(self):
        reranker = RerankerService()
        output = reranker.rerank("query", [])
        assert output == []
