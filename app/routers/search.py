from __future__ import annotations

import asyncio
import hashlib
import json
import time

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request

from app.auth import verify_api_key
from app.config import settings
from app.models.schemas import (
    ImageResult,
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from app.rate_limit import limiter

logger = structlog.get_logger(__name__)
router = APIRouter()


def _params_hash(req: SearchRequest) -> str:
    data = req.model_dump(exclude={"query"})
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:12]


@router.post("/search", response_model=SearchResponse)
@limiter.limit(settings.rate_limit.search_rate)
async def search(
    request: Request,
    body: SearchRequest,
    api_key: str | None = Depends(verify_api_key),
) -> SearchResponse:
    try:
        return await asyncio.wait_for(
            _do_search(request, body),
            timeout=settings.resilience.request_timeout,
        )
    except asyncio.TimeoutError:
        raise HTTPException(504, "Request timed out")


async def _do_search(request: Request, body: SearchRequest) -> SearchResponse:
    start = time.perf_counter()

    cache = request.app.state.cache
    backend = request.app.state.search_backend
    extractor = request.app.state.extractor
    reranker = request.app.state.reranker

    # Check cache
    ph = _params_hash(body)
    cached = await cache.get_search(body.query, ph)
    if cached:
        elapsed = time.perf_counter() - start
        return SearchResponse(**{**cached, "response_time": round(elapsed, 3)})

    # Query search backend with graceful degradation
    try:
        backend_resp = await backend.search(
            body.query,
            max_results=body.max_results,
            topic=body.topic.value,
            time_range=body.time_range.value if body.time_range else None,
            include_domains=body.include_domains or None,
            exclude_domains=body.exclude_domains or None,
            include_images=body.include_images,
        )
    except Exception as e:
        logger.error("search_backend_failed", error=str(e))
        stale = await cache.get_search(body.query, ph)
        if stale:
            elapsed = time.perf_counter() - start
            logger.info("serving_stale_cache", query=body.query)
            return SearchResponse(**{**stale, "response_time": round(elapsed, 3)})
        raise HTTPException(503, "Search service unavailable")

    # Build results
    results: list[SearchResult] = []
    for raw in backend_resp.results:
        results.append(
            SearchResult(
                title=raw.title, url=raw.url, content=raw.snippet,
                score=raw.score, raw_content=None,
            )
        )

    # Rerank results
    if reranker and settings.rerank.enabled and results:
        result_dicts = [r.model_dump() for r in results]
        reranked = reranker.rerank(body.query, result_dicts, top_k=body.max_results)
        results = [SearchResult(**r) for r in reranked]

    # Advanced depth: fetch and extract content
    if body.search_depth.value == "advanced" or body.include_raw_content:
        urls = [r.url for r in results]
        extractions = await extractor.extract_urls(urls)
        to_cache: list[tuple[str, str]] = []
        for result, extraction in zip(results, extractions):
            if extraction.success:
                result.raw_content = extraction.content
                to_cache.append((result.url, extraction.content))
        await cache.set_extract_batch(to_cache)

    # Images
    images = [ImageResult(url=img.url, description=img.description) for img in backend_resp.images]

    elapsed = time.perf_counter() - start
    response = SearchResponse(
        query=body.query, results=results, images=images,
        response_time=round(elapsed, 3),
    )

    # Cache response
    cache_data = response.model_dump()
    del cache_data["response_time"]
    await cache.set_search(body.query, ph, cache_data)

    return response
