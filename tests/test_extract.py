from __future__ import annotations

import pytest


# ---- Basic extraction ----

@pytest.mark.asyncio
async def test_extract_single_url(client):
    resp = await client.post("/extract", json={"urls": ["https://example.com"]})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["results"]) == 1
    assert data["results"][0]["url"] == "https://example.com"
    assert "Extracted content" in data["results"][0]["raw_content"]
    assert data["failed_results"] == []
    assert data["response_time"] >= 0


@pytest.mark.asyncio
async def test_extract_multiple_urls(client):
    urls = ["https://example.com/1", "https://example.com/2", "https://example.com/3"]
    resp = await client.post("/extract", json={"urls": urls})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["results"]) == 3
    returned_urls = {r["url"] for r in data["results"]}
    assert returned_urls == set(urls)


@pytest.mark.asyncio
async def test_extract_with_text_format(client):
    resp = await client.post("/extract", json={
        "urls": ["https://example.com"],
        "format": "text",
    })
    assert resp.status_code == 200
    assert len(resp.json()["results"]) == 1


@pytest.mark.asyncio
async def test_extract_with_markdown_format(client):
    resp = await client.post("/extract", json={
        "urls": ["https://example.com"],
        "format": "markdown",
    })
    assert resp.status_code == 200


# ---- Failed extractions ----

@pytest.mark.asyncio
async def test_extract_failed_url(client):
    resp = await client.post("/extract", json={
        "urls": ["https://fail.example.com/page"],
    })
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["results"]) == 0
    assert len(data["failed_results"]) == 1
    assert data["failed_results"][0]["url"] == "https://fail.example.com/page"
    assert data["failed_results"][0]["error"] != ""


@pytest.mark.asyncio
async def test_extract_mixed_success_and_failure(client):
    resp = await client.post("/extract", json={
        "urls": ["https://example.com/ok", "https://fail.example.com/bad"],
    })
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["results"]) == 1
    assert len(data["failed_results"]) == 1
    assert data["results"][0]["url"] == "https://example.com/ok"
    assert data["failed_results"][0]["url"] == "https://fail.example.com/bad"


# ---- Cache behavior ----

@pytest.mark.asyncio
async def test_extract_caches_results(client, app):
    url = "https://example.com/cached"
    resp1 = await client.post("/extract", json={"urls": [url]})
    assert resp1.status_code == 200

    # Verify it's in cache
    cache = app.state.cache
    cached = await cache.get_extract(url)
    assert cached is not None
    assert "Extracted content" in cached


@pytest.mark.asyncio
async def test_extract_serves_from_cache(client, app):
    url = "https://example.com/precached"
    # Pre-populate cache
    await app.state.cache.set_extract(url, "Cached content here")

    resp = await client.post("/extract", json={"urls": [url]})
    assert resp.status_code == 200
    assert resp.json()["results"][0]["raw_content"] == "Cached content here"


@pytest.mark.asyncio
async def test_extract_batch_cache_partial_hit(client, app):
    """Some URLs cached, some not — should fetch only uncached."""
    await app.state.cache.set_extract("https://example.com/a", "Cached A")

    resp = await client.post("/extract", json={
        "urls": ["https://example.com/a", "https://example.com/b"],
    })
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["results"]) == 2
    urls_to_content = {r["url"]: r["raw_content"] for r in data["results"]}
    assert urls_to_content["https://example.com/a"] == "Cached A"
    assert "Extracted content" in urls_to_content["https://example.com/b"]


# ---- Validation errors ----

@pytest.mark.asyncio
async def test_extract_empty_urls(client):
    resp = await client.post("/extract", json={"urls": []})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_extract_too_many_urls(client):
    urls = [f"https://example.com/{i}" for i in range(21)]
    resp = await client.post("/extract", json={"urls": urls})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_extract_missing_urls(client):
    resp = await client.post("/extract", json={})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_extract_invalid_format(client):
    resp = await client.post("/extract", json={
        "urls": ["https://example.com"],
        "format": "html",
    })
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_extract_no_body(client):
    resp = await client.post("/extract")
    assert resp.status_code == 422
