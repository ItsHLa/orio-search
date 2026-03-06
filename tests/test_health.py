from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["service"] == "orio-search"


@pytest.mark.asyncio
async def test_tool_schema(client):
    resp = await client.get("/tool-schema")
    assert resp.status_code == 200
    data = resp.json()
    assert "tools" in data
    assert len(data["tools"]) == 2

    tool_names = {t["function"]["name"] for t in data["tools"]}
    assert tool_names == {"web_search", "web_extract"}


@pytest.mark.asyncio
async def test_tool_schema_search_params(client):
    resp = await client.get("/tool-schema")
    tools = resp.json()["tools"]
    search_tool = next(t for t in tools if t["function"]["name"] == "web_search")
    props = search_tool["function"]["parameters"]["properties"]

    assert "query" in props
    assert "search_depth" in props
    assert "topic" in props
    assert "max_results" in props
    assert "include_images" in props
    assert "time_range" in props
    assert "include_domains" in props
    assert "exclude_domains" in props


@pytest.mark.asyncio
async def test_openapi_docs(client):
    resp = await client.get("/openapi.json")
    assert resp.status_code == 200
    data = resp.json()
    assert data["info"]["title"] == "OrioSearch"
    assert data["info"]["version"] == "2.0.0"


@pytest.mark.asyncio
async def test_request_id_header(client):
    resp = await client.get("/health")
    assert "x-request-id" in resp.headers


@pytest.mark.asyncio
async def test_custom_request_id(client):
    resp = await client.get("/health", headers={"X-Request-ID": "my-trace-123"})
    assert resp.headers["x-request-id"] == "my-trace-123"
