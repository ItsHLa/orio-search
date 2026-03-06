from __future__ import annotations

from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient


@pytest.mark.asyncio
async def test_auth_disabled_no_key_required(client):
    """When auth is disabled (default), requests work without API key."""
    resp = await client.post("/search", json={"query": "test", "max_results": 1})
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_auth_disabled_key_ignored(client):
    """When auth is disabled, providing a key still works."""
    resp = await client.post(
        "/search",
        json={"query": "test", "max_results": 1},
        headers={"Authorization": "Bearer some-key"},
    )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_health_no_auth_required(client):
    """/health is always unauthenticated."""
    resp = await client.get("/health")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_tool_schema_no_auth_required(client):
    """/tool-schema is always unauthenticated."""
    resp = await client.get("/tool-schema")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_auth_enabled_missing_key(app):
    """When auth is enabled, missing key returns 401."""
    with patch("app.auth.settings") as mock_settings:
        mock_settings.auth.enabled = True
        mock_settings.auth.api_keys = ["valid-key-123"]

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post("/search", json={"query": "test", "max_results": 1})
            assert resp.status_code == 401


@pytest.mark.asyncio
async def test_auth_enabled_invalid_key(app):
    """When auth is enabled, wrong key returns 403."""
    with patch("app.auth.settings") as mock_settings:
        mock_settings.auth.enabled = True
        mock_settings.auth.api_keys = ["valid-key-123"]

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/search",
                json={"query": "test", "max_results": 1},
                headers={"Authorization": "Bearer wrong-key"},
            )
            assert resp.status_code == 403


@pytest.mark.asyncio
async def test_auth_enabled_valid_key(app):
    """When auth is enabled, correct key works."""
    with patch("app.auth.settings") as mock_settings:
        mock_settings.auth.enabled = True
        mock_settings.auth.api_keys = ["valid-key-123"]

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/search",
                json={"query": "test", "max_results": 1},
                headers={"Authorization": "Bearer valid-key-123"},
            )
            assert resp.status_code == 200


@pytest.mark.asyncio
async def test_auth_enabled_extract_endpoint(app):
    """Auth also applies to /extract."""
    with patch("app.auth.settings") as mock_settings:
        mock_settings.auth.enabled = True
        mock_settings.auth.api_keys = ["valid-key-123"]

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            # Without key
            resp = await ac.post("/extract", json={"urls": ["https://example.com"]})
            assert resp.status_code == 401

            # With valid key
            resp = await ac.post(
                "/extract",
                json={"urls": ["https://example.com"]},
                headers={"Authorization": "Bearer valid-key-123"},
            )
            assert resp.status_code == 200


@pytest.mark.asyncio
async def test_auth_enabled_stream_endpoint(app):
    """Auth also applies to /search/stream."""
    with patch("app.auth.settings") as mock_settings:
        mock_settings.auth.enabled = True
        mock_settings.auth.api_keys = ["valid-key-123"]

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post(
                "/search/stream",
                json={"query": "test", "max_results": 1},
            )
            assert resp.status_code == 401


@pytest.mark.asyncio
async def test_auth_multiple_keys(app):
    """Multiple API keys should all work."""
    with patch("app.auth.settings") as mock_settings:
        mock_settings.auth.enabled = True
        mock_settings.auth.api_keys = ["key-1", "key-2", "key-3"]

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            for key in ["key-1", "key-2", "key-3"]:
                resp = await ac.post(
                    "/search",
                    json={"query": "test", "max_results": 1},
                    headers={"Authorization": f"Bearer {key}"},
                )
                assert resp.status_code == 200, f"Key {key} should be valid"
