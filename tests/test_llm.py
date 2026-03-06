"""Tests for AI answer generation (LLM integration)."""

from __future__ import annotations

import json

import pytest
from httpx import ASGITransport, AsyncClient

from tests.conftest import DisabledLLMService, FakeLLMService


# ============================================================
# /search with include_answer
# ============================================================


async def test_search_with_include_answer(client: AsyncClient):
    """include_answer=true returns an AI-generated answer."""
    resp = await client.post("/search", json={"query": "what is python", "include_answer": True})
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] is not None
    assert "test answer" in data["answer"]
    assert len(data["results"]) > 0


async def test_search_without_include_answer(client: AsyncClient):
    """include_answer=false (default) returns null answer."""
    resp = await client.post("/search", json={"query": "what is python"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] is None


async def test_search_answer_field_in_response(client: AsyncClient):
    """The answer field always appears in the response (null or string)."""
    resp = await client.post("/search", json={"query": "test", "include_answer": False})
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data


async def test_search_answer_llm_disabled(app, client: AsyncClient):
    """When LLM is disabled, include_answer=true returns null answer, search still works."""
    app.state.llm = DisabledLLMService()
    resp = await client.post("/search", json={"query": "test query", "include_answer": True})
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] is None
    assert len(data["results"]) > 0


async def test_search_answer_llm_failure(app, client: AsyncClient):
    """When LLM fails, answer is null but search results are still returned."""
    app.state.llm = FakeLLMService(fail=True)
    resp = await client.post("/search", json={"query": "test query", "include_answer": True})
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] is None
    assert len(data["results"]) > 0


async def test_search_answer_cached(app, client: AsyncClient):
    """Second request with include_answer uses cached answer, no extra LLM call."""
    llm = FakeLLMService()
    app.state.llm = llm

    # First request — generates answer
    resp1 = await client.post("/search", json={"query": "cache test", "include_answer": True})
    assert resp1.status_code == 200
    assert resp1.json()["answer"] is not None
    assert llm.call_count == 1

    # Second request — should use cached answer (note: full search response is also cached)
    resp2 = await client.post("/search", json={"query": "cache test", "include_answer": True})
    assert resp2.status_code == 200
    assert resp2.json()["answer"] is not None
    # Search response is cached at the top level, so LLM isn't called again
    assert llm.call_count == 1


async def test_search_answer_with_advanced_depth(client: AsyncClient):
    """include_answer works together with search_depth=advanced."""
    resp = await client.post("/search", json={
        "query": "docker compose tutorial",
        "include_answer": True,
        "search_depth": "advanced",
        "max_results": 2,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] is not None
    # Advanced depth also extracts content
    for result in data["results"]:
        assert result["raw_content"] is not None


# ============================================================
# /search/stream with include_answer
# ============================================================


async def test_stream_with_answer(client: AsyncClient):
    """Streaming with include_answer emits answer_chunk and answer_done events."""
    resp = await client.post(
        "/search/stream",
        json={"query": "test stream answer", "include_answer": True, "max_results": 2},
    )
    assert resp.status_code == 200

    events = _parse_sse(resp.text)

    event_types = [e["event"] for e in events]
    assert "result" in event_types
    assert "answer_chunk" in event_types
    assert "answer_done" in event_types
    assert "done" in event_types

    # answer_done should come before done
    assert event_types.index("answer_done") < event_types.index("done")

    # answer_chunks should contain text
    answer_chunks = [e for e in events if e["event"] == "answer_chunk"]
    assert len(answer_chunks) > 0
    for chunk in answer_chunks:
        chunk_data = json.loads(chunk["data"])
        assert "text" in chunk_data
        assert len(chunk_data["text"]) > 0


async def test_stream_without_answer(client: AsyncClient):
    """Streaming without include_answer does not emit answer events."""
    resp = await client.post(
        "/search/stream",
        json={"query": "test no answer", "max_results": 2},
    )
    assert resp.status_code == 200

    events = _parse_sse(resp.text)

    event_types = [e["event"] for e in events]
    assert "answer_chunk" not in event_types
    assert "answer_done" not in event_types
    assert "done" in event_types


async def test_stream_answer_llm_disabled(app, client: AsyncClient):
    """Streaming with disabled LLM skips answer events gracefully."""
    app.state.llm = DisabledLLMService()
    resp = await client.post(
        "/search/stream",
        json={"query": "test disabled", "include_answer": True, "max_results": 2},
    )
    assert resp.status_code == 200

    events = _parse_sse(resp.text)

    event_types = [e["event"] for e in events]
    # Disabled LLM produces no answer_chunk events
    assert "answer_chunk" not in event_types
    assert "done" in event_types


# ============================================================
# Helpers
# ============================================================


def _parse_sse(text: str) -> list[dict[str, str]]:
    """Parse Server-Sent Events text into a list of {event, data} dicts."""
    events: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("event:"):
            current["event"] = line[len("event:"):].strip()
        elif line.startswith("data:"):
            current["data"] = line[len("data:"):].strip()
        elif line == "" and current:
            if "event" in current and "data" in current:
                events.append(current)
            current = {}
    if "event" in current and "data" in current:
        events.append(current)
    return events
