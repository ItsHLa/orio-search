from __future__ import annotations

import json

import pytest


@pytest.mark.asyncio
async def test_stream_basic(client):
    resp = await client.post(
        "/search/stream",
        json={"query": "test stream", "max_results": 2},
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers.get("content-type", "")

    # Parse SSE events
    events = _parse_sse(resp.text)
    result_events = [e for e in events if e["event"] == "result"]
    done_events = [e for e in events if e["event"] == "done"]

    assert len(result_events) == 2
    assert len(done_events) == 1

    # Verify result structure
    first = json.loads(result_events[0]["data"])
    assert "title" in first
    assert "url" in first
    assert "content" in first
    assert "score" in first

    # Verify done event
    done_data = json.loads(done_events[0]["data"])
    assert "response_time" in done_data


@pytest.mark.asyncio
async def test_stream_with_images(client):
    resp = await client.post(
        "/search/stream",
        json={"query": "cats", "include_images": True, "max_results": 1},
    )
    assert resp.status_code == 200
    events = _parse_sse(resp.text)
    image_events = [e for e in events if e["event"] == "image"]
    assert len(image_events) > 0
    img = json.loads(image_events[0]["data"])
    assert "url" in img
    assert "description" in img


@pytest.mark.asyncio
async def test_stream_advanced_with_extraction(client):
    resp = await client.post(
        "/search/stream",
        json={"query": "test", "search_depth": "advanced", "max_results": 2},
    )
    assert resp.status_code == 200
    events = _parse_sse(resp.text)

    extraction_events = [e for e in events if e["event"] == "extraction"]
    assert len(extraction_events) > 0

    ext = json.loads(extraction_events[0]["data"])
    assert "url" in ext
    assert "raw_content" in ext


@pytest.mark.asyncio
async def test_stream_backend_error(client_failing_backend):
    resp = await client_failing_backend.post(
        "/search/stream",
        json={"query": "test", "max_results": 1},
    )
    assert resp.status_code == 200  # SSE always returns 200
    events = _parse_sse(resp.text)
    error_events = [e for e in events if e["event"] == "error"]
    assert len(error_events) == 1


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
