from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.models.schemas import (
    ContentFormat,
    ExtractRequest,
    ExtractResponse,
    ExtractResult,
    FailedResult,
    ImageResult,
    SearchDepth,
    SearchRequest,
    SearchResponse,
    SearchResult,
    TimeRange,
    Topic,
)


class TestSearchRequest:
    def test_minimal(self):
        req = SearchRequest(query="hello")
        assert req.query == "hello"
        assert req.search_depth == SearchDepth.basic
        assert req.topic == Topic.general
        assert req.max_results == 5
        assert req.include_images is False
        assert req.include_raw_content is False
        assert req.time_range is None
        assert req.include_domains == []
        assert req.exclude_domains == []

    def test_full(self):
        req = SearchRequest(
            query="test",
            search_depth="advanced",
            topic="news",
            max_results=10,
            include_images=True,
            include_raw_content=True,
            time_range="week",
            include_domains=["example.com"],
            exclude_domains=["spam.com"],
        )
        assert req.search_depth == SearchDepth.advanced
        assert req.topic == Topic.news
        assert req.time_range == TimeRange.week

    def test_max_results_validation(self):
        with pytest.raises(ValidationError):
            SearchRequest(query="test", max_results=0)
        with pytest.raises(ValidationError):
            SearchRequest(query="test", max_results=21)

    def test_invalid_depth(self):
        with pytest.raises(ValidationError):
            SearchRequest(query="test", search_depth="ultra")

    def test_invalid_topic(self):
        with pytest.raises(ValidationError):
            SearchRequest(query="test", topic="sports")

    def test_invalid_time_range(self):
        with pytest.raises(ValidationError):
            SearchRequest(query="test", time_range="decade")

    def test_missing_query(self):
        with pytest.raises(ValidationError):
            SearchRequest()


class TestSearchResponse:
    def test_structure(self):
        resp = SearchResponse(
            query="test",
            results=[
                SearchResult(title="T", url="https://u.com", content="C", score=0.9),
            ],
            images=[
                ImageResult(url="https://img.com/1.jpg", description="desc"),
            ],
            response_time=1.23,
        )
        assert resp.query == "test"
        assert len(resp.results) == 1
        assert len(resp.images) == 1
        assert resp.response_time == 1.23

    def test_empty_results(self):
        resp = SearchResponse(query="test", results=[], response_time=0.5)
        assert resp.results == []
        assert resp.images == []

    def test_result_with_raw_content(self):
        result = SearchResult(
            title="T", url="https://u.com", content="C",
            score=0.8, raw_content="Full text here",
        )
        assert result.raw_content == "Full text here"

    def test_result_without_raw_content(self):
        result = SearchResult(title="T", url="https://u.com", content="C")
        assert result.raw_content is None
        assert result.score == 0.0


class TestExtractRequest:
    def test_minimal(self):
        req = ExtractRequest(urls=["https://example.com"])
        assert req.format == ContentFormat.markdown

    def test_text_format(self):
        req = ExtractRequest(urls=["https://example.com"], format="text")
        assert req.format == ContentFormat.text

    def test_empty_urls(self):
        with pytest.raises(ValidationError):
            ExtractRequest(urls=[])

    def test_too_many_urls(self):
        urls = [f"https://example.com/{i}" for i in range(21)]
        with pytest.raises(ValidationError):
            ExtractRequest(urls=urls)

    def test_max_urls(self):
        urls = [f"https://example.com/{i}" for i in range(20)]
        req = ExtractRequest(urls=urls)
        assert len(req.urls) == 20

    def test_invalid_format(self):
        with pytest.raises(ValidationError):
            ExtractRequest(urls=["https://example.com"], format="html")


class TestExtractResponse:
    def test_structure(self):
        resp = ExtractResponse(
            results=[ExtractResult(url="https://u.com", raw_content="content")],
            failed_results=[FailedResult(url="https://bad.com", error="timeout")],
            response_time=2.5,
        )
        assert len(resp.results) == 1
        assert len(resp.failed_results) == 1
        assert resp.response_time == 2.5

    def test_empty_response(self):
        resp = ExtractResponse(results=[], response_time=0.1)
        assert resp.results == []
        assert resp.failed_results == []
