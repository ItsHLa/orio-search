from collections.abc import AsyncIterator
from typing import Any

import structlog
from openai import AsyncOpenAI

from app.config import settings

logger = structlog.get_logger(__name__)


class LLMService:
    """LLM answer generation using any OpenAI-compatible API (OpenAI, Ollama, Groq, etc.)."""

    def __init__(self) -> None:
        self._client: AsyncOpenAI | None = None

    def initialize(self) -> None:
        if not settings.llm.enabled:
            return
        self._client = AsyncOpenAI(
            base_url=settings.llm.base_url,
            api_key=settings.llm.api_key,
            timeout=settings.llm.timeout,
        )
        logger.info(
            "llm_initialized",
            provider=settings.llm.provider,
            model=settings.llm.model,
            base_url=settings.llm.base_url,
        )

    async def close(self) -> None:
        if self._client:
            await self._client.close()

    def _build_context(self, query: str, results: list[Any]) -> list[dict[str, str]]:
        """Build chat messages from search results."""
        context_parts: list[str] = []
        char_count = 0
        max_results = settings.llm.max_context_results
        max_chars = settings.llm.max_context_chars

        for i, r in enumerate(results[:max_results]):
            # Support both SearchResult objects and dicts
            title = r.title if hasattr(r, "title") else r.get("title", "")
            url = r.url if hasattr(r, "url") else r.get("url", "")
            content = r.content if hasattr(r, "content") else r.get("content", "")
            raw_content = (
                r.raw_content if hasattr(r, "raw_content") else r.get("raw_content")
            )

            # Prefer raw_content (full extraction) if available
            text = raw_content or content
            remaining = max_chars - char_count
            if remaining <= 0:
                break
            if len(text) > remaining:
                text = text[:remaining] + "..."

            context_parts.append(f"[{i + 1}] {title} - {url}\n{text}")
            char_count += len(text)

        context_block = "\n\n".join(context_parts)

        return [
            {"role": "system", "content": settings.llm.system_prompt},
            {
                "role": "user",
                "content": (
                    f"Search results for: {query}\n\n"
                    f"{context_block}\n\n"
                    f"Question: {query}"
                ),
            },
        ]

    async def generate_answer(self, query: str, results: list[Any]) -> str | None:
        """Generate a non-streaming answer. Returns None if disabled or on error."""
        if not self._client:
            return None

        messages = self._build_context(query, results)
        try:
            response = await self._client.chat.completions.create(
                model=settings.llm.model,
                messages=messages,
                max_tokens=settings.llm.max_tokens,
                temperature=settings.llm.temperature,
            )
            answer = response.choices[0].message.content
            logger.info(
                "llm_answer_generated",
                query=query,
                tokens=response.usage.total_tokens if response.usage else None,
            )
            return answer
        except Exception as e:
            logger.error("llm_answer_failed", query=query, error=str(e))
            return None

    async def generate_answer_stream(
        self, query: str, results: list[Any]
    ) -> AsyncIterator[str]:
        """Stream answer chunks. Yields nothing if disabled or on error."""
        if not self._client:
            return

        messages = self._build_context(query, results)
        try:
            stream = await self._client.chat.completions.create(
                model=settings.llm.model,
                messages=messages,
                max_tokens=settings.llm.max_tokens,
                temperature=settings.llm.temperature,
                stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content
        except Exception as e:
            logger.error("llm_stream_failed", query=query, error=str(e))
