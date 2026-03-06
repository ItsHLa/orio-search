from __future__ import annotations

from typing import Any

import structlog

from app.config import settings

logger = structlog.get_logger(__name__)


class RerankerService:
    def __init__(self) -> None:
        self._ranker = None

    def initialize(self) -> None:
        if not settings.rerank.enabled:
            return
        from flashrank import Ranker

        logger.info("loading_rerank_model", model=settings.rerank.model)
        self._ranker = Ranker(model_name=settings.rerank.model)
        logger.info("rerank_model_loaded")

    def rerank(
        self,
        query: str,
        results: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        if not self._ranker or not results:
            return results

        from flashrank import RerankRequest

        top_k = top_k or settings.rerank.top_k

        passages = [
            {"id": i, "text": f"{r.get('title', '')} {r.get('content', '')}", "meta": r}
            for i, r in enumerate(results)
        ]

        rerank_req = RerankRequest(query=query, passages=passages)
        reranked = self._ranker.rerank(rerank_req)

        output: list[dict[str, Any]] = []
        for item in reranked[:top_k]:
            original = item["meta"]
            original["score"] = round(item["score"], 4)
            output.append(original)

        return output
