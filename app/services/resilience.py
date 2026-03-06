from __future__ import annotations

import asyncio

import httpx
import structlog
from circuitbreaker import circuit

from app.config import settings

logger = structlog.get_logger(__name__)

search_circuit = circuit(
    failure_threshold=settings.resilience.circuit_breaker_failure_threshold,
    recovery_timeout=settings.resilience.circuit_breaker_recovery_timeout,
    expected_exception=Exception,
)

extract_circuit = circuit(
    failure_threshold=settings.resilience.circuit_breaker_failure_threshold,
    recovery_timeout=settings.resilience.circuit_breaker_recovery_timeout,
    expected_exception=Exception,
)


async def retry_async(
    coro_factory,
    *,
    max_attempts: int | None = None,
    backoff_base: float | None = None,
    retryable_status_codes: list[int] | None = None,
):
    _max = max_attempts or settings.resilience.retry_max_attempts
    _backoff = backoff_base or settings.resilience.retry_backoff_base
    _codes = retryable_status_codes or settings.resilience.retry_on_status_codes

    last_exception: Exception | None = None
    for attempt in range(1, _max + 1):
        try:
            return await coro_factory()
        except httpx.HTTPStatusError as e:
            if e.response.status_code not in _codes:
                raise
            last_exception = e
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout) as e:
            last_exception = e

        if attempt < _max:
            delay = _backoff * (2 ** (attempt - 1))
            logger.warning("retrying", attempt=attempt, delay=delay, error=str(last_exception))
            await asyncio.sleep(delay)

    raise last_exception  # type: ignore[misc]
