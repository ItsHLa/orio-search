from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.requests import Request

from app.config import settings


def _get_key(request: Request) -> str:
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:]
    return get_remote_address(request)


limiter = Limiter(
    key_func=_get_key,
    storage_uri=settings.cache.redis_url if settings.rate_limit.enabled else "memory://",
    enabled=settings.rate_limit.enabled,
)
