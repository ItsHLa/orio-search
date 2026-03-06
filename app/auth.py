from __future__ import annotations

import secrets

from fastapi import Depends, HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import settings

security = HTTPBearer(auto_error=False)


async def verify_api_key(
    credentials: HTTPAuthorizationCredentials | None = Security(security),
) -> str | None:
    if not settings.auth.enabled:
        return None
    if credentials is None:
        raise HTTPException(401, "Missing API key. Provide Authorization: Bearer <key>")
    if not any(
        secrets.compare_digest(credentials.credentials, key)
        for key in settings.auth.api_keys
    ):
        raise HTTPException(403, "Invalid API key")
    return credentials.credentials
