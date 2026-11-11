"""Shared FastAPI dependencies for route protection."""

import secrets

from fastapi import Header, HTTPException

from app.core.config import get_settings


def require_admin(x_admin_token: str | None = Header(default=None)) -> None:
    """Guards mutating/admin endpoints (/documents/*, /debug/*)."""
    settings = get_settings()
    if not settings.admin_api_key:
        raise HTTPException(
            status_code=503, detail="ADMIN_API_KEY not configured on the server"
        )
    if not x_admin_token or not secrets.compare_digest(
        x_admin_token, settings.admin_api_key
    ):
        raise HTTPException(status_code=403, detail="Invalid or missing admin token")
