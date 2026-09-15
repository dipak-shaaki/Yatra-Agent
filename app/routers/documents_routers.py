"""
Admin endpoints for managing the destination corpus, kept separate from the
end-user chat flow.
"""

import asyncio
import secrets

from fastapi import APIRouter, Depends, Header, HTTPException

from app.core.config import get_settings
from app.schemas.document_schema import DocumentUpsertRequest
from app.services.document_service import upsert_document

router = APIRouter()


def _require_admin_token(
    x_admin_token: str | None = Header(default=None),
) -> None:
    settings = get_settings()
    if not settings.admin_api_key:
        raise HTTPException(
            status_code=503, detail="ADMIN_API_KEY not configured on the server"
        )
    if not x_admin_token or not secrets.compare_digest(
        x_admin_token, settings.admin_api_key
    ):
        raise HTTPException(status_code=403, detail="Invalid or missing admin token")


@router.post(
    "/documents/upsert", dependencies=[Depends(_require_admin_token)]
)
async def upsert_document_endpoint(request: DocumentUpsertRequest):
    try:
        return await asyncio.to_thread(
            upsert_document, request.filename, request.content
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upsert document: {e}")
