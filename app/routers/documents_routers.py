"""
Admin endpoints for managing the destination corpus, kept separate from the
end-user chat flow.
"""

import asyncio

from fastapi import APIRouter, Depends, HTTPException

from app.routers.dependencies import require_admin
from app.schemas.document_schema import DocumentUpsertRequest
from app.services.document_service import upsert_document

router = APIRouter()


@router.post(
    "/documents/upsert",
    dependencies=[Depends(require_admin)],
    responses={422: {"description": "Document content failed schema validation"}},
)
async def upsert_document_endpoint(request: DocumentUpsertRequest):
    try:
        return await asyncio.to_thread(
            upsert_document, request.filename, request.content
        )
    except ValueError as e:
        # Schema-invalid documents are a client error, not a server error.
        raise HTTPException(status_code=422, detail=str(e))
