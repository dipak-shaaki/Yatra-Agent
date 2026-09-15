"""
Admin endpoint(s) for managing the destination corpus. Not part of the
end-user chat flow — separate router since this is a data-management
concern, not a conversational one.
"""

from fastapi import APIRouter, HTTPException

from app.schemas.document_schema import DocumentUpsertRequest
from app.services.document_service import upsert_document

router = APIRouter()


@router.post("/documents/upsert")
async def upsert_document_endpoint(request: DocumentUpsertRequest):
    try:
        return upsert_document(request.filename, request.content)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upsert document: {e}")
