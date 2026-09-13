"""
Debug endpoints for testing individual agent tools directly (bypassing the
orchestrator), plus the real /chat and /chat/stream endpoints.
"""
import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.agent.tools.check_scope_tool import check_scope
from app.agent.tools.search_destinations_tool import search_destinations
from app.services.chat_service import handle_turn, handle_turn_stream

router = APIRouter()


class QueryRequest(BaseModel):
    query: str
    n_results: int = 5


class ChatRequest(BaseModel):
    query: str
    sender: str


@router.post("/debug/check-scope")
async def debug_check_scope(request: QueryRequest):
    return check_scope(request.query)


@router.post("/debug/search")
async def debug_search(request: QueryRequest):
    return {"results": search_destinations(request.query, n_results=request.n_results)}


@router.post("/chat")
async def chat(request: ChatRequest):
    answer = await handle_turn(request.query, sender=request.sender)
    return {"answer": answer}


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    async def event_generator():
        async for chunk in handle_turn_stream(request.query, sender=request.sender):
            yield f"data: {json.dumps({'chunk': chunk})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")