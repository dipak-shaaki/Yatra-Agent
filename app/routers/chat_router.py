"""
Debug endpoints for testing individual agent tools directly (bypassing the
orchestrator), plus the real document-management and chat endpoints.

/chat takes a JSON body ({"query": ..., "sender": ...}) and streams its
reply when called with ?stream=true, or returns plain JSON otherwise —
one endpoint, one request shape, toggled by a query param.
"""
import json
from typing import AsyncGenerator

from fastapi import APIRouter, Query
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
async def chat(
    request: ChatRequest,
    stream: bool = Query(False, description="If true, streams the answer as SSE chunks instead of returning plain JSON."),
):
    """Send a message to Yatra. Pass ?stream=true to get the answer as a
    Server-Sent Events stream of text chunks instead of a single JSON body."""
    if stream:
        return StreamingResponse(_event_stream(request.query, request.sender), media_type="text/event-stream")

    result = await handle_turn(request.query, sender=request.sender)
    return {"answer": result["answer"], "retrieved_context": result["retrieved_context"]}


async def _event_stream(query: str, sender: str) -> AsyncGenerator[str, None]:
    async for chunk in handle_turn_stream(query, sender=sender):
        yield f"data: {json.dumps({'chunk': chunk})}\n\n"
    yield "data: [DONE]\n\n"