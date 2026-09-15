"""
Debug endpoints that exercise individual agent tools directly, plus the real
/chat endpoint.

/chat takes query, sender, and stream as query parameters (each rendered as
its own Swagger input) and streams when stream=true, else returns JSON.
"""

import asyncio
import json
from collections.abc import AsyncGenerator

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


@router.post("/debug/check-scope")
async def debug_check_scope(request: QueryRequest):
    return await asyncio.to_thread(check_scope, request.query)


@router.post("/debug/search")
async def debug_search(request: QueryRequest):
    return {
        "results": await asyncio.to_thread(
            search_destinations, request.query, n_results=request.n_results
        )
    }


@router.post("/chat")
async def chat(
    query: str = Query(
        ...,
        description="The user's message to Yatra",
        examples=["Compare an easy trek near Pokhara with something under NPR 30,000."],
    ),
    sender: str = Query(
        ...,
        description="Conversation/session identifier used for Redis memory",
        examples=["test_user"],
    ),
    stream: bool = Query(
        False,
        description="`false` returns the full answer as JSON; `true` streams it as SSE chunks.",
    ),
):
    """Send a message to Yatra.

    All fields are query parameters, so Swagger renders one labeled input box
    per field plus a true/false `stream` selector. Pass stream=true to get the
    answer as a Server-Sent Events (SSE) stream of text chunks.
    """
    if stream:
        return StreamingResponse(
            _event_stream(query, sender), media_type="text/event-stream"
        )

    result = await handle_turn(query, sender=sender)
    return {
        "answer": result["answer"],
        "retrieved_context": result["retrieved_context"],
    }


async def _event_stream(query: str, sender: str) -> AsyncGenerator[str]:
    turn_sink: list = []
    stream = handle_turn_stream(query, sender, turn_sink=turn_sink)
    # Drive manually so we can detect completion; once the turn is done,
    # emit a final metadata event with retrieved_context before [DONE].
    # The UI ignores non-{chunk} payloads, so this is backward compatible.
    while True:
        try:
            chunk = await stream.__anext__()
        except StopAsyncIteration:
            break
        yield f"data: {json.dumps({'chunk': chunk})}\n\n"
    if turn_sink:
        yield (
            "data: " + json.dumps({"retrieved_context": turn_sink[0].retrieved_context}) + "\n\n"
        )
    yield "data: [DONE]\n\n"
