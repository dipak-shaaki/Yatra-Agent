"""
The /chat endpoint (JSON body + session-token auth + rate limiting), the
/session endpoint that mints tokens, and the admin-guarded /debug endpoints.

Sessions: POST /api/v1/session returns a server-minted token; every /chat
call must send it as X-Session-Token. The token is the sender identity, so
clients cannot read or overwrite another session's history. Redis is
required for both (fail-closed 503 when it is down) and enforces a fixed
window rate limit per token.
"""

import asyncio
import json
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.agent.tools.search_destinations_tool import search_destinations
from app.components.redis.session_store import (
    check_rate_limit,
    create_session_token,
    validate_session_token,
)
from app.core.config import get_settings
from app.routers.dependencies import require_admin
from app.services.chat_service import handle_turn, handle_turn_stream

router = APIRouter()


class SessionResponse(BaseModel):
    session_token: str


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    stream: bool = False


class DebugQueryRequest(BaseModel):
    query: str
    n_results: int = 5


def _authorized_sender(x_session_token: str | None) -> str:
    """Validate the session token; returns the sender id (the token itself)."""
    if not x_session_token:
        raise HTTPException(status_code=401, detail="Missing X-Session-Token header")
    try:
        if not validate_session_token(x_session_token):
            raise HTTPException(
                status_code=401, detail="Unknown or expired session token"
            )
    except HTTPException:
        raise
    except Exception:
        # Redis unavailable: fail closed rather than trusting the client.
        raise HTTPException(
            status_code=503, detail="Session store unavailable, try again shortly"
        )
    return x_session_token


def _enforce_rate_limit(sender: str) -> None:
    settings = get_settings()
    try:
        allowed, retry_after = check_rate_limit(
            sender, settings.chat_rate_limit_per_minute
        )
    except Exception:
        raise HTTPException(
            status_code=503, detail="Session store unavailable, try again shortly"
        )
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded",
            headers={"Retry-After": str(retry_after)},
        )


@router.post("/session")
async def create_session() -> SessionResponse:
    try:
        return SessionResponse(session_token=create_session_token())
    except Exception:
        raise HTTPException(
            status_code=503, detail="Session store unavailable, try again shortly"
        )


@router.post("/debug/search", dependencies=[Depends(require_admin)])
async def debug_search(request: DebugQueryRequest):
    return {
        "results": await asyncio.to_thread(
            search_destinations, request.query, n_results=request.n_results
        )
    }


@router.post("/chat")
async def chat(
    request: ChatRequest,
    x_session_token: str | None = Header(default=None),
):
    """Send a message to Yatra.

    Requires a session token from POST /api/v1/session (X-Session-Token
    header). stream=true streams the answer as SSE chunks; stream=false
    returns the full answer as JSON. retrieved_context and tools_called are
    included only when the server runs with EXPOSE_EVAL_CONTEXT=true.
    """
    sender = _authorized_sender(x_session_token)
    _enforce_rate_limit(sender)

    if request.stream:
        return StreamingResponse(
            _event_stream(request.query, sender), media_type="text/event-stream"
        )

    result = await handle_turn(request.query, sender=sender)
    payload: dict = {"answer": result["answer"]}
    if get_settings().expose_eval_context:
        payload["retrieved_context"] = result["retrieved_context"]
        payload["tools_called"] = result["tools_called"]
    return payload


async def _event_stream(query: str, sender: str) -> AsyncGenerator[str]:
    turn_sink: list = []
    stream = handle_turn_stream(query, sender, turn_sink=turn_sink)
    # Drive manually so we can detect completion; once the turn is done,
    # emit a final metadata event before [DONE]. The UI ignores non-{chunk}
    # payloads, so this stays backward compatible.
    while True:
        try:
            chunk = await stream.__anext__()
        except StopAsyncIteration:
            break
        yield f"data: {json.dumps({'chunk': chunk})}\n\n"
    if turn_sink and get_settings().expose_eval_context:
        turn = turn_sink[0]
        yield (
            "data: "
            + json.dumps(
                {
                    "retrieved_context": turn.retrieved_context,
                    "tools_called": turn.tools_called,
                }
            )
            + "\n\n"
        )
    yield "data: [DONE]\n\n"
