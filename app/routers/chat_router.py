"""
The /chat endpoint (JSON body + session-token auth + rate limiting), the
/session endpoint that mints tokens, and the admin-guarded /debug endpoints.

Sessions: POST /api/v1/session returns a server-minted token; every /chat
call must send it as X-Session-Token. The token is the sender identity, so
clients cannot read or overwrite another session's history. Redis is
required for both (fail-closed 503 when it is down) and enforces a fixed
window rate limit per token; /session has its own per-IP limit so one caller
cannot mint endless tokens to sidestep the per-token limit.

SSE is typed: every frame is `event: <type>` + JSON `{"type": ...}` where
type is token | meta | error | done. The JSON mode returns ChatResponse.
"""

import asyncio
import json
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from app.agent.tools.search_destinations_tool import search_destinations
from app.components.redis.session_store import (
    check_rate_limit,
    check_session_mint_limit,
    create_session_token,
    validate_session_token,
)
from app.core.config import get_settings
from app.routers.dependencies import require_admin
from app.services.chat_service import handle_turn, handle_turn_stream
from app.utils.logger import log_event

router = APIRouter()

STREAM_ERROR_MESSAGE = (
    "Sorry, the connection to the model failed. Please try again in a moment."
)


class SessionResponse(BaseModel):
    session_token: str


class ChatRequest(BaseModel):
    query: str = Field(min_length=1)
    stream: bool = False

    @field_validator("query")
    @classmethod
    def validate_query_length(cls, v: str) -> str:
        limit = get_settings().max_query_length
        if len(v) > limit:
            raise ValueError(f"query must be at most {limit} characters")
        return v


class ChatResponse(BaseModel):
    answer: str
    failed: bool = False
    retrieved_context: list[str] | None = None
    tools_called: list[str] | None = None
    llm_calls: int | None = None


class ErrorResponse(BaseModel):
    detail: str


class DebugQueryRequest(BaseModel):
    query: str
    n_results: int | None = None


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


@router.post("/session", response_model=SessionResponse)
async def create_session(request: Request) -> SessionResponse:
    client_ip = request.client.host if request.client else "unknown"
    try:
        allowed, retry_after = check_session_mint_limit(client_ip)
    except Exception:
        raise HTTPException(
            status_code=503, detail="Session store unavailable, try again shortly"
        )
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Too many sessions created from this address",
            headers={"Retry-After": str(retry_after)},
        )
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


@router.post("/chat", response_model=ChatResponse, response_model_exclude_none=True)
async def chat(
    request: ChatRequest,
    x_session_token: str | None = Header(default=None),
):
    """Send a message to Yatra.

    Requires a session token from POST /api/v1/session (X-Session-Token
    header). stream=true streams typed SSE events; stream=false returns the
    full answer as JSON. retrieved_context and tools_called are included only
    when the server runs with EXPOSE_EVAL_CONTEXT=true.
    """
    sender = _authorized_sender(x_session_token)
    _enforce_rate_limit(sender)

    if request.stream:
        return StreamingResponse(
            _event_stream(request.query, sender), media_type="text/event-stream"
        )

    try:
        result = await handle_turn(request.query, sender=sender)
    except Exception as e:
        log_event("chat_turn_failed", error=type(e).__name__, detail=str(e)[:300])
        raise HTTPException(
            status_code=503,
            detail="Yatra is temporarily unavailable. Please try again.",
        )
    payload: dict = {"answer": result["answer"], "failed": result["failed"]}
    if get_settings().expose_eval_context:
        payload["retrieved_context"] = result["retrieved_context"]
        payload["tools_called"] = result["tools_called"]
        payload["llm_calls"] = result["llm_calls"]
    return payload


def _sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps({'type': event, **payload})}\n\n"


async def _event_stream(query: str, sender: str) -> AsyncGenerator[str]:
    turn_sink: list = []
    try:
        stream = handle_turn_stream(query, sender, turn_sink=turn_sink)
        async for chunk in stream:
            yield _sse("token", {"content": chunk})
    except Exception as e:
        log_event("chat_stream_failed", error=type(e).__name__, detail=str(e)[:300])
        yield _sse("error", {"message": STREAM_ERROR_MESSAGE})
    else:
        if turn_sink and get_settings().expose_eval_context:
            turn = turn_sink[0]
            yield _sse(
                "meta",
                {
                    "retrieved_context": turn.retrieved_context,
                    "tools_called": turn.tools_called,
                    "llm_calls": turn.llm_calls,
                    "failed": turn.failed,
                },
            )
    yield _sse("done", {})
