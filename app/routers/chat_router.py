from fastapi import APIRouter
from pydantic import BaseModel

from app.agent.agent import run_agent
from app.agent.tools.check_scope_tool import check_scope
from app.agent.tools.search_destinations_tool import search_destinations
from app.services.chat_service import handle_turn

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