from app.agent.tools.check_scope_tool import check_scope
from app.agent.tools.search_destinations_tool import search_destinations
from app.agent.tools.filter_by_criteria_tool import filter_by_criteria
from app.agent.tools.compare_destinations_tool import compare_destinations
from app.agent.tools.calculate_budget_estimate_tool import calculate_budget_estimate
from app.agent.tools.get_conversation_context_tool import get_conversation_context

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "check_scope",
            "description": (
                "Determines if a user query is within Yatra's domain "
                "(the 10 documented Nepal destinations). Run this early "
                "on every real query before other tools."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The user's query to check"},
                },
                "required": ["query"],
            },
        },
    },
        {
        "type": "function",
        "function": {
            "name": "filter_by_criteria",
            "description": (
                "Find destinations matching specific criteria like province, "
                "difficulty level, or type (trek/pilgrimage/heritage town). "
                "Use for discovery questions like 'easy treks near Pokhara' or "
                "'destinations in Gandaki province', not single-destination lookups."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "province": {
                        "type": "string",
                        "description": "e.g. Gandaki, Bagmati, Karnali, Lumbini, Sudurpashchim",
                    },
                    "difficulty_level": {
                        "type": "string",
                        "description": "e.g. easy, moderate, hard",
                    },
                    "type": {
                        "type": "string",
                        "description": "e.g. trek, pilgrimage, hill town, heritage village, lake",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_destinations",
            "description": (
                "Hybrid search (semantic + keyword) over the destination corpus. "
                "Use for direct factual questions about a single destination or topic "
                "(e.g. permits, budget, best time to visit, difficulty)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"},
                    "n_results": {
                        "type": "integer",
                        "description": "Number of chunks to retrieve",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
        {
        "type": "function",
        "function": {
            "name": "compare_destinations",
            "description": (
                "Compare 2 or more named destinations side by side across "
                "difficulty, duration, budget, best time to visit, and permits. "
                "Use for 'X vs Y' style questions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "destination_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Exact destination names, e.g. 'Manaslu Circuit', "
                            "'Annapurna Base Camp (ABC)', 'Rara Lake'"
                        ),
                    },
                },
                "required": ["destination_names"],
            },
        },
    },
            {
        "type": "function",
        "function": {
            "name": "calculate_budget_estimate",
            "description": (
                "Get budget estimate info for a destination, optionally scaled "
                "to a specific number of days. Use for cost/budget questions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "destination_name": {"type": "string"},
                    "num_days": {"type": "integer", "description": "Trip length in days, if specified"},
                },
                "required": ["destination_name"],
            },
        },
    },
        {
        "type": "function",
        "function": {
            "name": "get_conversation_context",
            "description": (
                "Retrieve recent conversation history for this session. Use when "
                "the current query references something from earlier in the "
                "conversation (e.g. 'that place', 'the one you mentioned', 'it')."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sender": {"type": "string", "description": "Session/user identifier"},
                    "num_turns": {"type": "integer", "description": "How many recent messages to retrieve"},
                },
                "required": ["sender"],
            },
        },
    },
]

# Maps tool name (as Groq will refer to it) -> actual callable
TOOL_DISPATCH = {
    "check_scope": check_scope,
    "search_destinations": search_destinations,
    "filter_by_criteria": filter_by_criteria,
    "compare_destinations": compare_destinations,
    "calculate_budget_estimate": calculate_budget_estimate,
    "get_conversation_context": get_conversation_context,
}


def get_tool_schemas() -> list[dict]:
    """Returns the full schema list to pass to Groq's `tools` parameter."""
    return TOOL_SCHEMAS


def call_tool(tool_name: str, **kwargs) -> dict | list:
    """Dispatch to the actual tool function by name. Raises KeyError if unknown."""
    if tool_name not in TOOL_DISPATCH:
        raise KeyError(f"Unknown tool: {tool_name}")
    return TOOL_DISPATCH[tool_name](**kwargs)