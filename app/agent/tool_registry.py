from collections.abc import Callable

from app.agent.tools.calculate_budget_estimate_tool import calculate_budget_estimate
from app.agent.tools.compare_destinations_tool import compare_destinations
from app.agent.tools.filter_by_criteria_tool import filter_by_criteria
from app.agent.tools.search_destinations_tool import search_destinations
from app.data_ingestion.corpus_meta import get_destinations


def _corpus_enums() -> tuple[list[str], list[int], list[str]]:
    """Provinces, difficulty levels and type tags actually present in the
    corpus — the tool schemas advertise exactly what the tools can match."""
    destinations = get_destinations()
    provinces = sorted({d.province for d in destinations})
    difficulties = sorted({d.difficulty for d in destinations})
    tags = sorted({tag for d in destinations for tag in d.tags})
    return provinces, difficulties, tags


def _tool_schemas() -> list[dict]:
    provinces, difficulties, tags = _corpus_enums()

    return [
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
                            "enum": provinces,
                            "description": "Nepali province the destination lies in",
                        },
                        "difficulty_level": {
                            "type": "integer",
                            "enum": difficulties,
                            "description": (
                                "Difficulty on a 1-5 scale (1 = easy, 5 = very hard); "
                                "matches adjacent levels too"
                            ),
                        },
                        "type": {
                            "type": "string",
                            "enum": tags,
                            "description": "Kind of destination",
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
                        "num_days": {
                            "type": "integer",
                            "description": "Trip length in days, if specified",
                        },
                    },
                    "required": ["destination_name"],
                },
            },
        },
    ]


# Maps tool name (as Groq will refer to it) -> actual callable
TOOL_DISPATCH: dict[str, Callable[..., dict | list]] = {
    "search_destinations": search_destinations,
    "filter_by_criteria": filter_by_criteria,
    "compare_destinations": compare_destinations,
    "calculate_budget_estimate": calculate_budget_estimate,
}


def get_tool_schemas() -> list[dict]:
    """Returns the full schema list to pass to Groq's `tools` parameter.

    Built per call from the current corpus, so a document upsert that adds a
    province or tag is advertised immediately without a process restart.
    """
    return _tool_schemas()


def call_tool(tool_name: str, **kwargs) -> dict | list:
    """Dispatch to the actual tool function by name. Raises KeyError if unknown."""
    if tool_name not in TOOL_DISPATCH:
        raise KeyError(f"Unknown tool: {tool_name}")
    return TOOL_DISPATCH[tool_name](**kwargs)
