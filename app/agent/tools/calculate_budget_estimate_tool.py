from app.agent.tools.destination_resolver import resolve_destination_name
from app.db.chroma.client import get_collection


def calculate_budget_estimate(destination_name: str, num_days: int | None = None) -> dict:
    """
    Returns the destination's raw budget estimate and typical time required,
    plus the user's requested num_days (if given) so the LLM can reason
    about whether/how to scale the estimate — rather than this tool
    fabricating a multiplied total from inconsistently formatted text.
    """
    resolved = resolve_destination_name(destination_name)
    if not resolved:
        return {"error": f"Unknown destination: '{destination_name}'"}

    collection = get_collection()
    results = collection.get(where={"name": {"$eq": resolved}}, limit=1)

    if not results["ids"]:
        return {"error": f"No data found for: '{resolved}'"}

    metadata = results["metadatas"][0]
    return {
        "destination": resolved,
        "estimated_budget_npr": metadata.get("estimated_budget_npr"),
        "typical_time_required": metadata.get("time_required"),
        "requested_num_days": num_days,
    }