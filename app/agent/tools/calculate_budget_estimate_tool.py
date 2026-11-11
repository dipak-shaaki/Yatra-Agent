"""
Budget estimates with the arithmetic done in code: the corpus stores
numeric budget ranges (per-day or whole-trip) plus typical day ranges, so
scaling to a requested trip length is computed here instead of improvised
by the LLM from formatted text.
"""

from app.data_ingestion.corpus_meta import get_destinations, resolve_destination


def calculate_budget_estimate(
    destination_name: str, num_days: int | None = None
) -> dict:
    resolved = resolve_destination(destination_name)
    if not resolved:
        return {"error": f"Unknown destination: '{destination_name}'"}
    if num_days is not None and num_days < 1:
        return {"error": "num_days must be a positive integer"}

    dest = next(d for d in get_destinations() if d.name == resolved)

    result = {
        "destination": dest.name,
        "estimated_budget_npr": dest.estimated_budget_npr,
        "typical_time_required": dest.time_required,
        "budget_per_day": dest.budget_per_day,
        "requested_num_days": num_days,
    }

    if num_days is None:
        result["typical_budget_npr"] = [dest.budget_min_npr, dest.budget_max_npr]
        return result

    if dest.budget_per_day:
        est_min = dest.budget_min_npr * num_days
        est_max = dest.budget_max_npr * num_days
        basis = "per-day budget multiplied by the requested number of days"
    elif dest.days_min <= num_days <= dest.days_max:
        est_min, est_max = dest.budget_min_npr, dest.budget_max_npr
        basis = "the corpus estimate already covers this trip length"
    else:
        # shortcut: linear extrapolation outside the typical range; upgrade
        # to per-section cost modelling if the corpus ever gains one.
        est_min = round(dest.budget_min_npr * num_days / dest.days_max)
        est_max = round(dest.budget_max_npr * num_days / dest.days_min)
        basis = (
            "rough proportional scaling from the corpus's typical "
            f"{dest.days_min}-{dest.days_max} day range"
        )

    result["estimated_total_npr"] = [est_min, est_max]
    result["basis"] = basis
    return result
