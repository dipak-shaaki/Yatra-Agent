from app.configs.agent_config import DESTINATION_NAMES

# Extra names a user or the model might say for each canonical destination.
# Canonical names themselves come from DESTINATION_NAMES so the two lists
# cannot drift apart.
DESTINATION_ALIASES: dict[str, list[str]] = {
    "Manaslu Circuit": ["manaslu"],
    "Annapurna Base Camp (ABC)": ["annapurna", "annapurna base camp", "abc"],
    "Mardi Himal": ["mardi", "mardi himal trek"],
    "Kori (Kori Danda)": ["kori", "kori danda"],
    "Badimalika": ["badimalika temple"],
    "Rara Lake": ["rara", "rara lake trek"],
    "Tansen (Palpa)": ["tansen", "palpa"],
}

_LOOKUP: dict[str, str] = {name.lower(): name for name in DESTINATION_NAMES}
for exact_name, aliases in DESTINATION_ALIASES.items():
    for alias in aliases:
        _LOOKUP[alias.lower()] = exact_name


def resolve_destination_name(raw_name: str) -> str | None:
    """
    Returns the exact corpus name for a given raw/casual name, or None
    if it doesn't match any known destination or alias.
    """
    return _LOOKUP.get(raw_name.strip().lower())
