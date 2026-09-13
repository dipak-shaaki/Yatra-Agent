DESTINATION_ALIASES: dict[str, list[str]] = {
    "Manaslu Circuit": ["manaslu"],
    "Annapurna Base Camp (ABC)": ["annapurna", "annapurna base camp", "abc"],
    "Mardi Himal": ["mardi", "mardi himal trek"],
    "Kori (Kori Danda)": ["kori", "kori danda"],
    "Badimalika": ["badimalika temple"],
    "Bandipur": [],
    "Panauti": [],
    "Gorkha": [],
    "Rara Lake": ["rara", "rara lake trek"],
    "Tansen (Palpa)": ["tansen", "palpa"],
}

_LOOKUP: dict[str, str] = {}
for exact_name, aliases in DESTINATION_ALIASES.items():
    _LOOKUP[exact_name.lower()] = exact_name
    for alias in aliases:
        _LOOKUP[alias.lower()] = exact_name


def resolve_destination_name(raw_name: str) -> str | None:
    """
    Returns the exact corpus name for a given raw/casual name, or None
    if it doesn't match any known destination or alias.
    """
    return _LOOKUP.get(raw_name.strip().lower())