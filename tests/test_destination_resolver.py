"""Destination resolution must agree with the corpus/config canonical names."""

from app.agent.tools.destination_resolver import (
    DESTINATION_ALIASES,
    resolve_destination_name,
)
from app.configs.agent_config import DESTINATION_NAMES


def test_aliases_only_reference_canonical_names() -> None:
    assert set(DESTINATION_ALIASES) <= set(DESTINATION_NAMES)


def test_every_canonical_name_resolves_to_itself() -> None:
    for name in DESTINATION_NAMES:
        assert resolve_destination_name(name) == name


def test_common_aliases_resolve() -> None:
    assert resolve_destination_name("abc") == "Annapurna Base Camp (ABC)"
    assert resolve_destination_name("kori danda") == "Kori (Kori Danda)"
    assert resolve_destination_name("palpa") == "Tansen (Palpa)"
    assert resolve_destination_name("manaslu") == "Manaslu Circuit"


def test_unknown_name_returns_none() -> None:
    assert resolve_destination_name("Everest Base Camp") is None
