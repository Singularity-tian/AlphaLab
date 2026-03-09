"""Factor registry for discovering and instantiating factors by name."""

from __future__ import annotations

import logging
from typing import Type

from alphalab.factors.base import Factor

logger = logging.getLogger(__name__)

_REGISTRY: dict[str, Type[Factor]] = {}


def register_factor(cls: Type[Factor]) -> Type[Factor]:
    """Decorator to register a factor class."""
    _REGISTRY[cls.name] = cls
    logger.debug("Registered factor: %s", cls.name)
    return cls


def get_factor(name: str) -> Factor:
    """Instantiate a registered factor by name."""
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise KeyError(f"Factor '{name}' not found. Available: {available}")
    return _REGISTRY[name]()


def list_factors() -> list[str]:
    """List all registered factor names."""
    return sorted(_REGISTRY.keys())
