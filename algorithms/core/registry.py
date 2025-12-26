"""
Lightweight registry for model builders so evaluation code can stay declarative.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List


ModelFactory = Callable[..., Any]
_MODEL_REGISTRY: Dict[str, ModelFactory] = {}


def register_model(name: str, factory: ModelFactory) -> None:
    """
    Register a model factory under a human-readable name.
    """
    _MODEL_REGISTRY[name] = factory


def get_model(name: str) -> ModelFactory:
    """
    Retrieve a registered factory.
    """
    try:
        return _MODEL_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Model '{name}' is not registered. Available: {list_models()}") from exc


def list_models() -> List[str]:
    """
    List registered model names.
    """
    return sorted(_MODEL_REGISTRY.keys())


__all__ = ["ModelFactory", "register_model", "get_model", "list_models"]
