"""Detector registry: `get_detector(cfg, n_features)` builds a detector from its config."""

from collections.abc import Callable, Mapping

from cascadelad.detectors.base import Detector

_REGISTRY: dict[str, type[Detector]] = {}


def register_detector(name: str) -> Callable[[type[Detector]], type[Detector]]:
    def decorator(cls: type[Detector]) -> type[Detector]:
        if name in _REGISTRY:
            raise ValueError(f"detector '{name}' is already registered")
        _REGISTRY[name] = cls
        return cls

    return decorator


def get_detector(cfg: Mapping, n_features: int) -> Detector:
    """Build the detector named cfg["name"]. `n_features` is the vocab size."""
    name = cfg["name"]
    if name not in _REGISTRY:
        raise KeyError(f"unknown detector '{name}'; known: {sorted(_REGISTRY)}")
    return _REGISTRY[name](cfg, n_features)
