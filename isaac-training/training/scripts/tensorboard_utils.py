"""Small, dependency-light helpers for writing scalar training metrics."""

from __future__ import annotations

import numbers
from collections.abc import Mapping
from typing import Any, Iterable


def _tag_name(tag: Any) -> str:
    """Convert Hydra/TorchRL keys into a stable TensorBoard tag."""

    if isinstance(tag, tuple):
        return "/".join(str(part) for part in tag)
    return str(tag)


def _scalar_value(value: Any) -> float | None:
    """Return a Python float for a scalar value, otherwise ``None``."""

    if isinstance(value, bool):
        return float(value)
    if isinstance(value, numbers.Number):
        return float(value)

    # Torch tensors are intentionally detected without importing torch so that
    # this utility remains importable in lightweight test and tooling jobs.
    numel = getattr(value, "numel", None)
    item = getattr(value, "item", None)
    if callable(numel) and callable(item):
        try:
            if numel() == 1:
                return float(item())
        except (TypeError, ValueError, RuntimeError):
            return None
    return None


def scalar_items(info: Mapping[Any, Any]) -> Iterable[tuple[str, float]]:
    """Yield TensorBoard-compatible scalar ``(tag, value)`` pairs."""

    for tag, value in info.items():
        scalar = _scalar_value(value)
        if scalar is not None:
            yield _tag_name(tag), scalar


def log_scalars(writer: Any, info: Mapping[Any, Any], step: int) -> None:
    """Write all scalar entries in ``info`` to a SummaryWriter-like object."""

    for tag, value in scalar_items(info):
        writer.add_scalar(tag, value, step)
