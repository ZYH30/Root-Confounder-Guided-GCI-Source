from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np


DirectionLabel = Literal["x_to_y", "y_to_x", "undirected", "invalid"]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


@dataclass(frozen=True)
class DirectionResult:
    """One direction method's result.

    The signed-score convention is shared by every method: positive supports X -> Y,
    negative supports Y -> X. Confidence is explicitly marked as calibrated or not.
    """

    method: str
    score: float
    direction: DirectionLabel
    confidence: float
    applicable: bool
    diagnostics: dict[str, Any] = field(default_factory=dict)
    runtime_seconds: float = 0.0
    seed: int = 0
    calibrated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))

    @staticmethod
    def direction_from_score(score: float, threshold: float) -> DirectionLabel:
        if not np.isfinite(score):
            return "invalid"
        if score > threshold:
            return "x_to_y"
        if score < -threshold:
            return "y_to_x"
        return "undirected"

    @classmethod
    def invalid(
        cls,
        method: str,
        reason: str,
        *,
        seed: int = 0,
        diagnostics: dict[str, Any] | None = None,
    ) -> "DirectionResult":
        details = dict(diagnostics or {})
        details["invalid_reason"] = reason
        return cls(
            method=method,
            score=float("nan"),
            direction="invalid",
            confidence=0.0,
            applicable=False,
            diagnostics=details,
            seed=seed,
        )
