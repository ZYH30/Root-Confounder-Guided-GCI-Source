from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PreparedPair:
    x: np.ndarray
    y: np.ndarray
    original_n: int
    dropped_n: int
    sampled_n: int
    x_center: float
    y_center: float
    x_scale: float
    y_scale: float


def _as_1d_float(values: np.ndarray | list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    return array


def robust_scale(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Median/IQR scaling with a standard-deviation fallback."""

    center = float(np.median(values))
    q25, q75 = np.quantile(values, [0.25, 0.75])
    scale = float((q75 - q25) / 1.349)
    if not np.isfinite(scale) or scale <= np.finfo(float).eps:
        scale = float(np.std(values, ddof=1))
    if not np.isfinite(scale) or scale <= np.finfo(float).eps:
        raise ValueError("constant_or_degenerate_variable")
    return (values - center) / scale, center, scale


def prepare_continuous_pair(
    x: np.ndarray | list[float],
    y: np.ndarray | list[float],
    *,
    min_samples: int = 40,
    max_samples: int | None = None,
    seed: int = 0,
) -> PreparedPair:
    x_array = _as_1d_float(x)
    y_array = _as_1d_float(y)
    if x_array.shape[0] != y_array.shape[0]:
        raise ValueError("x_y_length_mismatch")

    original_n = x_array.shape[0]
    finite = np.isfinite(x_array) & np.isfinite(y_array)
    x_array = x_array[finite]
    y_array = y_array[finite]
    dropped_n = int(original_n - x_array.shape[0])

    if x_array.shape[0] < min_samples:
        raise ValueError(f"insufficient_samples:{x_array.shape[0]}<{min_samples}")
    if np.unique(x_array).size < 4 or np.unique(y_array).size < 4:
        raise ValueError("too_few_unique_values_for_continuous_method")

    if max_samples is not None and x_array.shape[0] > max_samples:
        rng = np.random.default_rng(seed)
        chosen = np.sort(rng.choice(x_array.shape[0], size=max_samples, replace=False))
        x_array = x_array[chosen]
        y_array = y_array[chosen]

    x_scaled, x_center, x_scale = robust_scale(x_array)
    y_scaled, y_center, y_scale = robust_scale(y_array)
    return PreparedPair(
        x=x_scaled.astype(float, copy=False),
        y=y_scaled.astype(float, copy=False),
        original_n=original_n,
        dropped_n=dropped_n,
        sampled_n=x_scaled.shape[0],
        x_center=x_center,
        y_center=y_center,
        x_scale=x_scale,
        y_scale=y_scale,
    )

