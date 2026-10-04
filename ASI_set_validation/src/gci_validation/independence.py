from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import pdist, squareform


@dataclass(frozen=True)
class IndependenceResult:
    statistic: float
    p_value: float | None
    bandwidth_x: float
    bandwidth_y: float
    n_used: int
    permutations: int


def _median_bandwidth(values: np.ndarray) -> float:
    distances = pdist(values.reshape(-1, 1), metric="euclidean")
    positive = distances[distances > np.finfo(float).eps]
    if positive.size == 0:
        return 1.0
    bandwidth = float(np.median(positive))
    return max(bandwidth, np.finfo(float).eps)


def _rbf_kernel(values: np.ndarray, bandwidth: float) -> np.ndarray:
    squared = squareform(pdist(values.reshape(-1, 1), metric="sqeuclidean"))
    return np.exp(-squared / (2.0 * bandwidth**2))


def _center(kernel: np.ndarray) -> np.ndarray:
    row_mean = kernel.mean(axis=1, keepdims=True)
    col_mean = kernel.mean(axis=0, keepdims=True)
    return kernel - row_mean - col_mean + kernel.mean()


def _normalized_hsic_from_centered(kc: np.ndarray, lc: np.ndarray) -> float:
    numerator = float(np.sum(kc * lc))
    denominator = float(np.sqrt(np.sum(kc * kc) * np.sum(lc * lc)))
    if denominator <= np.finfo(float).eps:
        return 0.0
    return float(np.clip(numerator / denominator, 0.0, 1.0))


def hsic_test(
    x: np.ndarray,
    y: np.ndarray,
    *,
    permutations: int = 0,
    max_samples: int = 500,
    seed: int = 0,
) -> IndependenceResult:
    """Normalized RBF-HSIC with an optional exact permutation calibration.

    Subsampling is paired and deterministic. The returned statistic, rather than the
    p-value, is used as the primary direction score; p-values are diagnostic because
    finite permutation resolution can otherwise dominate direction ties.
    """

    x_array = np.asarray(x, dtype=float).reshape(-1)
    y_array = np.asarray(y, dtype=float).reshape(-1)
    if x_array.shape[0] != y_array.shape[0]:
        raise ValueError("hsic_length_mismatch")
    finite = np.isfinite(x_array) & np.isfinite(y_array)
    x_array, y_array = x_array[finite], y_array[finite]
    if x_array.size < 10:
        raise ValueError("hsic_insufficient_samples")

    rng = np.random.default_rng(seed)
    if x_array.size > max_samples:
        selected = np.sort(rng.choice(x_array.size, size=max_samples, replace=False))
        x_array, y_array = x_array[selected], y_array[selected]

    bandwidth_x = _median_bandwidth(x_array)
    bandwidth_y = _median_bandwidth(y_array)
    kc = _center(_rbf_kernel(x_array, bandwidth_x))
    lc = _center(_rbf_kernel(y_array, bandwidth_y))
    observed = _normalized_hsic_from_centered(kc, lc)

    p_value: float | None = None
    if permutations > 0:
        exceed = 0
        for _ in range(permutations):
            order = rng.permutation(x_array.size)
            permuted = _normalized_hsic_from_centered(kc, lc[order][:, order])
            exceed += int(permuted >= observed)
        p_value = float((exceed + 1) / (permutations + 1))

    return IndependenceResult(
        statistic=observed,
        p_value=p_value,
        bandwidth_x=bandwidth_x,
        bandwidth_y=bandwidth_y,
        n_used=x_array.size,
        permutations=permutations,
    )
