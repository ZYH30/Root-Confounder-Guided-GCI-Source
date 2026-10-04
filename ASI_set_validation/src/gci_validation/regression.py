from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.base import RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer


@dataclass(frozen=True)
class ResidualFit:
    residuals: np.ndarray
    predictions: np.ndarray
    mse: float
    r2: float
    n_splits: int
    regressor: str


def make_regressor(name: str, *, seed: int, n_train: int) -> RegressorMixin:
    if name == "spline_ridge":
        knots = int(np.clip(round(n_train ** 0.25) + 3, 5, 12))
        return make_pipeline(
            SplineTransformer(n_knots=knots, degree=3, include_bias=False),
            Ridge(alpha=1e-2),
        )
    if name == "linear":
        return Ridge(alpha=1e-6)
    raise ValueError(f"unknown_regressor:{name}")


def _folds(
    n: int,
    n_splits: int,
    seed: int,
    groups: np.ndarray | None,
) -> tuple[Iterable[tuple[np.ndarray, np.ndarray]], int]:
    indices = np.arange(n)
    if groups is None:
        actual_splits = min(n_splits, max(2, n // 20))
        splitter = KFold(n_splits=actual_splits, shuffle=True, random_state=seed)
        return splitter.split(indices), actual_splits

    group_array = np.asarray(groups).reshape(-1)
    if group_array.shape[0] != n:
        raise ValueError("group_length_mismatch")
    n_groups = np.unique(group_array).size
    if n_groups < 2:
        raise ValueError("insufficient_groups_for_cross_fitting")
    actual_splits = min(n_splits, n_groups)
    splitter = GroupKFold(n_splits=actual_splits)
    return splitter.split(indices, groups=group_array), actual_splits


def out_of_fold_residuals(
    cause: np.ndarray,
    effect: np.ndarray,
    *,
    regressor: str = "spline_ridge",
    n_splits: int = 5,
    seed: int = 0,
    groups: np.ndarray | None = None,
) -> ResidualFit:
    cause_array = np.asarray(cause, dtype=float).reshape(-1, 1)
    effect_array = np.asarray(effect, dtype=float).reshape(-1)
    if cause_array.shape[0] != effect_array.shape[0]:
        raise ValueError("cause_effect_length_mismatch")

    predictions = np.full(effect_array.shape[0], np.nan, dtype=float)
    splits, actual_splits = _folds(effect_array.shape[0], n_splits, seed, groups)
    for fold_id, (train, test) in enumerate(splits):
        model = make_regressor(
            regressor,
            seed=seed + 104729 * (fold_id + 1),
            n_train=train.size,
        )
        model.fit(cause_array[train], effect_array[train])
        predictions[test] = np.asarray(model.predict(cause_array[test])).reshape(-1)

    if not np.all(np.isfinite(predictions)):
        raise RuntimeError("cross_fit_predictions_incomplete")
    residuals = effect_array - predictions
    return ResidualFit(
        residuals=residuals,
        predictions=predictions,
        mse=float(mean_squared_error(effect_array, predictions)),
        r2=float(r2_score(effect_array, predictions)),
        n_splits=actual_splits,
        regressor=regressor,
    )
