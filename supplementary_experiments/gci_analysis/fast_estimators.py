"""Fast nonlinear outcome-regression estimators for robustness diagnostics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


@dataclass(frozen=True)
class FastEvaluationConfig:
    n_repeats: int = 10
    test_size: float = 0.2
    delta_t: float = 1.0
    random_seed: int = 2026
    poly_degree: int = 3
    ridge_alpha: float = 1.0


def _make_model(config: FastEvaluationConfig):
    # Polynomial ridge is used here because this robustness experiment requires many
    # repeated fits. It captures nonlinear interactions while remaining deterministic
    # and considerably faster than the main deep/reweighting estimators.
    return make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree=config.poly_degree, include_bias=False),
        Ridge(alpha=config.ridge_alpha),
    )


def evaluate_adjustment_set_fast(
    df: pd.DataFrame,
    adjustment_vars: Sequence[str],
    config: FastEvaluationConfig,
    treatment_col: str = "t",
    outcome_col: str = "y",
    counterfactual_col: str = "y_delta",
) -> pd.DataFrame:
    """Evaluate one adjustment set using a fast nonlinear outcome model.

    Only test-split metrics are returned. The goal is a robustness diagnostic for
    adjustment-set choices rather than a replacement for the main deep causal models.
    """
    required = [treatment_col, outcome_col, counterfactual_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    adjustment_vars = list(adjustment_vars)
    missing_adj = [col for col in adjustment_vars if col not in df.columns]
    if missing_adj:
        raise ValueError(f"Adjustment variables not found in dataframe: {missing_adj}")

    feature_cols = [treatment_col] + adjustment_vars
    indices = np.arange(len(df))
    rows: List[Dict[str, float]] = []

    for rep in range(config.n_repeats):
        seed = config.random_seed + rep
        train_idx, test_idx = train_test_split(
            indices,
            test_size=config.test_size,
            random_state=seed,
            shuffle=True,
        )
        model = _make_model(config)
        model.fit(df.loc[train_idx, feature_cols], df.loc[train_idx, outcome_col])

        x_now = df.loc[test_idx, feature_cols].copy()
        x_delta = x_now.copy()
        x_delta[treatment_col] = x_delta[treatment_col] - config.delta_t

        pred_now = model.predict(x_now)
        pred_delta = model.predict(x_delta)
        pred_mtef = pred_now - pred_delta
        true_mtef = df.loc[test_idx, outcome_col].to_numpy() - df.loc[test_idx, counterfactual_col].to_numpy()
        error = pred_mtef - true_mtef

        rows.append(
            {
                "repeat": rep,
                "split": "test",
                "rmse_mtef": float(np.sqrt(mean_squared_error(true_mtef, pred_mtef))),
                "bias_mtef": float(np.mean(error)),
                "mae_mtef": float(np.mean(np.abs(error))),
                "rmse_factual": float(np.sqrt(mean_squared_error(df.loc[test_idx, outcome_col], pred_now))),
            }
        )
    return pd.DataFrame(rows)


def summarize_fast_runs(run_df: pd.DataFrame) -> pd.DataFrame:
    if run_df.empty:
        raise ValueError("run_df is empty")
    return (
        run_df.groupby("split", as_index=False)
        .agg(
            rmse_mtef_mean=("rmse_mtef", "mean"),
            rmse_mtef_sd=("rmse_mtef", "std"),
            bias_mtef_mean=("bias_mtef", "mean"),
            bias_mtef_sd=("bias_mtef", "std"),
            mae_mtef_mean=("mae_mtef", "mean"),
            rmse_factual_mean=("rmse_factual", "mean"),
        )
    )
