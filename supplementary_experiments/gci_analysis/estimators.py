"""Lightweight estimators for adjustment-set comparisons.

These estimators are not intended to replace the paper's original ICPW, GBM, CBGPS,
npCBGPS, DRNets, or DRL experiments. They provide a fast and reproducible diagnostic
comparison of adjustment-set choices on the already generated synthetic datasets.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class EvaluationConfig:
    n_repeats: int = 20
    test_size: float = 0.2
    delta_t: float = 1.0
    n_neighbors: int = 50
    random_seed: int = 2026


def _make_model(config: EvaluationConfig):
    return make_pipeline(
        StandardScaler(),
        KNeighborsRegressor(n_neighbors=config.n_neighbors, weights="distance"),
    )


def evaluate_adjustment_set(
    df: pd.DataFrame,
    adjustment_vars: Sequence[str],
    config: EvaluationConfig,
    treatment_col: str = "t",
    outcome_col: str = "y",
    counterfactual_col: str = "y_delta",
) -> pd.DataFrame:
    """Evaluate one adjustment set with repeated train/test splits.

    The true MTEF target is y(t) - y(t-1), already stored as y - y_delta in the
    synthetic datasets. The prediction uses the fitted factual outcome model twice,
    once at the observed treatment and once at treatment minus delta_t, while keeping
    the selected pretreatment/safe covariates fixed.
    """
    missing = [c for c in [treatment_col, outcome_col, counterfactual_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    adjustment_vars = list(adjustment_vars)
    missing_adj = [c for c in adjustment_vars if c not in df.columns]
    if missing_adj:
        raise ValueError(f"Adjustment variables not found in dataframe: {missing_adj}")

    feature_cols = [treatment_col] + adjustment_vars
    rows: List[Dict[str, float]] = []
    indices = np.arange(len(df))

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

        for split_name, split_idx in (("train", train_idx), ("test", test_idx)):
            x_now = df.loc[split_idx, feature_cols].copy()
            x_delta = x_now.copy()
            x_delta[treatment_col] = x_delta[treatment_col] - config.delta_t

            pred_now = model.predict(x_now)
            pred_delta = model.predict(x_delta)
            pred_mtef = pred_now - pred_delta
            true_mtef = (
                df.loc[split_idx, outcome_col].to_numpy()
                - df.loc[split_idx, counterfactual_col].to_numpy()
            )
            factual_pred = pred_now
            factual_true = df.loc[split_idx, outcome_col].to_numpy()
            error = pred_mtef - true_mtef

            rows.append(
                {
                    "repeat": rep,
                    "split": split_name,
                    "rmse_mtef": float(np.sqrt(mean_squared_error(true_mtef, pred_mtef))),
                    "bias_mtef": float(np.mean(error)),
                    "mae_mtef": float(np.mean(np.abs(error))),
                    "rmse_factual": float(np.sqrt(mean_squared_error(factual_true, factual_pred))),
                }
            )
    return pd.DataFrame(rows)


def summarize_runs(run_df: pd.DataFrame) -> pd.DataFrame:
    if run_df.empty:
        raise ValueError("run_df is empty")
    summary = (
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
    return summary
