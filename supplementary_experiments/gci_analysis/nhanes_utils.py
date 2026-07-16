"""Utilities for the NHANES internal-baseline supplementary experiment.

The functions in this file deliberately avoid rerunning the ASI module.  They
reuse the NHANES processed table and evaluate several adjustment specifications
for the triglyceride--glucose application.  Because NHANES has no ground-truth
counterfactual outcomes, the experiment reports effect-estimate stability and
continuous-treatment balance diagnostics rather than counterfactual RMSE.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class NhanesConfig:
    treatment: str = "LBDSTRSI"  # triglycerides in the processed table used by the original code
    outcome: str = "LBDGLUSI"    # fasting glucose, SI unit
    diabetes_status: str = "DIQ010"
    adult_age_col: str = "RIDAGEYR"
    adult_min_age: int = 20
    outlier_z: float = 3.0
    random_state: int = 2026


VARIABLE_LABELS: Dict[str, str] = {
    "LBDSTRSI": "Triglycerides",
    "LBDTRSI": "Triglycerides",
    "LBDGLUSI": "Fasting glucose",
    "DIQ010": "Diabetes status",
    "LBXSASSI": "Aspartate aminotransferase",
    "LBXSATSI": "Alanine aminotransferase",
    "LBXSGTSI": "Gamma glutamyl transferase",
    "RIDAGEYR": "Age",
    "RIAGENDR": "Sex",
    "RIDRETH3": "Race/ethnicity",
    "BMXBMI": "Body mass index",
    "DMDEDUC2": "Education",
}


DEFAULT_GCI_BIOCHEMICAL = ["LBXSASSI", "LBXSATSI", "LBXSGTSI"]
DEFAULT_CLINICAL = ["RIDAGEYR", "RIAGENDR", "RIDRETH3", "BMXBMI", "DMDEDUC2"]
DEFAULT_GCI_GUIDED = ["LBXSASSI", "LBXSATSI", "LBXSGTSI", "RIDAGEYR"]


def _var_column(summary: pd.DataFrame) -> str:
    """Return the column containing NHANES variable names.

    The original dataSummary.csv stores the variable names in the first column,
    but the header can be inherited from a variable name.  This helper keeps the
    code robust to that file format.
    """
    return summary.columns[0]


def load_variable_summary(path: Path | str) -> pd.DataFrame:
    summary = pd.read_csv(path)
    var_col = _var_column(summary)
    summary = summary.rename(columns={var_col: "variable"})
    return summary


def lab_variables_from_summary(summary: pd.DataFrame, available_columns: Iterable[str]) -> List[str]:
    available = set(available_columns)
    if "Tclass" not in summary.columns:
        return []
    lab_vars = summary.loc[summary["Tclass"].astype(str).str.upper().eq("LAB"), "variable"]
    return [v for v in lab_vars.astype(str).tolist() if v in available]


def read_nhanes_processed(path: Path | str, config: NhanesConfig) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Convert everything possible to numeric.  The processed table should already
    # be numeric, but this avoids failures if R writes numeric columns as strings.
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if config.adult_age_col in df.columns:
        df = df.loc[df[config.adult_age_col] >= config.adult_min_age].copy()
    return df.reset_index(drop=True)


def remove_treatment_outcome_outliers(df: pd.DataFrame, treatment: str, outcome: str, z_threshold: float) -> pd.DataFrame:
    out = df.copy()
    for col in [treatment, outcome]:
        values = out[col].astype(float)
        std = values.std(ddof=0)
        if std == 0 or not np.isfinite(std):
            continue
        z = (values - values.mean()) / std
        out = out.loc[z.abs() <= z_threshold].copy()
    return out.reset_index(drop=True)


def available_vars(vars_: Sequence[str], columns: Iterable[str], exclude: Iterable[str] = ()) -> List[str]:
    colset = set(columns)
    ex = set(exclude)
    return [v for v in vars_ if v in colset and v not in ex]


def build_adjustment_sets(df: pd.DataFrame, summary: pd.DataFrame, config: NhanesConfig) -> Dict[str, List[str]]:
    exclude = {config.treatment, config.outcome, "SEQN"}
    clinical = available_vars(DEFAULT_CLINICAL, df.columns, exclude)
    gci_bio = available_vars(DEFAULT_GCI_BIOCHEMICAL, df.columns, exclude)
    gci_guided = available_vars(DEFAULT_GCI_GUIDED, df.columns, exclude)

    lab_vars = [v for v in lab_variables_from_summary(summary, df.columns) if v not in exclude]
    # Keep all-laboratory adjustment identifiable in a small table.  Demographics
    # are not included here because this baseline represents the naive laboratory
    # indicator adjustment criticised by the reviewers.
    all_labs = sorted(set(lab_vars))

    all_available = sorted(
        v for v in df.columns
        if v not in exclude and pd.api.types.is_numeric_dtype(df[v])
    )

    return {
        "No adjustment": [],
        "Clinical covariates": clinical,
        "All laboratory indicators": all_labs,
        "All available covariates": all_available,
        "GCI biochemical confounders": gci_bio,
        "GCI-guided adjustment": gci_guided,
    }


def candidate_balance_covariates(df: pd.DataFrame, summary: pd.DataFrame, config: NhanesConfig) -> List[str]:
    exclude = {config.treatment, config.outcome, "SEQN"}
    lab_vars = lab_variables_from_summary(summary, df.columns)
    covars = DEFAULT_CLINICAL + lab_vars
    return sorted(set(v for v in covars if v in df.columns and v not in exclude))


def _make_nuisance_model(random_state: int) -> object:
    # Ridge residualization is used by default for this observational diagnostic.
    # It is deterministic, fast for the all-laboratory baseline, and avoids
    # turning the NHANES analysis into a predictive modeling benchmark.
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        Ridge(alpha=1.0),
    )


def crossfit_residuals(
    y: np.ndarray,
    x: np.ndarray | None,
    random_state: int,
    n_splits: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return residuals and out-of-fold predictions for y~x.

    If x has no columns, the residual is y centered by the sample mean.
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    if x is None or x.shape[1] == 0:
        pred = np.full_like(y, np.mean(y), dtype=float)
        return y - pred, pred

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    pred = np.zeros_like(y, dtype=float)
    base_model = _make_nuisance_model(random_state)
    for fold, (train_idx, test_idx) in enumerate(kf.split(x)):
        model = clone(base_model)
        model.fit(x[train_idx], y[train_idx])
        pred[test_idx] = model.predict(x[test_idx])
    return y - pred, pred


def partial_linear_effect(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    adjust_vars: Sequence[str],
    random_state: int,
    n_splits: int = 5,
    n_bootstrap: int = 1000,
) -> Dict[str, float]:
    """Estimate a residualized local linear effect for a continuous treatment.

    The reported standardized coefficient is the slope of standardized fasting
    glucose on standardized triglycerides after residualizing both variables on
    the specified adjustment set.  The raw-scale slope is obtained by multiplying
    the standardized coefficient by sd(Y)/sd(T).
    """
    cols = [treatment, outcome] + list(adjust_vars)
    work = df[cols].copy()
    # Median imputation is used only for robustness.  The supplied processed
    # table currently contains no missing values.
    for col in cols:
        if work[col].isna().any():
            work[col] = work[col].fillna(work[col].median())

    t_raw = work[treatment].to_numpy(dtype=float)
    y_raw = work[outcome].to_numpy(dtype=float)
    t_mean, t_std = float(np.mean(t_raw)), float(np.std(t_raw, ddof=0))
    y_mean, y_std = float(np.mean(y_raw)), float(np.std(y_raw, ddof=0))
    if t_std == 0 or y_std == 0:
        raise ValueError("Treatment or outcome has zero variance.")

    t = (t_raw - t_mean) / t_std
    y = (y_raw - y_mean) / y_std
    x = work[list(adjust_vars)].to_numpy(dtype=float) if adjust_vars else None

    t_res, _ = crossfit_residuals(t, x, random_state=random_state, n_splits=n_splits)
    y_res, _ = crossfit_residuals(y, x, random_state=random_state + 17, n_splits=n_splits)

    denom = float(np.dot(t_res, t_res))
    theta_std = float(np.dot(t_res, y_res) / denom)
    residual = y_res - theta_std * t_res
    sigma2 = float(np.dot(residual, residual) / max(len(y_res) - 2, 1))
    se_std = float(np.sqrt(sigma2 / denom))

    rng = np.random.default_rng(random_state + 101)
    boots = []
    n = len(t_res)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        denom_b = float(np.dot(t_res[idx], t_res[idx]))
        if denom_b <= 1e-12:
            continue
        boots.append(float(np.dot(t_res[idx], y_res[idx]) / denom_b))
    if boots:
        ci_low, ci_high = np.percentile(boots, [2.5, 97.5])
    else:
        ci_low, ci_high = theta_std - 1.96 * se_std, theta_std + 1.96 * se_std

    theta_raw = theta_std * y_std / t_std
    return {
        "n": int(n),
        "theta_std": theta_std,
        "se_std": se_std,
        "ci_low_std": float(ci_low),
        "ci_high_std": float(ci_high),
        "theta_raw": float(theta_raw),
        "ci_low_raw": float(ci_low * y_std / t_std),
        "ci_high_raw": float(ci_high * y_std / t_std),
        "treatment_sd": t_std,
        "outcome_sd": y_std,
        "partial_t_var": float(np.var(t_res)),
    }


def _safe_abs_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return np.nan
    a = a[mask]
    b = b[mask]
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(abs(np.corrcoef(a, b)[0, 1]))


def balance_diagnostics(
    df: pd.DataFrame,
    treatment: str,
    adjust_vars: Sequence[str],
    tracked_covariates: Sequence[str],
    random_state: int,
    n_splits: int = 5,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Compute continuous-treatment balance before and after adjustment.

    For a continuous treatment, the diagnostic is the absolute correlation
    between each tracked covariate and the treatment.  After adjustment, the
    treatment is residualized on the adjustment set.  Smaller correlations with
    the residualized treatment indicate better balance for the variables that the
    adjustment set explains.
    """
    tracked = [v for v in tracked_covariates if v in df.columns and v not in {treatment}]
    cols = [treatment] + list(adjust_vars) + tracked
    work = df[sorted(set(cols))].copy()
    for col in work.columns:
        if work[col].isna().any():
            work[col] = work[col].fillna(work[col].median())

    t = work[treatment].to_numpy(dtype=float)
    t = (t - np.mean(t)) / np.std(t, ddof=0)
    x = work[list(adjust_vars)].to_numpy(dtype=float) if adjust_vars else None
    t_res, _ = crossfit_residuals(t, x, random_state=random_state + 31, n_splits=n_splits)

    rows = []
    for var in tracked:
        z = work[var].to_numpy(dtype=float)
        if np.nanstd(z) == 0:
            continue
        z = (z - np.nanmean(z)) / np.nanstd(z)
        rows.append({
            "variable": var,
            "label": VARIABLE_LABELS.get(var, var),
            "abs_corr_before": _safe_abs_corr(t, z),
            "abs_corr_after": _safe_abs_corr(t_res, z),
            "in_adjustment_set": var in set(adjust_vars),
        })
    detail = pd.DataFrame(rows)
    if detail.empty:
        return {
            "tracked_covariates": 0,
            "mean_abs_corr_before": np.nan,
            "mean_abs_corr_after": np.nan,
            "max_abs_corr_before": np.nan,
            "max_abs_corr_after": np.nan,
            "balance_reduction_pct": np.nan,
        }, detail

    mean_before = float(detail["abs_corr_before"].mean())
    mean_after = float(detail["abs_corr_after"].mean())
    reduction = float((mean_before - mean_after) / mean_before * 100.0) if mean_before > 0 else np.nan
    summary = {
        "tracked_covariates": int(len(detail)),
        "mean_abs_corr_before": mean_before,
        "mean_abs_corr_after": mean_after,
        "max_abs_corr_before": float(detail["abs_corr_before"].max()),
        "max_abs_corr_after": float(detail["abs_corr_after"].max()),
        "balance_reduction_pct": reduction,
    }
    return summary, detail


def format_adjustment_vars(vars_: Sequence[str]) -> str:
    if not vars_:
        return "--"
    return ", ".join(vars_)
