#!/usr/bin/env python3
"""Run NHANES internal-baseline diagnostics for the GCI manuscript.

This experiment does not use counterfactual outcomes because the NHANES
application is observational.  It compares several adjustment specifications for
continuous triglycerides and fasting glucose using a residualized local-linear
continuous-treatment estimator and balance diagnostics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from supplementary_experiments.gci_analysis.nhanes_utils import (  # noqa: E402
    NhanesConfig,
    VARIABLE_LABELS,
    balance_diagnostics,
    build_adjustment_sets,
    candidate_balance_covariates,
    format_adjustment_vars,
    load_variable_summary,
    partial_linear_effect,
    read_nhanes_processed,
    remove_treatment_outcome_outliers,
)


def latex_escape(text: str) -> str:
    return str(text).replace("_", r"\_").replace("%", r"\%")


def to_latex_table(df: pd.DataFrame, path: Path) -> None:
    display = df.copy()
    for col in display.columns:
        display[col] = display[col].map(lambda x: latex_escape(x) if isinstance(x, str) else x)
    latex = display.to_latex(index=False, escape=False, float_format=lambda x: f"{x:.3f}")
    path.write_text(latex, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="dataset/NhanesDataPro.csv")
    parser.add_argument("--summary", type=str, default="dataset/dataSummary.csv")
    parser.add_argument("--outdir", type=str, default="supplementary_experiments/results/tables")
    parser.add_argument("--treatment", type=str, default="LBDSTRSI")
    parser.add_argument("--outcome", type=str, default="LBDGLUSI")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--random-state", type=int, default=2026)
    args = parser.parse_args()

    data_path = ROOT / args.data
    summary_path = ROOT / args.summary
    outdir = ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    config = NhanesConfig(
        treatment=args.treatment,
        outcome=args.outcome,
        random_state=args.random_state,
    )
    df_raw = read_nhanes_processed(data_path, config)
    summary = load_variable_summary(summary_path)

    if config.treatment not in df_raw.columns:
        raise ValueError(f"Treatment variable {config.treatment} is not in {data_path}.")
    if config.outcome not in df_raw.columns:
        raise ValueError(f"Outcome variable {config.outcome} is not in {data_path}.")

    df = remove_treatment_outcome_outliers(
        df_raw,
        treatment=config.treatment,
        outcome=config.outcome,
        z_threshold=config.outlier_z,
    )
    adjustment_sets = build_adjustment_sets(df, summary, config)
    tracked_covariates = candidate_balance_covariates(df, summary, config)

    rows = []
    balance_rows = []
    for i, (name, vars_) in enumerate(adjustment_sets.items()):
        effect = partial_linear_effect(
            df=df,
            treatment=config.treatment,
            outcome=config.outcome,
            adjust_vars=vars_,
            random_state=config.random_state + 100 * i,
            n_splits=args.n_splits,
            n_bootstrap=args.n_bootstrap,
        )
        balance_summary, balance_detail = balance_diagnostics(
            df=df,
            treatment=config.treatment,
            adjust_vars=vars_,
            tracked_covariates=tracked_covariates,
            random_state=config.random_state + 200 * i,
            n_splits=args.n_splits,
        )
        row = {
            "Adjustment": name,
            "Set size": len(vars_),
            "Variables": format_adjustment_vars(vars_),
            "N": effect["n"],
            "Std. slope": effect["theta_std"],
            "Std. 95% CI low": effect["ci_low_std"],
            "Std. 95% CI high": effect["ci_high_std"],
            "Raw slope": effect["theta_raw"],
            "Raw 95% CI low": effect["ci_low_raw"],
            "Raw 95% CI high": effect["ci_high_raw"],
            "Mean abs. corr. before": balance_summary["mean_abs_corr_before"],
            "Mean abs. corr. after": balance_summary["mean_abs_corr_after"],
            "Max abs. corr. after": balance_summary["max_abs_corr_after"],
            "Balance reduction (%)": balance_summary["balance_reduction_pct"],
        }
        rows.append(row)
        if not balance_detail.empty:
            balance_detail.insert(0, "Adjustment", name)
            balance_rows.append(balance_detail)

    result = pd.DataFrame(rows)
    result.to_csv(outdir / "nhanes_internal_baselines.csv", index=False)
    if balance_rows:
        pd.concat(balance_rows, ignore_index=True).to_csv(outdir / "nhanes_balance_detail.csv", index=False)

    compact = result[[
        "Adjustment",
        "Set size",
        "Std. slope",
        "Std. 95% CI low",
        "Std. 95% CI high",
        "Raw slope",
        "Raw 95% CI low",
        "Raw 95% CI high",
        "Mean abs. corr. after",
        "Balance reduction (%)",
    ]].copy()
    compact.to_csv(outdir / "nhanes_internal_baselines_compact.csv", index=False)
    to_latex_table(compact, outdir / "table_nhanes_internal_baselines_compact.tex")

    set_table = pd.DataFrame([
        {
            "Adjustment": name,
            "Set size": len(vars_),
            "Variables": format_adjustment_vars(vars_),
        }
        for name, vars_ in adjustment_sets.items()
    ])
    set_table.to_csv(outdir / "nhanes_adjustment_sets.csv", index=False)
    to_latex_table(set_table, outdir / "table_nhanes_adjustment_sets.tex")

    metadata = {
        "data": str(data_path),
        "summary": str(summary_path),
        "original_n": int(len(df_raw)),
        "analysis_n_after_outlier_filter": int(len(df)),
        "treatment": config.treatment,
        "treatment_label": VARIABLE_LABELS.get(config.treatment, config.treatment),
        "outcome": config.outcome,
        "outcome_label": VARIABLE_LABELS.get(config.outcome, config.outcome),
        "diabetes_status_variable_present": config.diabetes_status in df_raw.columns,
        "n_splits": args.n_splits,
        "n_bootstrap": args.n_bootstrap,
        "random_state": args.random_state,
        "tracked_covariates": tracked_covariates,
        "interpretation_note": (
            "The NHANES experiment has no ground-truth counterfactual outcomes. "
            "The reported slopes are residualized observational effect estimates "
            "under each adjustment specification, complemented by continuous-treatment "
            "balance diagnostics."
        ),
    }
    (outdir / "nhanes_internal_baselines_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Saved NHANES internal-baseline results to", outdir)
    print(compact.to_string(index=False))


if __name__ == "__main__":
    main()
