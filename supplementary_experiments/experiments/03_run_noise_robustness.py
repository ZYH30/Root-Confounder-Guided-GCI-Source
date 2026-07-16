#!/usr/bin/env python3
"""Run ANM-violating robustness experiments for adjustment-set comparisons.

This script generates additional synthetic datasets on the original four DAGs but uses
structural mechanisms that deviate from the additive-noise form. It does not rerun ASI.
The experiment tests whether the root-confounder-guided adjustment principle
remains useful when the data-generating equations are multiplicative, heteroscedastic,
or interaction-based.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

# Limit numerical backend threads before importing sklearn-dependent analysis modules.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

THIS_FILE = Path(__file__).resolve()
EXPERIMENT_ROOT = THIS_FILE.parents[1]
PROJECT_ROOT = THIS_FILE.parents[2]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from gci_analysis.data_generators import (  # noqa: E402
    ALL_ROBUSTNESS_MECHANISMS,
    ANM_VIOLATING_MECHANISMS,
    MECHANISM_LABELS,
    GeneratorConfig,
    generate_case_dataset,
)
from gci_analysis.fast_estimators import FastEvaluationConfig, evaluate_adjustment_set_fast, summarize_fast_runs  # noqa: E402
from gci_analysis.graph_cases import ADJUSTMENT_SET_LABELS, adjustment_sets_for_case, fmt_vars  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT_ROOT / "results" / "tables")
    parser.add_argument("--generated-data-dir", type=Path, default=EXPERIMENT_ROOT / "results" / "generated_data")
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--n-repeats", type=int, default=10)
    parser.add_argument("--poly-degree", type=int, default=3)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--random-seed", type=int, default=2026)
    parser.add_argument(
        "--mechanisms",
        nargs="+",
        default=list(ALL_ROBUSTNESS_MECHANISMS),
        choices=list(ALL_ROBUSTNESS_MECHANISMS),
        help="Mechanisms to generate. Defaults include one additive reference and three ANM-violating mechanisms.",
    )
    parser.add_argument(
        "--save-generated-data",
        action="store_true",
        help="Save generated robustness datasets under supplementary_experiments/results/generated_data/.",
    )
    return parser.parse_args()


def _make_formatted_table(summary: pd.DataFrame, mechanisms: List[str]) -> pd.DataFrame:
    test_only = summary[summary["Split"] == "test"].copy()
    test_only["RMSE_MTEF"] = test_only.apply(
        lambda r: f"{r['RMSE_MTEF_mean']:.3f} $\\pm$ {r['RMSE_MTEF_sd']:.3f}", axis=1
    )
    display_cols = [
        "Mechanism",
        "Case",
        "Adjustment strategy",
        "Adjustment variables",
        "Set size",
        "RMSE_MTEF",
        "Bias_MTEF_mean",
        "MAE_MTEF_mean",
    ]
    display = test_only[display_cols].copy()
    display["Bias_MTEF_mean"] = display["Bias_MTEF_mean"].map(lambda x: f"{x:.3f}")
    display["MAE_MTEF_mean"] = display["MAE_MTEF_mean"].map(lambda x: f"{x:.3f}")
    mechanism_order = {MECHANISM_LABELS[m]: i for i, m in enumerate(mechanisms)}
    strategy_order = {label: i for i, label in enumerate(ADJUSTMENT_SET_LABELS)}
    display["_mechanism_order"] = display["Mechanism"].map(mechanism_order)
    display["_strategy_order"] = display["Adjustment strategy"].map(strategy_order)
    display["_case_order"] = display["Case"].str.extract(r"(\d+)").astype(int)
    display = display.sort_values(["_mechanism_order", "_case_order", "_strategy_order"]).drop(
        columns=["_mechanism_order", "_strategy_order", "_case_order"]
    )
    return display


def _make_compact_mean_table(summary: pd.DataFrame, mechanisms: List[str]) -> pd.DataFrame:
    """Create a manuscript-friendly table averaged across the four DAG cases."""
    selected = [
        "No adjustment",
        "All covariates",
        "Root-only",
        "Treatment-parent",
        "Outcome-parent",
        "GCI-guided valid",
    ]
    test_only = summary[(summary["Split"] == "test") & (summary["Adjustment strategy"].isin(selected))].copy()
    compact = (
        test_only.groupby(["Mechanism", "Adjustment strategy"], as_index=False)
        .agg(
            Mean_RMSE_MTEF=("RMSE_MTEF_mean", "mean"),
            Mean_MAE_MTEF=("MAE_MTEF_mean", "mean"),
            Mean_abs_bias=("Bias_MTEF_mean", lambda x: float(x.abs().mean())),
        )
    )
    compact["RMSE_MTEF"] = compact["Mean_RMSE_MTEF"].map(lambda x: f"{x:.3f}")
    compact["MAE_MTEF"] = compact["Mean_MAE_MTEF"].map(lambda x: f"{x:.3f}")
    compact["Abs. Bias"] = compact["Mean_abs_bias"].map(lambda x: f"{x:.3f}")
    compact = compact[["Mechanism", "Adjustment strategy", "RMSE_MTEF", "MAE_MTEF", "Abs. Bias"]]
    mechanism_order = {MECHANISM_LABELS[m]: i for i, m in enumerate(mechanisms)}
    strategy_order = {label: i for i, label in enumerate(selected)}
    compact["_mechanism_order"] = compact["Mechanism"].map(mechanism_order)
    compact["_strategy_order"] = compact["Adjustment strategy"].map(strategy_order)
    compact = compact.sort_values(["_mechanism_order", "_strategy_order"]).drop(
        columns=["_mechanism_order", "_strategy_order"]
    )
    return compact


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_generated_data:
        args.generated_data_dir.mkdir(parents=True, exist_ok=True)

    generator_config = GeneratorConfig(
        n_samples=args.n_samples,
        random_seed=args.random_seed,
    )
    eval_config = FastEvaluationConfig(
        n_repeats=args.n_repeats,
        random_seed=args.random_seed,
        poly_degree=args.poly_degree,
        ridge_alpha=args.ridge_alpha,
    )

    raw_frames: List[pd.DataFrame] = []
    summary_rows: List[Dict[str, object]] = []

    for mechanism in args.mechanisms:
        mechanism_label = MECHANISM_LABELS[mechanism]
        for case_id in range(1, 5):
            df = generate_case_dataset(case_id, mechanism, generator_config)
            if args.save_generated_data:
                out_data = args.generated_data_dir / f"data-G-Case{case_id}-{mechanism}.csv"
                df.to_csv(out_data, index=False)

            set_map = adjustment_sets_for_case(case_id)
            for label in ADJUSTMENT_SET_LABELS:
                adjustment_vars = set_map[label]
                run_df = evaluate_adjustment_set_fast(df, adjustment_vars, eval_config)
                run_df.insert(0, "mechanism", mechanism)
                run_df.insert(1, "mechanism_label", mechanism_label)
                run_df.insert(2, "case_id", case_id)
                run_df.insert(3, "adjustment_strategy", label)
                run_df.insert(4, "adjustment_variables", fmt_vars(adjustment_vars))
                run_df.insert(5, "adjustment_set_size", len(adjustment_vars))
                raw_frames.append(run_df)

                summary = summarize_fast_runs(run_df)
                for _, row in summary.iterrows():
                    summary_rows.append(
                        {
                            "Mechanism": mechanism_label,
                            "Mechanism key": mechanism,
                            "Case": f"Case {case_id}",
                            "Adjustment strategy": label,
                            "Adjustment variables": fmt_vars(adjustment_vars),
                            "Set size": len(adjustment_vars),
                            "Split": row["split"],
                            "RMSE_MTEF_mean": row["rmse_mtef_mean"],
                            "RMSE_MTEF_sd": row["rmse_mtef_sd"],
                            "Bias_MTEF_mean": row["bias_mtef_mean"],
                            "Bias_MTEF_sd": row["bias_mtef_sd"],
                            "MAE_MTEF_mean": row["mae_mtef_mean"],
                            "RMSE_factual_mean": row["rmse_factual_mean"],
                        }
                    )
                print(
                    f"Finished {mechanism_label}, Case {case_id}: {label} ({fmt_vars(adjustment_vars)})",
                    flush=True,
                )

    raw = pd.concat(raw_frames, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    for col in [
        "RMSE_MTEF_mean",
        "RMSE_MTEF_sd",
        "Bias_MTEF_mean",
        "Bias_MTEF_sd",
        "MAE_MTEF_mean",
        "RMSE_factual_mean",
    ]:
        summary[col] = summary[col].astype(float)

    raw_path = args.output_dir / "noise_robustness_raw.csv"
    summary_path = args.output_dir / "noise_robustness_summary.csv"
    detailed_test_path = args.output_dir / "noise_robustness_test_only.csv"
    detailed_tex_path = args.output_dir / "table_noise_robustness_test_only.tex"
    compact_path = args.output_dir / "noise_robustness_compact_mean.csv"
    compact_tex_path = args.output_dir / "table_noise_robustness_compact_mean.tex"
    config_path = args.output_dir / "noise_robustness_config.json"

    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)

    test_display = _make_formatted_table(summary, args.mechanisms)
    test_display.to_csv(detailed_test_path, index=False)
    test_display.to_latex(detailed_tex_path, index=False, escape=False)

    compact = _make_compact_mean_table(summary, args.mechanisms)
    compact.to_csv(compact_path, index=False)
    compact.to_latex(compact_tex_path, index=False, escape=False)

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "generator_config": generator_config.__dict__,
                "evaluation_config": eval_config.__dict__,
                "mechanisms": list(args.mechanisms),
                "mechanism_labels": {m: MECHANISM_LABELS[m] for m in args.mechanisms},
                "note": "ASI is not rerun. Robustness datasets are generated on the original DAGs with non-additive mechanisms.",
            },
            f,
            indent=2,
        )

    print(f"Saved: {raw_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {detailed_test_path}")
    print(f"Saved: {detailed_tex_path}")
    print(f"Saved: {compact_path}")
    print(f"Saved: {compact_tex_path}")
    print(f"Saved: {config_path}")
    print("\nCompact mean test results across four cases:")
    print(compact.to_string(index=False))


if __name__ == "__main__":
    main()
