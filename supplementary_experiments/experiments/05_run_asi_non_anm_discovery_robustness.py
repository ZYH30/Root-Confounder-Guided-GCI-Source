#!/usr/bin/env python3
"""Run ASI discovery diagnostics under ANM-violating mechanisms.

This experiment complements ``03_run_noise_robustness.py``.  The previous robustness
experiment used known graphs to isolate adjustment-set construction.  Here we rerun a
quiet ASI-style local discovery procedure on the same generated datasets and evaluate
whether outcome ancestors and root confounding sources can still be recovered when the
structural equations deviate from the additive-noise reference mechanism.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List

import pandas as pd

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
warnings.filterwarnings("ignore", message="X does not have valid feature names.*")

THIS_FILE = Path(__file__).resolve()
EXPERIMENT_ROOT = THIS_FILE.parents[1]
PROJECT_ROOT = THIS_FILE.parents[2]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from gci_analysis.asi_discovery import ASIDiscoveryConfig, binary_metrics, discover_case_structure  # noqa: E402
from gci_analysis.data_generators import ALL_ROBUSTNESS_MECHANISMS, MECHANISM_LABELS, GeneratorConfig, generate_case_dataset  # noqa: E402
from gci_analysis.graph_cases import CASE_GRAPHS, fmt_vars  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT_ROOT / "results" / "tables")
    parser.add_argument("--n-samples", type=int, default=1500)
    parser.add_argument("--n-repeats", type=int, default=10)
    parser.add_argument("--random-seed", type=int, default=2026)
    parser.add_argument("--cor-threshold", type=float, default=0.20)
    parser.add_argument("--ci-alpha", type=float, default=0.05)
    parser.add_argument("--direction-alpha", type=float, default=0.05)
    parser.add_argument("--root-source-alpha", type=float, default=0.02)
    parser.add_argument("--max-condition-set", type=int, default=3)
    parser.add_argument("--n-estimators", type=int, default=80)
    parser.add_argument("--min-samples-leaf", type=int, default=12)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--backend", choices=["lightgbm", "extra_trees"], default="lightgbm")
    parser.add_argument(
        "--force-orientation",
        action="store_true",
        help=(
            "Use the original strict fallback that orients ambiguous ANM pairs by p-value size. "
            "By default, ambiguous non-ANM directions are retained as upstream candidates."
        ),
    )
    parser.add_argument(
        "--mechanisms",
        nargs="+",
        default=list(ALL_ROBUSTNESS_MECHANISMS),
        choices=list(ALL_ROBUSTNESS_MECHANISMS),
    )
    return parser.parse_args()


def _true_outcome_ancestors(case_id: int) -> tuple[str, ...]:
    case = CASE_GRAPHS[case_id]
    return tuple(sorted(case.ancestors(case.outcome) & set(case.observed_covariates), key=_var_sort_key))


def _var_sort_key(x: str) -> tuple[int, str]:
    if x.startswith("X") and x[1:].isdigit():
        return (0, f"{int(x[1:]):04d}")
    return (1, x)


def _summarize(raw: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["accuracy", "precision", "recall", "f1", "auc"]
    summary = (
        raw.groupby(["Mechanism", "Mechanism key", "Target", "Case"], as_index=False)[metric_cols]
        .agg(["mean", "std"])
    )
    summary.columns = [
        "_".join(col).rstrip("_") if isinstance(col, tuple) else col for col in summary.columns.to_flat_index()
    ]
    return summary.rename(
        columns={
            "accuracy_mean": "Accuracy",
            "precision_mean": "Precision",
            "recall_mean": "Recall",
            "f1_mean": "F1",
            "auc_mean": "AUC",
            "accuracy_std": "Accuracy_sd",
            "precision_std": "Precision_sd",
            "recall_std": "Recall_sd",
            "f1_std": "F1_sd",
            "auc_std": "AUC_sd",
        }
    )


def _compact_table(summary: pd.DataFrame) -> pd.DataFrame:
    compact = (
        summary.groupby(["Mechanism", "Mechanism key", "Target"], as_index=False)[
            ["Accuracy", "Precision", "Recall", "F1", "AUC"]
        ]
        .mean()
    )
    mechanism_order = {MECHANISM_LABELS[m]: i for i, m in enumerate(ALL_ROBUSTNESS_MECHANISMS)}
    target_order = {"Outcome ancestors": 0, "Root sources": 1}
    compact["_mechanism_order"] = compact["Mechanism"].map(mechanism_order)
    compact["_target_order"] = compact["Target"].map(target_order)
    compact = compact.sort_values(["_mechanism_order", "_target_order"]).drop(
        columns=["_mechanism_order", "_target_order", "Mechanism key"]
    )
    for col in ["Accuracy", "Precision", "Recall", "F1", "AUC"]:
        compact[col] = compact[col].map(lambda x: f"{x:.3f}")
    return compact


def _write_latex_table(table: pd.DataFrame, path: Path) -> None:
    path.write_text(table.to_latex(index=False, escape=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    asi_config = ASIDiscoveryConfig(
        cor_threshold=args.cor_threshold,
        ci_alpha=args.ci_alpha,
        direction_alpha=args.direction_alpha,
        root_source_alpha=args.root_source_alpha,
        max_condition_set=args.max_condition_set,
        n_estimators=args.n_estimators,
        min_samples_leaf=args.min_samples_leaf,
        n_splits=args.n_splits,
        random_seed=args.random_seed,
        backend=args.backend,
        ambiguous_as_parent=not args.force_orientation,
    )

    rows: List[Dict[str, object]] = []
    detail_rows: List[Dict[str, object]] = []
    for mechanism in args.mechanisms:
        mechanism_label = MECHANISM_LABELS[mechanism]
        for case_id, case in CASE_GRAPHS.items():
            true_ancestors = _true_outcome_ancestors(case_id)
            true_roots = tuple(case.root_sources)
            universe = tuple(case.observed_covariates)
            for repeat in range(args.n_repeats):
                gen_config = GeneratorConfig(
                    n_samples=args.n_samples,
                    random_seed=args.random_seed + 7919 * repeat,
                )
                df = generate_case_dataset(case_id, mechanism, gen_config)
                result = discover_case_structure(
                    df=df,
                    observed_covariates=case.observed_covariates,
                    treatment=case.treatment,
                    outcome=case.outcome,
                    config=asi_config,
                )

                for target, true_set, pred_set in [
                    ("Outcome ancestors", true_ancestors, result.predicted_outcome_ancestors),
                    ("Root sources", true_roots, result.predicted_root_sources),
                ]:
                    metrics = binary_metrics(true_set, pred_set, universe)
                    rows.append(
                        {
                            "Mechanism": mechanism_label,
                            "Mechanism key": mechanism,
                            "Case": f"Case {case_id}",
                            "case_id": case_id,
                            "repeat": repeat,
                            "Target": target,
                            "True set": fmt_vars(true_set),
                            "Predicted set": fmt_vars(pred_set),
                            **metrics,
                        }
                    )

                detail_rows.append(
                    {
                        "Mechanism": mechanism_label,
                        "Mechanism key": mechanism,
                        "Case": f"Case {case_id}",
                        "case_id": case_id,
                        "repeat": repeat,
                        "searched_targets": fmt_vars(result.searched_targets),
                        "predicted_outcome_ancestors": fmt_vars(result.predicted_outcome_ancestors),
                        "predicted_treatment_ancestors": fmt_vars(result.predicted_treatment_ancestors),
                        "predicted_common_ancestors": fmt_vars(result.predicted_common_ancestors),
                        "predicted_root_sources": fmt_vars(result.predicted_root_sources),
                        "root_source_candidate_pvalues": json.dumps(
                            {k: round(v, 6) for k, v in result.root_source_candidate_pvalues.items()},
                            sort_keys=True,
                        ),
                        "parent_dict": json.dumps(result.parent_dict, sort_keys=True),
                    }
                )
                print(
                    f"Finished {mechanism_label}, Case {case_id}, repeat {repeat + 1}/{args.n_repeats}",
                    flush=True,
                )

    raw = pd.DataFrame(rows)
    details = pd.DataFrame(detail_rows)
    summary = _summarize(raw)
    compact = _compact_table(summary)

    raw_path = args.output_dir / "asi_non_anm_discovery_raw.csv"
    detail_path = args.output_dir / "asi_non_anm_discovery_details.csv"
    summary_path = args.output_dir / "asi_non_anm_discovery_summary.csv"
    compact_path = args.output_dir / "asi_non_anm_discovery_compact.csv"
    compact_tex_path = args.output_dir / "table_asi_non_anm_discovery_compact.tex"
    config_path = args.output_dir / "asi_non_anm_discovery_config.json"

    raw.to_csv(raw_path, index=False)
    details.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    compact.to_csv(compact_path, index=False)
    _write_latex_table(compact, compact_tex_path)

    config_path.write_text(
        json.dumps(
            {
                "generator": {"n_samples": args.n_samples, "n_repeats": args.n_repeats},
                "asi_discovery": asi_config.__dict__,
                "mechanisms": args.mechanisms,
                "note": (
                    "This is a supplementary ASI-style discovery diagnostic using marginal "
                    "screening, residual conditional-independence testing, and ANM-style "
                    "directional residual testing under alternative structural mechanisms. "
                    "When ambiguous_as_parent is true, variable pairs for which both ANM "
                    "directions are accepted are retained as upstream candidates instead "
                    "of being forcibly oriented by small p-value differences."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Saved: {raw_path}")
    print(f"Saved: {detail_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {compact_path}")
    print(f"Saved: {compact_tex_path}")
    print(f"Saved: {config_path}")
    print("\nCompact ASI discovery results averaged across four DAG cases:")
    print(compact.to_string(index=False))


if __name__ == "__main__":
    main()
