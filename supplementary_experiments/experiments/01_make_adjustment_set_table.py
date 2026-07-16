#!/usr/bin/env python3
"""Create the graph-role and GCI adjustment-set table for the four DAGs."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

THIS_FILE = Path(__file__).resolve()
EXPERIMENT_ROOT = THIS_FILE.parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from gci_analysis.graph_cases import CASE_GRAPHS, fmt_vars  # noqa: E402


def build_table() -> pd.DataFrame:
    rows = []
    for case_id, case in CASE_GRAPHS.items():
        rows.append(
            {
                "Case": f"Case {case_id}",
                "Scenario": case.label,
                "Root sources": fmt_vars(case.root_sources),
                "Treatment parents": fmt_vars(case.observed_covariate_parents(case.treatment)),
                "Outcome parents": fmt_vars(case.observed_covariate_parents(case.outcome)),
                "GCI-guided set": fmt_vars(case.gci_valid_adjustment),
                "Key excluded variables": fmt_vars(case.excluded_by_gci),
                "Notes": case.note,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EXPERIMENT_ROOT / "results" / "tables",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    table = build_table()
    csv_path = args.output_dir / "table_adjustment_set_relationship.csv"
    tex_path = args.output_dir / "table_adjustment_set_relationship.tex"
    table.to_csv(csv_path, index=False)
    table.to_latex(tex_path, index=False, escape=True)

    print(f"Saved: {csv_path}")
    print(f"Saved: {tex_path}")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
