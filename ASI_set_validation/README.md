# Final adjustment-set validation for GCI

This module accompanies **Root-Confounder-Guided Causal Effect Estimation in Complex Covariate Systems**. It evaluates three questions: recovery and graphical validity of the final adjustment set; downstream effect-estimation error; and the consequences of limiting the local CI search.

The implementation retains marginal screening, conditional-independence pruning, parent recursion, root-closure construction, and graph-based adjustment. GCM uses five-fold spline residuals, ANM orientation uses paired residual HSIC scores, and a local CI safeguard retains neighbors when the local dependence evidence conflicts with removing them as children. `configs/final_settings.json` specifies the final settings. The release exposes a single ASI configuration and its prescribed conditioning-order sensitivity settings.

## Installation

Python 3.13.5 was used for the archived results. Python 3.11 or newer is required. For numerical reproduction, use the pinned dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e . --no-deps
```

The module is self-contained and requires only the pinned Python packages; no clinical data, external service, or GPU is needed. It can be placed in a new subdirectory of the article's repository without replacing existing code.

## Read the supplied results without rerunning discovery

```bash
python -m gci_validation.summarize --input results --output reproduced_tables
python scripts/verify_release.py
pytest -q
```

`results/validation/` contains 160 fixed-setting runs (four DAGs × four mechanisms × ten repeats). `results/order_sensitivity/` contains 240 runs on a separate matched 80-dataset sample, evaluated at maximum CI orders 2, 3, and 4. These are 400 method/data-setting runs, not 400 independent datasets. The order analysis is descriptive; its samples were available before the final validation configuration was locked. The independent 160-dataset validation uses different seeds and is the primary final-set evaluation.

The saved tables are generated from individual records. Predictions are included, allowing all 880 reported RMSE values to be independently recalculated. No graph/seed/mechanism failure has been omitted. All 160 primary constructions returned a set; 131 satisfy the true-graph criterion.

## Reproduce the prescribed analyses

An empty output directory is required to avoid overwriting recorded results. All simulation data are generated locally from the fixed coefficients and seeds.

```bash
# One complete dataset as a smoke test
python -m gci_validation.run --analysis validation --case 1 \
  --mechanism additive_sine --repeats 1 --output reproduced_smoke

# Primary validation: 160 datasets, fixed maximum order 3
python -m gci_validation.run --analysis validation --output reproduced

# Matched conditioning-order analysis: 80 datasets at each of orders 2, 3, 4
python -m gci_validation.run --analysis order_sensitivity --output reproduced

# Summarize the reproduced results
python -m gci_validation.summarize --input reproduced --output reproduced/tables
```

The runner is serial by default. For multiple processes, partition jobs by `--case` or `--mechanism` and use separate output directories. Avoid oversubscribing BLAS threads; the entry point sets single-threaded numerical libraries. Structural coefficients, learning thresholds, estimator penalties, and seed rules must remain fixed to reproduce the reported tables.

## Table correspondence

| Result file in `results/tables/` | Manuscript content |
|---|---|
| `final_set_recovery.csv` | Final-set P/R/F1, true-graph validity, size, and error categories (Table V) |
| `downstream_effects.csv` | Mechanism-specific effect error (Table VI) |
| `graph_specific_structure.csv`, `graph_specific_effects.csv` | All 16 mechanism/DAG combinations (Table VII) |
| `conditioning_order_sensitivity.csv` | Maximum CI conditioning-order analysis (Table VIII) |
| `individual_structure_metrics.csv`, `individual_effect_metrics.csv` | Individual runs underlying the aggregate tables |
| `verification.json` | Counts and prediction-based RMSE verification |

The table numbers refer to the full-text revision draft. The correspondence by caption remains valid if tables are moved in a later pagination pass.

## Interpretation

Final-set P/R/F1 use the representative set returned by the same deterministic constructor on the true DAG. A different valid alternative need not match that reference. True-graph validity is therefore a separate metric: no true treatment descendant may be included, and treatment/outcome must be d-separated in the true back-door graph. Failed constructions remain in the denominator. Exact matching to one representative set is not a primary reported endpoint.

The common outcome estimator is standardized degree-three polynomial ridge regression with penalty 1.0. It is fitted on 1,500 discovery/training samples and evaluated on 1,500 independent samples. The effect target is the generator's individual unit finite difference under treatment and treatment minus one, with descendants regenerated and exogenous noise held fixed. This is not a population-average treatment-effect error or an interval-coverage experiment.

True graph, true roles, counterfactual labels, and test observations are not inputs to ASI. The true graph is used only for evaluation and for the separately labeled true-graph GCI reference. `docs/METHOD_SETTINGS.md` records the numerical rules; `NOTICE.md` and `licenses/` preserve source attribution.

## Release verification

The release passed 23 unit tests. All 880 saved prediction errors were recalculated without discrepancy. One primary-validation repeat was also replayed for every mechanism/DAG cell: all 16 parent dictionaries, selected sets, CI-call counts, and prediction arrays matched the archived fixed-setting outputs. `docs/REPRODUCIBILITY_CHECK.json` records the checks. These replay checks verify packaging, not additional independent scientific evidence.
