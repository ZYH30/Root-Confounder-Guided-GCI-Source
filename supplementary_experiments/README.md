# Supplementary Experiments

This directory contains supplementary experimental scripts and result tables for the GCI project.  The scripts focus on adjustment-set construction, ASI discovery robustness under alternative structural mechanisms, and NHANES internal-baseline diagnostics.

## Directory Layout

| Path | Description |
|---|---|
| `gci_analysis/graph_cases.py` | Encodes graph roles and adjustment sets for the four controlled synthetic DAGs. |
| `gci_analysis/estimators.py` | Legacy repeated-split k-nearest-neighbor diagnostic. |
| `gci_analysis/fast_estimators.py` | Smooth polynomial-ridge estimator used for Table III and robustness diagnostics. |
| `gci_analysis/asi_discovery.py` | Quiet ASI-style discovery diagnostic used for non-ANM robustness checks. |
| `gci_analysis/data_generators.py` | Synthetic generators for additive-reference, multiplicative, heteroscedastic, and interaction-based mechanisms. |
| `gci_analysis/nhanes_utils.py` | Processed NHANES loading, adjustment-set construction, residualized slope estimation, and balance diagnostics. |
| `experiments/` | Executable scripts for generating tables. |
| `results/tables/` | Generated CSV and LaTeX tables. |
| `results/generated_data/` | Synthetic datasets generated for robustness diagnostics. |

## Running the Experiments

Run all commands from the repository root.

```bash
python supplementary_experiments/experiments/01_make_adjustment_set_table.py
python supplementary_experiments/experiments/02_run_synthetic_adjustment_baselines.py
python supplementary_experiments/experiments/03_run_noise_robustness.py --n-repeats 5 --poly-degree 3 --ridge-alpha 1.0 --save-generated-data
python supplementary_experiments/experiments/05_run_asi_non_anm_discovery_robustness.py --n-repeats 10 --n-samples 1500 --n-estimators 80
python supplementary_experiments/experiments/04_run_nhanes_internal_baselines.py --n-bootstrap 1000 --n-splits 5
```

## Outputs

The main tables are saved in:

```text
supplementary_experiments/results/tables/
```

Generated robustness datasets are saved in:

```text
supplementary_experiments/results/generated_data/
```

The JSON files in `results/tables/` record the main run parameters for each experiment.

The ASI non-ANM discovery diagnostic writes:

```text
asi_non_anm_discovery_compact.csv
table_asi_non_anm_discovery_compact.tex
asi_non_anm_discovery_details.csv
asi_non_anm_discovery_config.json
```
