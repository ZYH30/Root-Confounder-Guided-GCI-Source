# Table III GitHub Update

This update synchronizes the public code with the revised graph-role definitions and
the controlled adjustment-specification experiment reported in Table III.

## Replace

- `supplementary_experiments/gci_analysis/graph_cases.py`
- `supplementary_experiments/experiments/01_make_adjustment_set_table.py`
- `supplementary_experiments/experiments/03_run_noise_robustness.py`
- `supplementary_experiments/README.md`
- `README.md`
- `GITHUB_UPLOAD_GUIDE.md`
- `requirements.txt`

## Delete and add

Delete the legacy script:

- `supplementary_experiments/experiments/02_run_existing_synthetic_adjustment_baselines.py`

Add its replacement:

- `supplementary_experiments/experiments/02_run_synthetic_adjustment_baselines.py`

The replacement does not read the legacy `dataset/data-G-Case*.csv` counterfactual
columns. It generates factual and counterfactual outcomes from the same SCM, changes
only the treatment intervention, and recomputes treatment descendants with shared
node-level noise.

## Remove legacy Table III results

- `existing_synthetic_adjustment_baselines_config.json`
- `existing_synthetic_adjustment_baselines_raw.csv`
- `existing_synthetic_adjustment_baselines_summary.csv`
- `existing_synthetic_adjustment_baselines_test_only.csv`
- `table_existing_synthetic_adjustment_baselines_test_only.tex`

## Add regenerated Table III results

- `synthetic_adjustment_baselines_config.json`
- `synthetic_adjustment_baselines_raw.csv`
- `synthetic_adjustment_baselines_summary.csv`
- `synthetic_adjustment_baselines_table.csv`
- `table_synthetic_adjustment_baselines.tex`

All result files are under `supplementary_experiments/results/tables/`.

## Run

From the repository root:

```bash
pip install -r requirements.txt
python supplementary_experiments/experiments/01_make_adjustment_set_table.py
python supplementary_experiments/experiments/02_run_synthetic_adjustment_baselines.py
```

The default Table III configuration is:

- 1,500 observations per dataset;
- 10 independently generated datasets;
- 10 80/20 train--test splits per dataset;
- treatment perturbation `delta_t = 1`;
- nonroot structural-noise scale `0.5`;
- third-degree polynomial-ridge outcome model;
- ridge parameter `alpha = 1`;
- base random seed `2026`.

## Important consistency note

Do not combine the new Table III values with results produced by the legacy KNN script
or the legacy `y_delta` columns. All six specifications in Table III must be generated
by the new script under the same SCM, estimator, datasets, and splits.
