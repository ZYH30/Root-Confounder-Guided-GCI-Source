# Uploading This Project to a New GitHub Repository

This guide describes how to upload the complete GCI code package to a new GitHub repository.

## 1. Create a new GitHub repository

Create a new empty repository on GitHub, for example:

```text
Root-Confounder-Guided-GCI
```

Do not initialize it with a README, license, or `.gitignore`, because this package already contains the required files.

## 2. Unzip the package locally

```bash
unzip GCI_public_code_package.zip
cd GCI-main
```

## 3. Initialize Git and connect to GitHub

Replace `<USER_OR_ORG>` and `<REPOSITORY>` with your account and repository name.

```bash
git init
git branch -M main
git remote add origin https://github.com/<USER_OR_ORG>/<REPOSITORY>.git
```

## 4. Check the Python scripts

```bash
python -m py_compile supplementary_experiments/gci_analysis/*.py supplementary_experiments/experiments/*.py
```

Some original scripts require R packages, LightGBM, Optuna, PyTorch, or notebook environments.  Those components should be checked in their corresponding environments.

## 5. Install lightweight dependencies for supplementary experiments

```bash
pip install -r requirements.txt
```

## 6. Optional: regenerate supplementary tables

```bash
python supplementary_experiments/experiments/01_make_adjustment_set_table.py
python supplementary_experiments/experiments/02_run_synthetic_adjustment_baselines.py
python supplementary_experiments/experiments/03_run_noise_robustness.py --n-repeats 5 --poly-degree 3 --ridge-alpha 1.0 --save-generated-data
python supplementary_experiments/experiments/04_run_nhanes_internal_baselines.py --n-bootstrap 1000 --n-splits 5
```

Generated tables are saved in:

```text
supplementary_experiments/results/tables/
```

## 7. Commit and push

```bash
git add .
git commit -m "Release code for root-confounder-guided GCI"
git push -u origin main
```

## Suggested repository description

```text
Code for Root-Confounder-Guided Causal Effect Estimation in Complex Covariate Systems.
```
