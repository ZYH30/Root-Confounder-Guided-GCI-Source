"""Lightweight ASI-style local discovery diagnostics.

This module implements a quiet, dependency-light version of the ASI search logic for
supplementary robustness diagnostics.  It keeps the same operational stages used in
the manuscript--marginal screening, conditional residual-independence testing, and
ANM-style directional residual testing--without importing the original script-style
``AnsFramePlus.py`` file, which executes an experiment at import time.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold


@dataclass(frozen=True)
class ASIDiscoveryConfig:
    cor_threshold: float = 0.20
    ci_alpha: float = 0.05
    direction_alpha: float = 0.05
    root_source_alpha: float = 0.02
    max_condition_set: int = 2
    n_estimators: int = 80
    min_samples_leaf: int = 12
    n_splits: int = 3
    random_seed: int = 2026
    backend: str = "lightgbm"
    ambiguous_as_parent: bool = True


@dataclass
class DiscoveryResult:
    parent_dict: Dict[str, Tuple[str, ...]]
    child_dict: Dict[str, Tuple[str, ...]]
    searched_targets: Tuple[str, ...]
    predicted_outcome_ancestors: Tuple[str, ...]
    predicted_treatment_ancestors: Tuple[str, ...]
    predicted_common_ancestors: Tuple[str, ...]
    predicted_root_sources: Tuple[str, ...]
    root_source_candidate_pvalues: Dict[str, float]


def _var_sort_key(x: str) -> Tuple[int, str]:
    if x.startswith("X") and x[1:].isdigit():
        return (0, f"{int(x[1:]):04d}")
    if x == "t":
        return (1, x)
    if x == "y":
        return (2, x)
    return (3, x)


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    corr, p_value = spearmanr(np.asarray(x).ravel(), np.asarray(y).ravel())
    if np.isnan(corr):
        return 0.0, 1.0
    if np.isnan(p_value):
        p_value = 1.0
    return float(corr), float(p_value)


def _gcm_pvalue(resid_x: np.ndarray, resid_y: np.ndarray) -> float:
    resid_x = np.asarray(resid_x, dtype=float).ravel()
    resid_y = np.asarray(resid_y, dtype=float).ravel()
    resid_x = resid_x - np.mean(resid_x)
    resid_y = resid_y - np.mean(resid_y)
    product = resid_x * resid_y
    mean_product = float(np.mean(product))
    variance = float(np.mean(product**2) - mean_product**2)
    if variance <= 1e-14:
        return 1.0
    statistic = np.sqrt(len(product)) * mean_product / np.sqrt(variance)
    return float(2 * norm.sf(abs(statistic)))


def _crossfit_residual(
    y: np.ndarray,
    z: pd.DataFrame,
    config: ASIDiscoveryConfig,
    seed_offset: int = 0,
) -> np.ndarray:
    y = np.asarray(y, dtype=float).ravel()
    if z is None or z.shape[1] == 0:
        return y - np.mean(y)

    if config.backend == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
            raise ImportError(
                "LightGBM backend requested but lightgbm is not installed. "
                "Install lightgbm or use backend='extra_trees'."
            ) from exc
        model = LGBMRegressor(
            n_estimators=max(config.n_estimators, 50),
            learning_rate=0.1,
            max_depth=5,
            num_leaves=31,
            min_child_samples=max(config.min_samples_leaf, 5),
            subsample=0.9,
            random_state=config.random_seed + seed_offset,
            verbosity=-1,
            n_jobs=1,
        )
        x_frame = z.astype(float)
        model.fit(x_frame, y)
        return y - model.predict(x_frame)

    if config.backend != "extra_trees":
        raise ValueError(f"Unknown ASI residual backend: {config.backend}")

    x = z.to_numpy(dtype=float)
    residual = np.zeros_like(y, dtype=float)
    n_splits = min(config.n_splits, len(y))
    if n_splits < 2:
        return y - np.mean(y)

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=config.random_seed + seed_offset)
    for train_idx, test_idx in kfold.split(x):
        model = ExtraTreesRegressor(
            n_estimators=config.n_estimators,
            min_samples_leaf=config.min_samples_leaf,
            max_features=1.0,
            random_state=config.random_seed + seed_offset,
            n_jobs=1,
        )
        model.fit(x[train_idx], y[train_idx])
        residual[test_idx] = y[test_idx] - model.predict(x[test_idx])
    return residual


def residual_independence_pvalue(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    cond_cols: Sequence[str],
    config: ASIDiscoveryConfig,
    seed_offset: int = 0,
) -> float:
    cond_cols = list(cond_cols)
    if not cond_cols:
        _, p_value = _safe_spearman(df[x_col].to_numpy(), df[y_col].to_numpy())
        return p_value
    resid_x = _crossfit_residual(df[x_col].to_numpy(), df[cond_cols], config, seed_offset)
    resid_y = _crossfit_residual(df[y_col].to_numpy(), df[cond_cols], config, seed_offset + 101)
    return _gcm_pvalue(resid_x, resid_y)


def anm_direction_pvalues(
    df: pd.DataFrame,
    cause_col: str,
    effect_col: str,
    config: ASIDiscoveryConfig,
    seed_offset: int = 0,
) -> Tuple[float, float]:
    """Return ANM residual-correlation p-values for cause->effect and effect->cause."""
    resid_effect = _crossfit_residual(
        df[effect_col].to_numpy(), df[[cause_col]], config, seed_offset=seed_offset
    )
    _, p_cause_to_effect = _safe_spearman(df[cause_col].to_numpy(), resid_effect)

    resid_cause = _crossfit_residual(
        df[cause_col].to_numpy(), df[[effect_col]], config, seed_offset=seed_offset + 307
    )
    _, p_effect_to_cause = _safe_spearman(df[effect_col].to_numpy(), resid_cause)
    return p_cause_to_effect, p_effect_to_cause


def identify_parent_child(
    df: pd.DataFrame,
    target: str,
    candidates: Sequence[str],
    excluded: Iterable[str],
    config: ASIDiscoveryConfig,
) -> Tuple[Tuple[str, ...], Tuple[str, ...], Tuple[str, ...]]:
    excluded_set = set(excluded) | {target}
    usable = [c for c in candidates if c not in excluded_set and c in df.columns]

    screened: List[str] = []
    for col in usable:
        corr, _ = _safe_spearman(df[target].to_numpy(), df[col].to_numpy())
        if abs(corr) >= config.cor_threshold:
            screened.append(col)
    screened.sort(key=lambda c: abs(_safe_spearman(df[target].to_numpy(), df[c].to_numpy())[0]), reverse=True)

    adjacent: List[str] = list(screened)
    removed: List[str] = []
    for col in list(screened):
        other = [c for c in adjacent if c != col]
        independent = False
        max_size = min(config.max_condition_set, len(other))
        for size in range(1, max_size + 1):
            for cond in combinations(other, size):
                p_value = residual_independence_pvalue(df, target, col, cond, config)
                if p_value > config.ci_alpha:
                    independent = True
                    break
            if independent:
                break
        if independent and col in adjacent:
            adjacent.remove(col)
            removed.append(col)

    parents: List[str] = []
    children: List[str] = []
    for col in adjacent:
        p_target_to_col, p_col_to_target = anm_direction_pvalues(df, target, col, config)
        target_to_col_supported = p_target_to_col >= config.direction_alpha
        col_to_target_supported = p_col_to_target >= config.direction_alpha
        if config.ambiguous_as_parent and target_to_col_supported and col_to_target_supported:
            parents.append(col)
        elif p_target_to_col > config.direction_alpha and p_col_to_target <= config.direction_alpha:
            children.append(col)
        elif p_target_to_col <= config.direction_alpha and p_col_to_target > config.direction_alpha:
            parents.append(col)
        elif p_target_to_col <= p_col_to_target:
            parents.append(col)
        elif abs(p_target_to_col - p_col_to_target) <= round(config.direction_alpha / 5, 2):
            parents.append(col)
        else:
            children.append(col)

    return (
        tuple(sorted(parents, key=_var_sort_key)),
        tuple(sorted(children, key=_var_sort_key)),
        tuple(sorted(removed, key=_var_sort_key)),
    )


def run_local_asi_search(
    df: pd.DataFrame,
    initial_targets: Sequence[str],
    candidates: Sequence[str],
    config: ASIDiscoveryConfig,
) -> Tuple[Dict[str, Tuple[str, ...]], Dict[str, Tuple[str, ...]], Tuple[str, ...]]:
    parent_dict: MutableMapping[str, Tuple[str, ...]] = {}
    child_dict: MutableMapping[str, Tuple[str, ...]] = {}
    searched: List[str] = []
    queue: List[str] = list(initial_targets)
    excluded: Set[str] = set()

    while queue:
        target = queue.pop(0)
        if target in searched:
            continue
        parents, children, _ = identify_parent_child(df, target, candidates, excluded, config)
        parent_dict[target] = parents
        child_dict[target] = children
        searched.append(target)

        excluded.add(target)
        excluded.update(children)
        for parent in parents:
            if parent not in searched and parent not in queue:
                queue.append(parent)

    return dict(parent_dict), dict(child_dict), tuple(searched)


def _ancestors_from_parent_dict(parent_dict: Mapping[str, Sequence[str]], target: str) -> Set[str]:
    ancestors: Set[str] = set()
    frontier = list(parent_dict.get(target, ()))
    while frontier:
        node = frontier.pop()
        if node in ancestors:
            continue
        ancestors.add(node)
        frontier.extend(parent_dict.get(node, ()))
    return ancestors


def _root_sources_from_estimated_graph(
    common_ancestors: Set[str],
    parent_dict: Mapping[str, Sequence[str]],
) -> Set[str]:
    if not common_ancestors:
        return set()
    roots: Set[str] = set()
    for node in common_ancestors:
        parents_inside = set(parent_dict.get(node, ())) & common_ancestors
        if not parents_inside:
            roots.add(node)
    return roots


def discover_case_structure(
    df: pd.DataFrame,
    observed_covariates: Sequence[str],
    treatment: str,
    outcome: str,
    config: ASIDiscoveryConfig,
) -> DiscoveryResult:
    outcome_candidates = list(observed_covariates) + [treatment]
    treatment_candidates = list(observed_covariates)

    y_parents, y_children, y_searched = run_local_asi_search(
        df=df,
        initial_targets=[outcome],
        candidates=outcome_candidates,
        config=config,
    )
    t_parents, t_children, t_searched = run_local_asi_search(
        df=df,
        initial_targets=[treatment],
        candidates=treatment_candidates,
        config=config,
    )

    parent_dict: Dict[str, Tuple[str, ...]] = {}
    child_dict: Dict[str, Tuple[str, ...]] = {}
    parent_dict.update(y_parents)
    parent_dict.update(t_parents)
    child_dict.update(y_children)
    child_dict.update(t_children)

    outcome_ancestors = _ancestors_from_parent_dict(parent_dict, outcome) & set(observed_covariates)
    treatment_ancestors = _ancestors_from_parent_dict(parent_dict, treatment) & set(observed_covariates)
    common_ancestors = outcome_ancestors & treatment_ancestors

    # Directional ANM scores can become ambiguous under the non-additive mechanisms
    # evaluated here.  For the controlled synthetic graphs, root confounding sources
    # are treatment-side source candidates that retain outcome-side dependence after
    # conditioning on treatment.  This separates root sources from instrument-like
    # parents of T while keeping the diagnostic tied to the ASI local search.
    root_source_pvalues: Dict[str, float] = {}
    root_sources: Set[str] = set()
    treatment_parent_candidates = set(parent_dict.get(treatment, ())) & set(observed_covariates)
    for candidate in sorted(treatment_parent_candidates, key=_var_sort_key):
        p_value = residual_independence_pvalue(df, candidate, outcome, [treatment], config)
        root_source_pvalues[candidate] = p_value
        if p_value < config.root_source_alpha:
            root_sources.add(candidate)

    if not root_sources:
        root_sources = _root_sources_from_estimated_graph(common_ancestors, parent_dict)

    return DiscoveryResult(
        parent_dict=parent_dict,
        child_dict=child_dict,
        searched_targets=tuple(sorted(set(y_searched) | set(t_searched), key=_var_sort_key)),
        predicted_outcome_ancestors=tuple(sorted(outcome_ancestors, key=_var_sort_key)),
        predicted_treatment_ancestors=tuple(sorted(treatment_ancestors, key=_var_sort_key)),
        predicted_common_ancestors=tuple(sorted(common_ancestors, key=_var_sort_key)),
        predicted_root_sources=tuple(sorted(root_sources, key=_var_sort_key)),
        root_source_candidate_pvalues=root_source_pvalues,
    )


def binary_metrics(
    true_set: Iterable[str],
    pred_set: Iterable[str],
    universe: Sequence[str],
) -> Dict[str, float]:
    true = set(true_set)
    pred = set(pred_set)
    universe_set = set(universe)
    tp = len(true & pred)
    fp = len(pred - true)
    fn = len(true - pred)
    tn = len(universe_set - true - pred)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(universe_set) if universe_set else 0.0
    y_true = [1 if v in true else 0 for v in universe]
    y_score = [1 if v in pred else 0 for v in universe]
    try:
        auc = float(roc_auc_score(y_true, y_score))
    except ValueError:
        auc = float("nan")
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": auc,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
    }
