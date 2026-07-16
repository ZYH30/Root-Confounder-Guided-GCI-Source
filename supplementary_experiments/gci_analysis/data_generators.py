"""Synthetic data generators for robustness experiments.

The functions in this module generate additional datasets on the same four DAGs used
in the original GCI synthetic experiments. They are intentionally separated from the
slow ASI implementation. The purpose is to evaluate whether the adjustment-set
principle remains useful when the structural equations deviate from the additive-noise
mechanism used in the initial synthetic data.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import pandas as pd

from .graph_cases import CASE_GRAPHS, CaseGraph


MECHANISM_LABELS: Mapping[str, str] = {
    "additive_sine": "Additive nonlinear",
    "multiplicative": "Multiplicative noise",
    "heteroscedastic": "Heteroscedastic noise",
    "interaction": "Interaction nonlinearity",
}

ANM_VIOLATING_MECHANISMS: Tuple[str, ...] = (
    "multiplicative",
    "heteroscedastic",
    "interaction",
)

ALL_ROBUSTNESS_MECHANISMS: Tuple[str, ...] = (
    "additive_sine",
    "multiplicative",
    "heteroscedastic",
    "interaction",
)


# Coefficients are copied from the original DataSetup.py definitions. The generated
# datasets differ only in how parent signals and noise are combined.
CASE_COEFFICIENTS: Mapping[int, Mapping[str, Mapping[str, float]]] = {
    1: {
        "X4": {"X1": 1.82},
        "t": {"X2": 0.75, "X3": 0.85},
        "X5": {"X3": 1.84},
        "y": {"t": 0.95, "X5": 0.80, "X4": 0.72},
        "X6": {"X5": 1.45},
        "X7": {"y": 1.18},
        "X8": {"X6": 0.74, "y": 0.95},
    },
    2: {
        "X4": {"X1": 1.95},
        "t": {"X2": 0.75, "X3": 0.85},
        "X5": {"X3": 1.05},
        "X6": {"t": 1.50},
        "X7": {"X5": 1.40},
        "X8": {"X4": 1.75},
        "y": {"X6": 0.70, "X7": 0.80, "X8": 1.50},
        "X9": {"X6": 0.55, "y": 0.88},
    },
    3: {
        "X4": {"X1": 1.82},
        "t": {"X2": 0.98, "X3": 0.62},
        "X5": {"X3": 1.84},
        "y": {"X4": 0.70, "t": 0.95, "X5": 1.35},
        "X6": {"t": 0.75, "y": 1.15},
        "X7": {"t": 0.62, "y": 0.85},
    },
    4: {
        "t": {"X1": 0.76, "X2": 0.82, "X3": 0.95, "X4": 1.06},
        "X5": {"X2": 1.84},
        "y": {"X3": 0.95, "X4": 0.82, "X5": 0.78, "t": 0.70},
        "X6": {"t": 1.35},
        "X7": {"X6": 0.62, "y": 1.05},
    },
}


@dataclass(frozen=True)
class GeneratorConfig:
    n_samples: int = 1000
    delta_t: float = 1.0
    root_scale: float = 1.0
    noise_scale: float = 1.0
    random_seed: int = 2026


def topological_order(case: CaseGraph) -> List[str]:
    """Return a deterministic topological order for the case graph."""
    nodes = sorted(case.nodes, key=_node_sort_key)
    parents = {node: set(case.parents(node)) for node in nodes}
    children = {node: set(case.children(node)) for node in nodes}
    ready = [node for node in nodes if not parents[node]]
    ready.sort(key=_node_sort_key)
    order: List[str] = []

    while ready:
        node = ready.pop(0)
        order.append(node)
        for child in sorted(children[node], key=_node_sort_key):
            parents[child].discard(node)
            if not parents[child] and child not in order and child not in ready:
                ready.append(child)
        ready.sort(key=_node_sort_key)

    if len(order) != len(nodes):
        missing = sorted(set(nodes) - set(order), key=_node_sort_key)
        raise ValueError(f"Graph for Case {case.case_id} is not acyclic. Remaining nodes: {missing}")
    return order


def _node_sort_key(node: str) -> Tuple[int, str]:
    if node.startswith("X") and node[1:].isdigit():
        return (0, f"{int(node[1:]):04d}")
    if node == "t":
        return (1, node)
    if node == "y":
        return (2, node)
    return (3, node)


def _parent_signal(parent_values: Sequence[np.ndarray], weights: Sequence[float], mechanism: str) -> np.ndarray:
    if not parent_values:
        raise ValueError("A non-root variable must have at least one parent.")
    transformed = [np.sin(w * v) for v, w in zip(parent_values, weights)]
    base = np.sum(transformed, axis=0)

    if mechanism == "interaction" and len(parent_values) >= 2:
        interaction = np.zeros_like(base)
        for i in range(len(parent_values)):
            for j in range(i + 1, len(parent_values)):
                interaction += np.tanh(weights[i] * parent_values[i] * weights[j] * parent_values[j])
        base = base + 0.25 * interaction
    return base


def _apply_noise(base: np.ndarray, eps: np.ndarray, mechanism: str, noise_scale: float) -> np.ndarray:
    if mechanism == "additive_sine":
        return base + noise_scale * eps
    if mechanism == "multiplicative":
        # Bounded multiplicative perturbation avoids numerical explosions while breaking
        # the additive-noise form used by ANM.
        return base * (1.0 + 0.45 * np.tanh(eps)) + 0.10 * noise_scale * eps
    if mechanism == "heteroscedastic":
        local_scale = 0.25 + 0.65 * np.abs(np.tanh(base))
        return base + noise_scale * local_scale * eps
    if mechanism == "interaction":
        return base + noise_scale * eps
    raise ValueError(f"Unknown mechanism: {mechanism}")


def _compute_node(
    node: str,
    values: Mapping[str, np.ndarray],
    eps: np.ndarray,
    case_id: int,
    mechanism: str,
    noise_scale: float,
) -> np.ndarray:
    coeffs = CASE_COEFFICIENTS[case_id].get(node)
    if coeffs is None:
        raise KeyError(f"No structural coefficients available for Case {case_id}, node {node}")
    parent_values = [values[parent] for parent in coeffs.keys()]
    weights = [float(weight) for weight in coeffs.values()]
    base = _parent_signal(parent_values, weights, mechanism)
    return _apply_noise(base, eps, mechanism, noise_scale)


def generate_case_dataset(
    case_id: int,
    mechanism: str,
    config: GeneratorConfig | None = None,
) -> pd.DataFrame:
    """Generate one synthetic dataset and its structural counterfactual outcome.

    The counterfactual column `y_delta` is computed under do(T=t-delta_t), with the
    same exogenous root variables and node-level noises as in the factual world. All
    descendants of T are recomputed in topological order.
    """
    if mechanism not in MECHANISM_LABELS:
        raise ValueError(f"Unknown mechanism {mechanism}. Available: {sorted(MECHANISM_LABELS)}")
    if config is None:
        config = GeneratorConfig()

    case = CASE_GRAPHS[case_id]
    rng = np.random.default_rng(config.random_seed + 1000 * case_id + 37 * list(MECHANISM_LABELS).index(mechanism))
    order = topological_order(case)

    values: MutableMapping[str, np.ndarray] = {}
    eps_by_node: Dict[str, np.ndarray] = {}

    for node in order:
        if not case.parents(node):
            values[node] = rng.normal(0.0, config.root_scale, size=config.n_samples)
        else:
            eps = rng.normal(0.0, 1.0, size=config.n_samples)
            eps_by_node[node] = eps
            values[node] = _compute_node(
                node=node,
                values=values,
                eps=eps,
                case_id=case_id,
                mechanism=mechanism,
                noise_scale=config.noise_scale,
            )

    descendants_of_t = case.descendants(case.treatment)
    cf_values: MutableMapping[str, np.ndarray] = {}
    for node in order:
        if node == case.treatment:
            cf_values[node] = values[node] - config.delta_t
        elif node in descendants_of_t:
            cf_values[node] = _compute_node(
                node=node,
                values=cf_values,
                eps=eps_by_node[node],
                case_id=case_id,
                mechanism=mechanism,
                noise_scale=config.noise_scale,
            )
        else:
            cf_values[node] = values[node]

    cols = list(case.observed_covariates) + [case.treatment, case.outcome]
    data = {col: values[col] for col in cols}
    data["y_delta"] = cf_values[case.outcome]
    df = pd.DataFrame(data)
    return df


def generate_all_case_datasets(
    mechanisms: Iterable[str],
    config: GeneratorConfig | None = None,
) -> Dict[Tuple[str, int], pd.DataFrame]:
    if config is None:
        config = GeneratorConfig()
    out: Dict[Tuple[str, int], pd.DataFrame] = {}
    for mechanism in mechanisms:
        for case_id in CASE_GRAPHS:
            out[(mechanism, case_id)] = generate_case_dataset(case_id, mechanism, config)
    return out
