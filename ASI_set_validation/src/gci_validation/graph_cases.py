"""Graph specifications and adjustment-set definitions for the four synthetic GCI cases.

The goal of this module is deliberately modest: it records the known DAGs used by
DataSetup.py and exposes the root-source and adjustment-set relationships needed for
supplementary experiments. It does not call the ASI module and therefore avoids the slow
GCM/ANM identification routine.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple

Edge = Tuple[str, str]


@dataclass(frozen=True)
class CaseGraph:
    case_id: int
    label: str
    observed_covariates: Tuple[str, ...]
    edges: Tuple[Edge, ...]
    treatment: str = "t"
    outcome: str = "y"
    root_sources: Tuple[str, ...] = ()
    treatment_parents: Tuple[str, ...] = ()
    gci_valid_adjustment: Tuple[str, ...] = ()
    excluded_by_gci: Tuple[str, ...] = ()
    note: str = ""

    @property
    def nodes(self) -> Set[str]:
        nodes = set(self.observed_covariates) | {self.treatment, self.outcome}
        for u, v in self.edges:
            nodes.add(u)
            nodes.add(v)
        return nodes

    def parents(self, node: str) -> Set[str]:
        return {u for u, v in self.edges if v == node}

    def children(self, node: str) -> Set[str]:
        return {v for u, v in self.edges if u == node}

    def observed_covariate_parents(self, node: str) -> Tuple[str, ...]:
        """Observed covariate parents of ``node`` (treatment is excluded)."""
        parents = self.parents(node) & set(self.observed_covariates)
        return tuple(sorted(parents, key=_var_sort_key))

    def ancestors(self, node: str) -> Set[str]:
        out: Set[str] = set()
        frontier = list(self.parents(node))
        while frontier:
            cur = frontier.pop()
            if cur in out:
                continue
            out.add(cur)
            frontier.extend(self.parents(cur))
        return out

    def descendants(self, node: str) -> Set[str]:
        out: Set[str] = set()
        frontier = list(self.children(node))
        while frontier:
            cur = frontier.pop()
            if cur in out:
                continue
            out.add(cur)
            frontier.extend(self.children(cur))
        return out

    def ancestral_candidate_set(self) -> Tuple[str, ...]:
        """Observed non-treatment variables in the local ancestral closure of T or Y.

        Descendants of treatment are not removed here, because the purpose of this set
        is to show that common adjustment candidates originate from the local ancestral
        structure. The adjustment-set construction step then removes mediators,
        descendants, and unsafe colliders.
        """
        cand = (self.ancestors(self.treatment) | self.ancestors(self.outcome))
        cand = {v for v in cand if v in self.observed_covariates}
        return tuple(sorted(cand, key=_var_sort_key))


def _var_sort_key(x: str) -> Tuple[int, str]:
    if x.startswith("X") and x[1:].isdigit():
        return (0, f"{int(x[1:]):04d}")
    return (1, x)


CASE_GRAPHS: Mapping[int, CaseGraph] = {
    1: CaseGraph(
        case_id=1,
        label="Case A: pretreatment covariates only",
        observed_covariates=("X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"),
        edges=(
            ("X1", "X4"), ("X2", "t"), ("X3", "t"), ("X3", "X5"),
            ("t", "y"), ("X5", "y"), ("X4", "y"), ("X5", "X6"),
            ("y", "X7"), ("X6", "X8"), ("y", "X8"),
        ),
        root_sources=("X3",),
        treatment_parents=("X2", "X3"),
        gci_valid_adjustment=("X5",),
        excluded_by_gci=("X1", "X2", "X6", "X7", "X8"),
        note="X3 is redundant after conditioning on X5; X2 is instrument-like; X8 is a collider descendant.",
    ),
    2: CaseGraph(
        case_id=2,
        label="Case B: post-treatment mediator",
        observed_covariates=("X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9"),
        edges=(
            ("X1", "X4"), ("X2", "t"), ("X3", "t"), ("X3", "X5"),
            ("t", "X6"), ("X5", "X7"), ("X4", "X8"), ("X6", "y"),
            ("X7", "y"), ("X8", "y"), ("X6", "X9"), ("y", "X9"),
        ),
        root_sources=("X3",),
        treatment_parents=("X2", "X3"),
        gci_valid_adjustment=("X7",),
        excluded_by_gci=("X1", "X2", "X4", "X6", "X8", "X9"),
        note="X3 is redundant after conditioning on X7; X6 is a mediator; X9 is a collider descendant.",
    ),
    3: CaseGraph(
        case_id=3,
        label="Case C: post-treatment colliders",
        observed_covariates=("X1", "X2", "X3", "X4", "X5", "X6", "X7"),
        edges=(
            ("X1", "X4"), ("X2", "t"), ("X3", "t"), ("X3", "X5"),
            ("X4", "y"), ("t", "y"), ("X5", "y"), ("t", "X6"),
            ("y", "X6"), ("t", "X7"), ("y", "X7"),
        ),
        root_sources=("X3",),
        treatment_parents=("X2", "X3"),
        gci_valid_adjustment=("X5",),
        excluded_by_gci=("X1", "X2", "X6", "X7"),
        note="X3 is redundant after conditioning on X5; X6 and X7 are post-treatment colliders.",
    ),
    4: CaseGraph(
        case_id=4,
        label="Case D: multiple confounders and collider",
        observed_covariates=("X1", "X2", "X3", "X4", "X5", "X6", "X7"),
        edges=(
            ("X1", "t"), ("X2", "t"), ("X3", "t"), ("X4", "t"),
            ("X2", "X5"), ("X3", "y"), ("X4", "y"), ("X5", "y"),
            ("t", "y"), ("t", "X6"), ("X6", "X7"), ("y", "X7"),
        ),
        root_sources=("X2", "X3", "X4"),
        treatment_parents=("X1", "X2", "X3", "X4"),
        gci_valid_adjustment=("X3", "X4", "X5"),
        excluded_by_gci=("X1", "X6", "X7"),
        note="X2 is redundant after conditioning on X5; X1 is instrument-like; X6 and X7 are treatment descendants.",
    ),
}


ADJUSTMENT_SET_LABELS: Tuple[str, ...] = (
    "No adjustment",
    "All covariates",
    "Root-only",
    "Treatment-parent",
    "Outcome-parent",
    "GCI-guided valid",
)


def get_case_graph(case_id: int) -> CaseGraph:
    try:
        return CASE_GRAPHS[case_id]
    except KeyError as exc:
        raise ValueError(f"Unknown case_id {case_id}. Available cases: {sorted(CASE_GRAPHS)}") from exc


def adjustment_sets_for_case(case_id: int) -> Dict[str, Tuple[str, ...]]:
    case = get_case_graph(case_id)
    return {
        "No adjustment": tuple(),
        "All covariates": tuple(case.observed_covariates),
        "Root-only": tuple(case.root_sources),
        "Treatment-parent": tuple(case.treatment_parents),
        "Outcome-parent": case.observed_covariate_parents(case.outcome),
        "GCI-guided valid": tuple(case.gci_valid_adjustment),
    }


def fmt_vars(vars_: Sequence[str]) -> str:
    if not vars_:
        return "--"
    return ", ".join(vars_)
