"""Deterministic root-closure adjustment; the input graph is never repaired.

Selection uses only the supplied graph, not the true graph or effect-estimation
scores. Ground-truth validity is a separate evaluation operation.
"""
from __future__ import annotations
from itertools import combinations
import math
from typing import Iterable, Mapping, Sequence
import networkx as nx


def node_key(node: str) -> tuple[int, int | str]:
    return (0, int(node[1:])) if node.startswith('X') and node[1:].isdigit() else (1, node)


def sorted_nodes(nodes: Iterable[str]) -> list[str]:
    return sorted(nodes, key=node_key)


def graph_from_parents(parents: Mapping[str, Sequence[str]], covariates: Sequence[str], treatment='t', outcome='y') -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_nodes_from([*covariates, treatment, outcome])
    universe = set(g)
    for child, ps in parents.items():
        if child not in universe or not set(ps) <= universe:
            raise ValueError('Parent dictionary contains unknown variables')
        g.add_edges_from((p, child) for p in ps)
    return g


def backdoor_graph(g: nx.DiGraph, treatment='t') -> nx.DiGraph:
    b = g.copy()
    b.remove_edges_from(list(b.out_edges(treatment)))
    return b


def d_separated(g: nx.DiGraph, x: str, y: str, z: Iterable[str]) -> bool:
    """Ancestral moral-graph test, independently checked against NetworkX."""
    if not nx.is_directed_acyclic_graph(g):
        raise ValueError('d-separation requires a DAG')
    z = set(z)
    if x in z or y in z or not z <= set(g):
        raise ValueError('Invalid conditioning variables')
    retained = {x, y} | z
    for v in tuple(retained):
        retained |= nx.ancestors(g, v)
    sub = g.subgraph(retained)
    moral = nx.Graph()
    moral.add_nodes_from(sub)
    moral.add_edges_from(sub.edges())
    for v in sub:
        moral.add_edges_from(combinations(sub.predecessors(v), 2))
    moral.remove_nodes_from(z)
    return not nx.has_path(moral, x, y)


def adjustment_validity(g: nx.DiGraph, z: Iterable[str], treatment='t', outcome='y') -> dict:
    z = set(z)
    forbidden = z & nx.descendants(g, treatment)
    separated = d_separated(backdoor_graph(g, treatment), treatment, outcome, z)
    return {'valid': not forbidden and separated, 'forbidden': sorted_nodes(forbidden),
            'backdoor_blocked': separated}


def select_adjustment(g: nx.DiGraph, covariates: Sequence[str], treatment='t', outcome='y') -> dict:
    """Enumerate by size, then sum of backdoor distances to Y, then node IDs.

    Empty sets can be successful outputs. Cyclic graphs and exhausted searches
    produce explicit failures rather than arbitrary graph deletion or fallback.
    """
    base = {'status': None, 'roots': [], 'candidates': [], 'closure': [],
            'selected': None, 'min_valid_alternatives': [], 'subsets_tested': 0,
            'distance_sum': None, 'cycle_edges': []}
    if not nx.is_directed_acyclic_graph(g):
        base.update(status='cyclic_input_graph', cycle_edges=[list(e) for e in nx.find_cycle(g)])
        return base
    x = set(covariates)
    b = backdoor_graph(g, treatment)
    common = nx.ancestors(b, treatment) & nx.ancestors(b, outcome) & x
    roots = {v for v in common if not (set(b.predecessors(v)) & common)}
    candidates = ((nx.ancestors(g, treatment) | nx.ancestors(g, outcome)) & x) - nx.descendants(g, treatment)
    downstream = set(roots)
    for r in roots:
        downstream |= nx.descendants(g, r)
    closure = sorted_nodes(downstream & candidates)
    base.update(roots=sorted_nodes(roots), candidates=sorted_nodes(candidates), closure=closure)
    distances = dict(nx.single_target_shortest_path_length(b, outcome))
    def tie_key(z: tuple[str, ...]):
        return (sum(distances.get(v, math.inf) for v in z), tuple(node_key(v) for v in z))
    for size in range(len(closure) + 1):
        valid = []
        for z in combinations(closure, size):
            base['subsets_tested'] += 1
            if d_separated(b, treatment, outcome, z):
                valid.append(z)
        if valid:
            valid.sort(key=tie_key)
            chosen = valid[0]
            score = tie_key(chosen)[0]
            base.update(status='ok', selected=list(chosen),
                        min_valid_alternatives=[list(z) for z in valid],
                        distance_sum=score if math.isfinite(score) else 'infinity')
            return base
    base['status'] = 'no_admissible_set'
    return base


def recovery(true: Iterable[str], selected: Iterable[str] | None) -> dict:
    """Failed construction is not a valid empty adjustment set; scores are zero."""
    true = set(true)
    pred = set(selected) if selected is not None else set()
    tp, fp, fn = len(true & pred), len(pred - true), len(true - pred)
    precision = tp / (tp + fp) if (tp + fp) else float(not true and selected is not None)
    recall = tp / (tp + fn) if (tp + fn) else float(selected is not None)
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float(selected is not None)
    return dict(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall, f1=f1,
                exact_match=selected is not None and pred == true)
