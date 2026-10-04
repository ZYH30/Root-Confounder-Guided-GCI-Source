"""Local recursive ASI with fixed GCM, ANM-HSIC and local CI settings."""
from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Set, Tuple
import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr
from sklearn.model_selection import KFold
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from .direction import analyze_direction

@dataclass(frozen=True)
class ASIConfig:
    cor_threshold: float = 0.20
    ci_alpha: float = 0.05
    root_source_alpha: float = 0.02
    max_condition_set: int = 3
    folds: int = 5
    spline_knots: int = 7
    hsic_margin: float = 0.005
    seed: int = 2026

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

def _moment_pvalue(rx,ry):
    product=np.asarray(rx)*np.asarray(ry)
    m=product.mean(); v=product.var()
    if v<=1e-14:return 1. if abs(m)<=1e-14 else 0.
    return float(2*norm.sf(np.sqrt(len(product))*abs(m)/np.sqrt(v)))

class NumericalSettings:
    def __init__(self,df,config):
        self.df=df.copy(); self.config=config
        self.names=list(df.columns); self.index={n:i for i,n in enumerate(self.names)}
        a=df.to_numpy(dtype=float)
        if not np.isfinite(a).all():raise ValueError('nonfinite_discovery_data')
        sd=a.std(0); self.a=(a-a.mean(0))/np.maximum(sd,1e-12)
        self.cache={};self.dir_cache={}
        self.ci_rows=[];self.direction_rows=[];self.guard_rows=[]
        self.fit_count=0;self.oof_rows_tested=0;self.fold_overlap=0
        degree=1
        self.degree=degree
        # Use the observed feature transform, then OOF conditional mean, never powers of residuals.
        ts=[]
        for j in range(len(self.names)):
            for power in range(1,degree+1):
                v=np.clip(self.a[:,j],-6,6)**power
                ts.append((v-v.mean())/max(v.std(),1e-12))
        self.targets=np.column_stack(ts)
        self.splits=list(KFold(n_splits=config.folds,shuffle=True,random_state=config.seed).split(a))


    def residual_matrix(self,cond):
        key=tuple(sorted(cond))
        if key in self.cache:return self.cache[key]
        prediction=np.empty_like(self.targets)
        if not key:
            for fit,val in self.splits:prediction[val]=self.targets[fit].mean(0)
        else:
            x=self.a[:,[self.index[c] for c in key]]
            for fit,val in self.splits:
                assert not np.intersect1d(fit,val).size
                model=make_pipeline(SplineTransformer(n_knots=self.config.spline_knots,degree=3,include_bias=False,extrapolation='linear'),StandardScaler(),RidgeCV(alphas=np.logspace(-4,1,8)))
                model.fit(x[fit],self.targets[fit]);prediction[val]=model.predict(x[val])
                self.fit_count+=1;self.oof_rows_tested+=len(val)
        self.cache[key]=self.targets-prediction
        return self.cache[key]

    def residual(self,name,cond,power=1):
        return self.residual_matrix(cond)[:,self.index[name]*self.degree+power-1]

    def ci_test(self,df,x,y,cond,config,seed_offset=0):
        if not cond:return _safe_spearman(df[x].to_numpy(),df[y].to_numpy())[1]
        deg=self.degree
        residuals=self.residual_matrix(cond)
        p=[]
        for a in range(deg):
            for b in range(deg):
                p.append(_moment_pvalue(residuals[:,self.index[x]*deg+a],residuals[:,self.index[y]*deg+b]))
        out=min(1.,len(p)*min(p))
        self.ci_rows.append(dict(x=x,y=y,conditioning=list(cond),p_value=out,n_moments=len(p)))
        return out

    def protected_local_parents(self,target,adjacent):
        # A conservative local collider-signature safeguard, not a global PC pass.
        # No true graph, role labels, or treatment/outcome identities enter this rule.
        protected={};n_pairs=max(1,len(adjacent)*(len(adjacent)-1)//2)
        for a,b in combinations(adjacent,2):
            transformed=[]
            for node in (a,b):
                z=np.clip(self.a[:,self.index[node]],-6.,6.)
                basis=np.column_stack([z**k for k in (1,2,3)])
                basis=(basis-basis.mean(0))/np.maximum(basis.std(0),1e-12)
                transformed.append(basis)
            p0=min(1.,9*min(_moment_pvalue(transformed[0][:,i],transformed[1][:,j]) for i in range(3) for j in range(3)))
            p1=_moment_pvalue(self.residual(a,[target]),self.residual(b,[target]))
            supported=bool(p0>.10 and p1<.01/n_pairs)
            self.guard_rows.append(dict(target=target,a=a,b=b,marginal_p=p0,conditional_p=p1,conditional_family_threshold=.01/n_pairs,protected=supported))
            if supported:
                protected.setdefault(a,[]).append(b);protected.setdefault(b,[]).append(a)
        return protected

    def audit(self):
        return dict(local_ci_guard_tests=self.guard_rows,ci_tests=self.ci_rows,direction_tests=self.direction_rows,conditional_oof_fits=self.fit_count,oof_validation_rows=self.oof_rows_tested,overlapping_fit_validation_indices=self.fold_overlap)

    def direction(self, x: str, y: str) -> str:
        pair = tuple(sorted((x,y)))
        swap = (x != pair[0])
        if pair not in self.dir_cache:
            self.dir_cache[pair] = analyze_direction(self.df[pair[0]].to_numpy(),
                                                    self.df[pair[1]].to_numpy(),
                                                    self.config.seed, self.config.hsic_margin)
        result = self.dir_cache[pair]
        label = result.direction
        score = float(result.score) if np.isfinite(result.score) else 0.0
        if swap:
            score = -score
            label = {'x_to_y':'y_to_x','y_to_x':'x_to_y'}.get(label,label)
        relation = 'child' if label == 'x_to_y' else 'parent'
        self.direction_rows.append(dict(target=x,candidate=y,score=score,
             decisive=label in ('x_to_y','y_to_x'),relation=relation,
             method='anm_hsic',source_result=result.to_dict()))
        return relation

    def _pc(self,df,target,candidates,excluded,config):
        usable=[c for c in candidates if c not in (set(excluded)|{target}) and c in df]
        screened=[c for c in usable if abs(_safe_spearman(df[target],df[c])[0])>=config.cor_threshold]
        screened.sort(key=lambda c:abs(_safe_spearman(df[target],df[c])[0]),reverse=True)
        adjacent=list(screened);removed=[]
        for col in screened:
            other=[c for c in adjacent if c!=col]; independent=False
            for size in range(1,min(config.max_condition_set,len(other))+1):
                for cond in combinations(other,size):
                    p=self.ci_test(df,target,col,cond,config)
                    if p>config.ci_alpha:independent=True;break
                if independent:break
            if independent and col in adjacent:adjacent.remove(col);removed.append(col)
        protected=self.protected_local_parents(target,adjacent)
        parents=[];children=[]
        for col in adjacent:
            relation=self.direction(target,col)
            if col in protected and relation=='child':
                self.direction_rows[-1].update(pre_guard_relation=relation,relation='parent',ci_guard_witnesses=protected[col])
                relation='parent'
            (children if relation=='child' else parents).append(col)
        return tuple(sorted(parents,key=_var_sort_key)),tuple(sorted(children,key=_var_sort_key)),tuple(sorted(removed,key=_var_sort_key))

def run_local_asi_search(
    df: pd.DataFrame,
    initial_targets: Sequence[str],
    candidates: Sequence[str],
    config: ASIConfig,
    engine: NumericalSettings,
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
        parents, children, _ = engine._pc(df, target, candidates, excluded, config)
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
    config: ASIConfig,
    engine: NumericalSettings,
) -> DiscoveryResult:
    outcome_candidates = list(observed_covariates) + [treatment]
    treatment_candidates = list(observed_covariates)

    y_parents, y_children, y_searched = run_local_asi_search(
        df=df,
        initial_targets=[outcome],
        candidates=outcome_candidates,
        config=config, engine=engine,
    )
    t_parents, t_children, t_searched = run_local_asi_search(
        df=df,
        initial_targets=[treatment],
        candidates=treatment_candidates,
        config=config, engine=engine,
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

    # Optional source-candidate diagnostic. The final constructor independently
    # computes graph-defined roots and never uses these diagnostic labels.
    root_source_pvalues: Dict[str, float] = {}
    root_sources: Set[str] = set()
    treatment_parent_candidates = set(parent_dict.get(treatment, ())) & set(observed_covariates)
    for candidate in sorted(treatment_parent_candidates, key=_var_sort_key):
        p_value = engine.ci_test(df, candidate, outcome, [treatment], config)
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

def discover(df: pd.DataFrame, covariates: Sequence[str], config: ASIConfig | None = None):
    """Discover the treatment/outcome local graph from observed columns only.

    Counterfactual labels and test data are excluded at entry. No true graph or
    true variable role is accepted by this interface.
    """
    config = config or ASIConfig()
    if config.max_condition_set < 1:
        raise ValueError('max_condition_set must be at least one')
    observed = df.loc[:, [*covariates,'t','y']]
    engine = NumericalSettings(observed, config)
    result = discover_case_structure(observed, covariates, 't', 'y', config, engine)
    return result, engine.audit()
