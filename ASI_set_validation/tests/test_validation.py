from pathlib import Path
import json
import networkx as nx
import numpy as np
import pandas as pd
import pytest
from gci_validation.adjustment import select_adjustment,d_separated,adjustment_validity,recovery,graph_from_parents
from gci_validation.graph_cases import CASE_GRAPHS
from gci_validation.asi import ASIConfig,NumericalSettings,_moment_pvalue
from gci_validation.data_generators import generate_case_dataset,GeneratorConfig
from gci_validation.direction import analyze_direction

ROOT=Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('cid,expected',[(1,['X5']),(2,['X7']),(3,['X5']),(4,['X3','X4','X5'])])
def test_reference_adjustment(cid,expected):
    c=CASE_GRAPHS[cid];g=nx.DiGraph(c.edges);g.add_nodes_from(c.nodes)
    result=select_adjustment(g,c.observed_covariates)
    assert result['status']=='ok' and result['selected']==expected
    assert adjustment_validity(g,result['selected'])['valid']

@pytest.mark.parametrize('edges,z,expected',[
    ([('t','v'),('y','v')],[],True),
    ([('t','v'),('y','v')],['v'],False),
    ([('v','t'),('v','y')],[],False),
    ([('v','t'),('v','y')],['v'],True),
    ([('t','v'),('v','y')],['v'],True),
])
def test_d_separation(edges,z,expected):
    assert d_separated(nx.DiGraph(edges),'t','y',z)==expected

def test_cycle_is_explicit_failure():
    g=nx.DiGraph([('t','X1'),('X1','y'),('y','t')])
    assert select_adjustment(g,['X1'])['status']=='cyclic_input_graph'

def test_exhausted_search_is_not_empty_success():
    g=nx.DiGraph([('u','t'),('u','y')])
    assert select_adjustment(g,[])['status']=='no_admissible_set'

def test_empty_success():
    g=nx.DiGraph([('t','y')]);g.add_node('X1')
    result=select_adjustment(g,['X1'])
    assert result['status']=='ok' and result['selected']==[]

def test_outcome_distance_tie_break():
    g=nx.DiGraph([('X1','t'),('X1','X2'),('X2','y'),('t','y')])
    assert select_adjustment(g,['X1','X2'])['selected']==['X2']

def test_posttreatment_inclusion_is_invalid():
    g=nx.DiGraph([('t','y'),('t','X1')])
    assert not adjustment_validity(g,['X1'])['valid']

def test_unknown_parent_rejected():
    with pytest.raises(ValueError):graph_from_parents({'y':['unknown']},['X1'])

def test_failed_recovery_not_successful_empty_set():
    assert recovery([],None)['precision']==0
    assert recovery([],[])['precision']==1

def test_oof_partitions_and_cache():
    df=pd.DataFrame(np.random.default_rng(1).normal(size=(80,3)),columns=['X1','t','y'])
    engine=NumericalSettings(df,ASIConfig())
    for fit,val in engine.splits:assert not set(fit)&set(val)
    a=engine.residual_matrix(['X1']);b=engine.residual_matrix(['X1'])
    assert a is b and engine.fit_count==5 and np.isfinite(a).all()

@pytest.mark.parametrize('mechanism',['additive_sine','multiplicative','heteroscedastic','interaction'])
def test_generator_determinism(mechanism):
    a=generate_case_dataset(1,mechanism,GeneratorConfig(n_samples=40,random_seed=1))
    b=generate_case_dataset(1,mechanism,GeneratorConfig(n_samples=40,random_seed=1))
    assert a.equals(b) and np.isfinite(a.to_numpy()).all()

def test_primary_records_complete():
    files=list((ROOT/'results/validation').glob('*.json'))
    assert len(files)==160
    records=[json.loads(p.read_text()) for p in files]
    assert sum(x['structure']['true_valid'] for x in records)==131
    assert len({(x['mechanism'],x['case_id'],x['repeat']) for x in records})==160

def test_sensitivity_records_complete():
    records=[json.loads(p.read_text()) for p in (ROOT/'results/order_sensitivity').glob('*.json')]
    assert len(records)==240
    for q in [2,3,4]:
        rows=[x for x in records if x['max_order']==q]
        assert len(rows)==80 and sum(x['structure']['true_valid'] for x in rows)==63
